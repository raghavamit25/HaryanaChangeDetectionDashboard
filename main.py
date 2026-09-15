"""
Pixel Change Detection API
--------------------------
Serves before/after Sentinel-2 imagery plus a computed change map to a
public-facing dashboard. Designed to run unattended on a server, so it
authenticates to Google Earth Engine with a service account rather than
the interactive `ee.Authenticate()` flow (which needs a human + browser
and cannot run on a headless server).

Change detection logic:
  - "urban"      -> difference in Dynamic World's 'built' class probability.
                     Dynamic World is Google's pretrained deep-learning land
                     cover model (a fully convolutional neural net) already
                     served through Earth Engine at 10 m resolution, so this
                     gives model-based urban change detection with zero
                     training required.
  - "vegetation" -> difference in NDVI (vegetation loss/gain, e.g. deforestation).
  - "water"      -> difference in NDWI (waterbody loss/gain).

Each mode returns: a true-color "before" image, a true-color "after" image,
a transparent-background change-map overlay (red = decrease, green =
increase), and area statistics in hectares.
"""

import os
import sys
import json
import time
import base64
import hashlib
import asyncio
import tempfile
import urllib.request
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# fcntl is POSIX-only; the earthengine-api imports it transitively on some
# platforms. Mock it so the service also boots on Windows dev machines.
try:
    import fcntl  # noqa: F401
except ImportError:
    import types
    fcntl_mock = types.ModuleType("fcntl")
    fcntl_mock.ioctl = lambda *args, **kwargs: 0
    sys.modules["fcntl"] = fcntl_mock

import ee
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account

# ------------------------------------------------------------------
# 1. Earth Engine auth — service account, NOT interactive
# ------------------------------------------------------------------
# Required for anything that runs on a server for the general public:
#   GEE_PROJECT_ID              GCP project registered for Earth Engine
#   GEE_SERVICE_ACCOUNT_EMAIL   e.g. cd-bot@my-project.iam.gserviceaccount.com
#   GEE_PRIVATE_KEY_JSON_B64    base64-encoded contents of the service
#                                account's JSON key file
#
# On Cloud Run / GCE / Cloud Functions you can instead grant the Earth
# Engine role to the instance's default service account and skip the key
# entirely — ee.Initialize() will pick up Application Default Credentials.
#
# GEE_ALLOW_INTERACTIVE_AUTH=1 is an escape hatch for local development
# only; it must never be set in production, since ee.Authenticate() opens
# a browser window and blocks forever on a headless server.

GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID", "change-detection-haryana")


def _init_earth_engine():
    project_id = os.environ.get("GEE_PROJECT", "change-detection-haryana")
    creds_raw = os.environ.get("GEE_CREDENTIALS_JSON")

    if not creds_raw:
        try:
            ee.Initialize(project=project_id)
            print("Earth Engine initialized via default environment.")
            return
        except Exception as exc:
            raise RuntimeError("GEE_CREDENTIALS_JSON variable is missing.") from exc

    try:
        info = json.loads(creds_raw)

        # Case A: Service Account JSON Key
        if info.get("type") == "service_account" or ("private_key" in info and "client_email" in info):
            credentials = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/earthengine"]
            )
            ee.Initialize(credentials=credentials, project=project_id)
            print("Earth Engine initialized via Service Account.")
            return

        # Case B: Local user credentials (gcloud / earthengine authenticate style)
        # Write to Earth Engine's expected credentials path in the container
        ee_dir = os.path.expanduser("~/.config/earthengine")
        os.makedirs(ee_dir, exist_ok=True)
        creds_file_path = os.path.join(ee_dir, "credentials")

        with open(creds_file_path, "w", encoding="utf-8") as f:
            json.dump(info, f)

        # ee.Initialize() will automatically pick up ~/.config/earthengine/credentials
        # using the SDK's internal OAuth client credentials
        ee.Initialize(project=project_id)
        print("Earth Engine initialized successfully via user credentials file.")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Earth Engine with GEE_CREDENTIALS_JSON: {e}")
_init_earth_engine()

# ------------------------------------------------------------------
# 2. FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="Pixel Change Detection Engine")

app.add_middleware(
    CORSMiddleware,
    # Lock this down to your dashboard's real domain before going live —
    # "*" is fine for local development only.
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ChangeType = Literal["urban", "vegetation", "water"]


class ChangeRequest(BaseModel):
    aoi: list = Field(..., description="Closed ring of [lng, lat] pairs")
    startDate: str
    endDate: str
    changeType: ChangeType = "urban"

    class Config:
        json_schema_extra = {
            "example": {
                "aoi": [[76.9, 28.4], [77.1, 28.4], [77.1, 28.6], [76.9, 28.6], [76.9, 28.4]],
                "startDate": "2019-01-01",
                "endDate": "2024-01-01",
                "changeType": "urban",
            }
        }


# Per-index defaults: which bands to diff, how big a change has to be
# before it counts (filters out normal seasonal / sensor noise), and the
# thumbnail dimension used for the two RGB previews.
INDEX_CONFIG = {
    "urban": {"threshold": 0.30, "label": "built-up probability"},
    "vegetation": {"threshold": 0.15, "label": "NDVI"},
    "water": {"threshold": 0.15, "label": "NDWI"},
}

THUMB_DIMENSIONS = 1024
MAX_AOI_AREA_KM2 = 3000  # keep public requests fast and within GEE quotas
CACHE_TTL_SECONDS = 6 * 3600

_cache: dict[str, tuple[float, dict]] = {}


def _cache_key(payload: ChangeRequest) -> str:
    raw = json.dumps(
        {"aoi": payload.aoi, "s": payload.startDate, "e": payload.endDate, "t": payload.changeType},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


# ------------------------------------------------------------------
# 3. Earth Engine helpers (all synchronous — run via asyncio.to_thread)
# ------------------------------------------------------------------
def _safe_mean(collection: "ee.ImageCollection", band: str) -> "ee.Image":
    """Mean of `band` over a collection, or an all-zero fallback image if
    the collection is empty (e.g. no Dynamic World coverage for a date/AOI)."""
    count = collection.size()
    return ee.Image(
        ee.Algorithms.If(
            count.gt(0),
            collection.select(band).mean(),
            ee.Image.constant(0).rename(band),
        )
    )


def _s2_collection(geometry: "ee.Geometry", target_date: str, window_days: int = 15):
    start = ee.Date(target_date).advance(-window_days, "day")
    end = ee.Date(target_date).advance(window_days, "day")
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start, end)
    )
    filtered = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
    return ee.ImageCollection(ee.Algorithms.If(filtered.size().gt(0), filtered, col)), start, end


def _composite_and_index(geometry: "ee.Geometry", target_date: str, change_type: ChangeType):
    s2_col, start, end = _s2_collection(geometry, target_date)
    rgb = s2_col.median().clip(geometry)

    if change_type == "vegetation":
        index_img = rgb.normalizedDifference(["B8", "B4"]).rename("IDX")
    elif change_type == "water":
        index_img = rgb.normalizedDifference(["B3", "B8"]).rename("IDX")
    else:  # urban
        dw_col = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(geometry)
            .filterDate(start, end)
        )
        index_img = _safe_mean(dw_col, "built").rename("IDX").clip(geometry)

    return rgb, index_img


def _run_change_detection(payload: ChangeRequest) -> dict:
    geometry = ee.Geometry.Polygon([payload.aoi])

    area_km2 = geometry.area(maxError=100).divide(1e6).getInfo()
    if area_km2 > MAX_AOI_AREA_KM2:
        raise HTTPException(
            status_code=400,
            detail=f"Selected area is {area_km2:,.0f} km² — please zoom in to under "
                   f"{MAX_AOI_AREA_KM2:,} km² so the analysis stays fast and accurate.",
        )

    cfg = INDEX_CONFIG[payload.changeType]
    threshold = cfg["threshold"]

    rgb_before, idx_before = _composite_and_index(geometry, payload.startDate, payload.changeType)
    rgb_after, idx_after = _composite_and_index(geometry, payload.endDate, payload.changeType)

    diff = idx_after.subtract(idx_before).rename("DIFF")
    change_mask = diff.abs().gt(threshold)
    masked_diff = diff.updateMask(change_mask)

    # Red = decrease in the index, green = increase. Pixels below the
    # threshold stay masked -> transparent in the PNG, so the overlay only
    # highlights meaningful change.
    overlay = masked_diff.visualize(
        min=-0.6, max=0.6, palette=["b91c1c", "f3f4f6", "16a34a"]
    )

    rgb_vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.2}

    url_before = rgb_before.visualize(**rgb_vis).getThumbURL(
        {"dimensions": THUMB_DIMENSIONS, "format": "png", "region": geometry}
    )
    url_after = rgb_after.visualize(**rgb_vis).getThumbURL(
        {"dimensions": THUMB_DIMENSIONS, "format": "png", "region": geometry}
    )
    url_overlay = overlay.getThumbURL(
        {"dimensions": THUMB_DIMENSIONS, "format": "png", "region": geometry}
    )

    pixel_area = ee.Image.pixelArea()
    stats = ee.Dictionary(
        {
            "increase_m2": pixel_area.updateMask(diff.gt(threshold)).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=geometry, scale=10, maxPixels=1e9, bestEffort=True
            ).get("area", 0),
            "decrease_m2": pixel_area.updateMask(diff.lt(-threshold)).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=geometry, scale=10, maxPixels=1e9, bestEffort=True
            ).get("area", 0),
            "total_m2": pixel_area.reduceRegion(
                reducer=ee.Reducer.sum(), geometry=geometry, scale=10, maxPixels=1e9, bestEffort=True
            ).get("area", 0),
        }
    ).getInfo()

    def _dl(url: str) -> str:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return base64.b64encode(resp.read()).decode("utf-8")

    b64_before = _dl(url_before)
    b64_after = _dl(url_after)
    b64_overlay = _dl(url_overlay)

    increase_ha = (stats.get("increase_m2") or 0) / 10_000
    decrease_ha = (stats.get("decrease_m2") or 0) / 10_000
    total_ha = (stats.get("total_m2") or 0) / 10_000
    percent_changed = ((increase_ha + decrease_ha) / total_ha * 100) if total_ha else 0

    return {
        "status": "success",
        "changeType": payload.changeType,
        "indexLabel": cfg["label"],
        "imgBefore": f"data:image/png;base64,{b64_before}",
        "imgAfter": f"data:image/png;base64,{b64_after}",
        "changeOverlay": f"data:image/png;base64,{b64_overlay}",
        "stats": {
            "totalAreaHa": round(total_ha, 1),
            "increaseHa": round(increase_ha, 1),
            "decreaseHa": round(decrease_ha, 1),
            "percentChanged": round(percent_changed, 2),
        },
    }


# ------------------------------------------------------------------
# 4. Endpoint
# ------------------------------------------------------------------
@app.post("/api/change-detection")
async def change_detection(payload: ChangeRequest):
    key = _cache_key(payload)
    cached = _cache.get(key)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        # ee's Python client is synchronous / blocking network I/O — run it
        # off the event loop so one slow request doesn't stall every other
        # visitor to the dashboard.
        result = await asyncio.to_thread(_run_change_detection, payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _cache[key] = (time.time(), result)
    return result


@app.get("/api/health")
async def health():
    return {"status": "ok", "project": GEE_PROJECT_ID}


# Serve the dashboard from this same app so the whole thing is one
# deployable service with one URL — handy for demos. Put index.html next
# to this file (see deployment notes).
_FRONTEND_PATH = Path(__file__).parent / "index.html"


@app.get("/")
async def serve_dashboard():
    if _FRONTEND_PATH.exists():
        return FileResponse(_FRONTEND_PATH)
    return {"detail": "index.html not found next to main.py — see deployment notes."}


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI Server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
