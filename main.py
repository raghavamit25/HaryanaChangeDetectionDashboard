"""
Pixel Change Detection API with PDF Reporting & Haryana Map Constraints
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

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fpdf import FPDF

# Windows mock for fcntl
try:
    import fcntl  # noqa: F401
except ImportError:
    import types
    fcntl_mock = types.ModuleType("fcntl")
    fcntl_mock.ioctl = lambda *args, **kwargs: 0
    sys.modules["fcntl"] = fcntl_mock

import ee
from google.oauth2 import service_account

GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID", "change-detection-haryana")

# ------------------------------------------------------------------
# 1. Earth Engine Authentication
# ------------------------------------------------------------------
def _init_earth_engine():
    project_id = os.environ.get("GEE_PROJECT", GEE_PROJECT_ID)
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

        # Service Account JSON Key
        if info.get("type") == "service_account" or ("private_key" in info and "client_email" in info):
            credentials = service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/earthengine"]
            )
            ee.Initialize(credentials=credentials, project=project_id)
            print("Earth Engine initialized via Service Account.")
            return

        # Local user credentials fallback
        ee_dir = os.path.expanduser("~/.config/earthengine")
        os.makedirs(ee_dir, exist_ok=True)
        creds_file_path = os.path.join(ee_dir, "credentials")

        with open(creds_file_path, "w", encoding="utf-8") as f:
            json.dump(info, f)

        ee.Initialize(project=project_id)
        print("Earth Engine initialized successfully via user credentials file.")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Earth Engine: {e}")

_init_earth_engine()

# ------------------------------------------------------------------
# 2. FastAPI Config & Models
# ------------------------------------------------------------------
app = FastAPI(title="Haryana Change Detection Engine")

app.add_middleware(
    CORSMiddleware,
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

class ReportRequest(BaseModel):
    startDate: str
    endDate: str
    changeType: str
    indexLabel: str
    stats: dict
    imgBefore: str
    imgAfter: str
    changeOverlay: str

INDEX_CONFIG = {
    "urban": {"threshold": 0.30, "label": "Built-up Probability (Dynamic World)"},
    "vegetation": {"threshold": 0.15, "label": "NDVI"},
    "water": {"threshold": 0.15, "label": "NDWI"},
}

THUMB_DIMENSIONS = 1024
MAX_AOI_AREA_KM2 = 5000
CACHE_TTL_SECONDS = 6 * 3600
_cache: dict[str, tuple[float, dict]] = {}

def _cache_key(payload: ChangeRequest) -> str:
    raw = json.dumps(
        {"aoi": payload.aoi, "s": payload.startDate, "e": payload.endDate, "t": payload.changeType},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()

# ------------------------------------------------------------------
# 3. GEE Image Computation
# ------------------------------------------------------------------
def _safe_mean(collection: "ee.ImageCollection", band: str) -> "ee.Image":
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
            detail=f"Selected area is {area_km2:,.0f} km² — please restrict to under {MAX_AOI_AREA_KM2:,} km².",
        )

    cfg = INDEX_CONFIG[payload.changeType]
    threshold = cfg["threshold"]

    rgb_before, idx_before = _composite_and_index(geometry, payload.startDate, payload.changeType)
    rgb_after, idx_after = _composite_and_index(geometry, payload.endDate, payload.changeType)

    diff = idx_after.subtract(idx_before).rename("DIFF")
    change_mask = diff.abs().gt(threshold)
    masked_diff = diff.updateMask(change_mask)

    # Bright Red (#EF4444) for loss/decrease, Bright Green (#22C55E) for gain/increase
    overlay = masked_diff.visualize(
        min=-0.6, max=0.6, palette=["ef4444", "00000000", "22c55e"]
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

    # Measurements
    total_m2 = stats.get("total_m2") or 0
    increase_m2 = stats.get("increase_m2") or 0
    decrease_m2 = stats.get("decrease_m2") or 0

    total_ha = total_m2 / 10_000
    increase_ha = increase_m2 / 10_000
    decrease_ha = decrease_m2 / 10_000

    total_km2 = total_m2 / 1_000_000
    increase_km2 = increase_m2 / 1_000_000
    decrease_km2 = decrease_m2 / 1_000_000

    changed_m2 = increase_m2 + decrease_m2
    percent_changed = (changed_m2 / total_m2 * 100) if total_m2 else 0

    return {
        "status": "success",
        "changeType": payload.changeType,
        "indexLabel": cfg["label"],
        "imgBefore": f"data:image/png;base64,{b64_before}",
        "imgAfter": f"data:image/png;base64,{b64_after}",
        "changeOverlay": f"data:image/png;base64,{b64_overlay}",
        "stats": {
            "totalAreaKm2": round(total_km2, 3),
            "increaseKm2": round(increase_km2, 3),
            "decreaseKm2": round(decrease_km2, 3),
            "totalAreaHa": round(total_ha, 1),
            "increaseHa": round(increase_ha, 1),
            "decreaseHa": round(decrease_ha, 1),
            "percentChanged": round(percent_changed, 2),
        },
    }

# ------------------------------------------------------------------
# 4. PDF Generation Helper
# ------------------------------------------------------------------
def _build_pdf_report(data: ReportRequest) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(0, 10, "Satellite Pixel Change Detection Report", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(107, 114, 128)
    pdf.cell(0, 6, f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} UTC | Region: Haryana, India", ln=True)
    pdf.ln(4)

    # Metadata & Parameters Box
    pdf.set_fill_color(243, 244, 246)
    pdf.rect(10, pdf.get_y(), 190, 22, "F")
    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(45, 6, f"Mode: {data.changeType.capitalize()}")
    pdf.cell(65, 6, f"Baseline Date: {data.startDate}")
    pdf.cell(65, 6, f"Comparison Date: {data.endDate}", ln=True)

    pdf.set_x(12)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Metric: {data.indexLabel}", ln=True)
    pdf.ln(8)

    # Statistics Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(0, 8, "Quantitative Change Summary", ln=True)

    stats = data.stats
    pdf.set_font("Helvetica", "", 10)
    col_w = 47.5
    h = 9
    pdf.set_fill_color(229, 231, 235)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(col_w, h, "Metric", border=1, fill=True)
    pdf.cell(col_w, h, "Square Kilometers", border=1, fill=True)
    pdf.cell(col_w, h, "Hectares", border=1, fill=True)
    pdf.cell(col_w, h, "Relative %", border=1, fill=True, ln=True)

    pdf.set_font("Helvetica", "", 9)
    rows = [
        ("Total AOI Area", f"{stats.get('totalAreaKm2', 0):,} km2", f"{stats.get('totalAreaHa', 0):,} ha", "100.0%"),
        ("Net Increase (Gain)", f"{stats.get('increaseKm2', 0):,} km2", f"{stats.get('increaseHa', 0):,} ha", f"{round((stats.get('increaseKm2', 0) / max(stats.get('totalAreaKm2', 1), 1e-6)) * 100, 2)}%"),
        ("Net Decrease (Loss)", f"{stats.get('decreaseKm2', 0):,} km2", f"{stats.get('decreaseHa', 0):,} ha", f"{round((stats.get('decreaseKm2', 0) / max(stats.get('totalAreaKm2', 1), 1e-6)) * 100, 2)}%"),
        ("Total Changed Area", f"{round(stats.get('increaseKm2', 0) + stats.get('decreaseKm2', 0), 3):,} km2", f"{round(stats.get('increaseHa', 0) + stats.get('decreaseHa', 0), 1):,} ha", f"{stats.get('percentChanged', 0)}%"),
    ]

    for label, km2_val, ha_val, pct_val in rows:
        pdf.cell(col_w, h, label, border=1)
        pdf.cell(col_w, h, km2_val, border=1)
        pdf.cell(col_w, h, ha_val, border=1)
        pdf.cell(col_w, h, pct_val, border=1, ln=True)

    pdf.ln(8)

    # Save images to temp files and append them to PDF
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Visual Change Evidence", ln=True)

    with tempfile.TemporaryDirectory() as td:
        p_before = os.path.join(td, "before.png")
        p_after = os.path.join(td, "after.png")
        p_diff = os.path.join(td, "diff.png")

        with open(p_before, "wb") as f:
            f.write(base64.b64decode(data.imgBefore.split(",")[1]))
        with open(p_after, "wb") as f:
            f.write(base64.b64decode(data.imgAfter.split(",")[1]))
        with open(p_diff, "wb") as f:
            f.write(base64.b64decode(data.changeOverlay.split(",")[1]))

        y = pdf.get_y()
        img_w = 60
        pdf.image(p_before, x=10, y=y, w=img_w)
        pdf.image(p_after, x=75, y=y, w=img_w)
        pdf.image(p_diff, x=140, y=y, w=img_w)

        pdf.set_y(y + 62)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_x(10)
        pdf.cell(img_w, 5, "Baseline (Before)", align="C")
        pdf.set_x(75)
        pdf.cell(img_w, 5, "Comparison (After)", align="C")
        pdf.set_x(140)
        pdf.cell(img_w, 5, "Red=Loss / Green=Gain", align="C")

    return bytes(pdf.output())

# ------------------------------------------------------------------
# 5. API Endpoints
# ------------------------------------------------------------------
@app.post("/api/change-detection")
async def change_detection(payload: ChangeRequest):
    key = _cache_key(payload)
    cached = _cache.get(key)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        result = await asyncio.to_thread(_run_change_detection, payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _cache[key] = (time.time(), result)
    return result

@app.post("/api/download-report")
async def download_report(payload: ReportRequest):
    try:
        pdf_bytes = await asyncio.to_thread(_build_pdf_report, payload)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=haryana_change_report_{int(time.time())}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

@app.get("/api/health")
async def health():
    return {"status": "ok", "project": GEE_PROJECT_ID}

_FRONTEND_PATH = Path(__file__).parent / "index.html"

@app.get("/")
async def serve_dashboard():
    if _FRONTEND_PATH.exists():
        return FileResponse(_FRONTEND_PATH)
    return {"detail": "index.html not found next to main.py"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)