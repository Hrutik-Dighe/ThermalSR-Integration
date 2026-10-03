"""
ThermalSR Backend — Pure Python/OpenCV  +  Optional MATLAB Engine
==================================================================
Two processing backends share one FastAPI server:

  Python backend  → POST /process
  MATLAB backend  → POST /process/matlab  (requires matlab.engine)

12-panel output (matches MATLAB "MULTI IMAGE PROCESSING SYSTEM"):
  1. Original          7.  Prewitt Edge
  2. Preprocessed      8.  Histogram EQ
  3. Low Resolution    9.  Threshold
  4. Super Resolution  10. Morphology
  5. Canny Edge        11. FFT
  6. Sobel Edge        12. Heatmap

All results saved locally to ./outputs/<timestamp>/

Run (from project root):
  pip install -r requirements.txt
  uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import re
import shutil
import tempfile

import numpy as np
import cv2
from scipy.ndimage import median_filter
from skimage import exposure, morphology
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

import base64, io, logging, os, json
from pathlib import Path
from datetime import datetime
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("thermalsr")

# ── Check MATLAB availability at startup ──────────
MATLAB_AVAILABLE = False
try:
    import matlab.engine as _me
    MATLAB_AVAILABLE = True
    log.info("matlab.engine found — MATLAB backend is available")
except ImportError:
    log.info("matlab.engine not found — MATLAB backend is disabled")

# ── Directory layout ───────────────────────────────
# backend/main.py lives one level below the project root.
ROOT_DIR    = Path(__file__).parent.parent   # ThermalSR_Integrated/
WEB_DIR     = ROOT_DIR / "frontend"          # HTML pages + CSS
SAMPLES_DIR = ROOT_DIR / "samples"          # Sample images
MATLAB_DIR  = ROOT_DIR / "matlab"           # MATLAB scripts
OUTPUTS_DIR = ROOT_DIR / "outputs"          # Saved run outputs
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ThermalSR API — Multi Image Processing System", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved output files statically
app.mount("/outputs",     StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
# Serve CSS assets
app.mount("/css",         StaticFiles(directory=str(WEB_DIR / "css")), name="css")
# Serve sample images
app.mount("/samples-img", StaticFiles(directory=str(SAMPLES_DIR)), name="samples-img")


# ══════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════

def _get_cmap(name: str):
    """Resolve a colormap name across matplotlib versions (cm.get_cmap was removed in 3.9)."""
    if hasattr(cm, "get_cmap"):  # matplotlib < 3.9
        return cm.get_cmap(name)
    return matplotlib.colormaps[name]  # matplotlib >= 3.6


def arr_to_b64(arr: np.ndarray, colormap: str = None) -> str:
    """Convert float [0,1] numpy array to base64 PNG data-URL."""
    arr = np.clip(arr, 0, 1)
    if colormap:
        rgba = _get_cmap(colormap)(arr)
        img8 = (rgba[:, :, :3] * 255).astype(np.uint8)
    else:
        if arr.ndim == 2:
            img8 = (arr * 255).astype(np.uint8)
        else:
            img8 = (arr * 255).astype(np.uint8)
    pil = Image.fromarray(img8)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def save_image(arr: np.ndarray, path: Path, colormap: str = None):
    """Save a float [0,1] array as PNG to disk."""
    arr = np.clip(arr, 0, 1)
    if colormap:
        rgba = _get_cmap(colormap)(arr)
        img8 = (rgba[:, :, :3] * 255).astype(np.uint8)
    else:
        if arr.ndim == 2:
            img8 = (arr * 255).astype(np.uint8)
        else:
            img8 = (arr * 255).astype(np.uint8)
    Image.fromarray(img8).save(str(path))


def normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


# ══════════════════════════════════════════════════
# PYTHON PIPELINE  (12 panels)
# ══════════════════════════════════════════════════

def run_pipeline(
    img_bgr: np.ndarray,
    edge_weight: float    = 0.6,
    detail_weight: float  = 0.4,
    sharpen_amount: float = 1.8,
    lr_scale: float       = 0.4,
    median_size: int      = 5,
    clahe_clip: float     = 0.02,
    save_folder: str      = None,
) -> dict:

    # ── Resize if too large ──────────────────────
    h, w = img_bgr.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    h, w = img_bgr.shape[:2]
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float64) / 255.0

    # 1. ORIGINAL
    original = img_float.copy()

    # 2. PREPROCESSED — grayscale → median → CLAHE → jet
    gray = np.dot(img_float[..., :3], [0.2989, 0.5870, 0.1140])
    gray = median_filter(gray, size=median_size)
    gray = exposure.equalize_adapthist(gray, clip_limit=clahe_clip)
    preprocessed_vis = gray

    # 3. LOW RESOLUTION
    lh, lw = int(h * lr_scale), int(w * lr_scale)
    low    = cv2.resize(gray, (lw, lh), interpolation=cv2.INTER_CUBIC)
    low_res = cv2.resize(low, (w, h),  interpolation=cv2.INTER_CUBIC)

    b1 = cv2.GaussianBlur(low_res, (0, 0), 1)
    b2 = cv2.GaussianBlur(low_res, (0, 0), 2)
    detail = (low_res - b1) + (b1 - b2)

    # 4. SUPER RESOLUTION
    g8 = (gray * 255).astype(np.uint8)
    canny_raw      = cv2.Canny(g8, 50, 150).astype(np.float64) / 255.0
    canny_weighted = canny_raw * np.mean(gray) * 1.5

    SR = low_res + edge_weight * canny_weighted + detail_weight * detail
    SR = normalize(SR)
    blur_sr = cv2.GaussianBlur(SR, (0, 0), 2)
    SR = np.clip(SR + sharpen_amount * (SR - blur_sr), 0, 1)

    # 5. CANNY EDGE
    canny_edge = canny_raw

    # 6. SOBEL EDGE
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edge = normalize(np.hypot(sobelx, sobely))

    # 7. PREWITT EDGE
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)
    prewitt_edge = normalize(np.hypot(
        cv2.filter2D(gray, cv2.CV_64F, kernelx),
        cv2.filter2D(gray, cv2.CV_64F, kernely),
    ))

    # 8. HISTOGRAM EQ
    hist_eq = exposure.equalize_hist(gray)

    # 9. THRESHOLD (Otsu)
    thresh_val = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    threshold  = thresh_val.astype(np.float64) / 255.0

    # 10. MORPHOLOGY
    selem      = morphology.disk(3)
    morph_open  = morphology.opening(threshold, selem)
    morph_close = morphology.closing(morph_open, selem)
    morph_out   = normalize(morph_close)

    # 11. FFT
    fshift    = np.fft.fftshift(np.fft.fft2(gray))
    fft_vis   = normalize(np.log1p(np.abs(fshift)))

    # 12. HEATMAP  (SR with jet colormap)
    heatmap = SR

    # ── METRICS ──────────────────────────────────
    psnr_val = float(psnr_fn(gray, SR, data_range=1.0))
    ssim_val = float(ssim_fn(gray, SR, data_range=1.0))
    rmse_val = float(np.sqrt(np.mean((gray - SR) ** 2)))

    # ── SAVE LOCALLY ─────────────────────────────
    ts      = save_folder or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    jet, bwr = "jet", "bwr"
    save_image(original,         run_dir / "01_original.png")
    save_image(preprocessed_vis, run_dir / "02_preprocessed.png",   jet)
    save_image(low_res,          run_dir / "03_low_resolution.png",  jet)
    save_image(SR,               run_dir / "04_super_resolution.png",jet)
    save_image(canny_edge,       run_dir / "05_canny_edge.png",      jet)
    save_image(sobel_edge,       run_dir / "06_sobel_edge.png",      jet)
    save_image(prewitt_edge,     run_dir / "07_prewitt_edge.png",    jet)
    save_image(hist_eq,          run_dir / "08_histogram_eq.png",    jet)
    save_image(threshold,        run_dir / "09_threshold.png",       bwr)
    save_image(morph_out,        run_dir / "10_morphology.png",      bwr)
    save_image(fft_vis,          run_dir / "11_fft.png",             jet)
    save_image(heatmap,          run_dir / "12_heatmap.png",         jet)

    metrics = {"psnr": round(psnr_val, 2), "ssim": round(ssim_val, 4), "rmse": round(rmse_val, 4)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"[Python] Saved run to {run_dir}")

    return {
        "original":     arr_to_b64(original),
        "preprocessed": arr_to_b64(preprocessed_vis, jet),
        "low_res":      arr_to_b64(low_res,           jet),
        "super_res":    arr_to_b64(SR,                jet),
        "canny":        arr_to_b64(canny_edge,        jet),
        "sobel":        arr_to_b64(sobel_edge,        jet),
        "prewitt":      arr_to_b64(prewitt_edge,      jet),
        "hist_eq":      arr_to_b64(hist_eq,           jet),
        "threshold":    arr_to_b64(threshold,         bwr),
        "morphology":   arr_to_b64(morph_out,         bwr),
        "fft":          arr_to_b64(fft_vis,           jet),
        "heatmap":      arr_to_b64(heatmap,           jet),
        "save_folder":      str(run_dir),
        "save_folder_name": ts,
        "backend": "Python · OpenCV · scikit-image",
        "metrics": metrics,
    }


# ══════════════════════════════════════════════════
# MATLAB PIPELINE  (wraps run_multi.m → 12 panels)
# ══════════════════════════════════════════════════

def run_matlab_pipeline(
    img_bgr: np.ndarray,
    median_size: int  = 5,
    clahe_clip: float = 0.02,
    save_folder: str  = None,
) -> dict:
    """Run the MATLAB run_multi.m pipeline and return the same 12-image dict."""
    import matlab.engine

    # Save input image to a temp file MATLAB can read
    tmp_input = Path(tempfile.mkdtemp()) / "input.png"
    cv2.imwrite(str(tmp_input), img_bgr)

    # Also run Python pipeline for any panels MATLAB doesn't produce
    # (MATLAB produces 9 out of 12; we fill the rest via Python)
    h, w = img_bgr.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    h, w = img_bgr.shape[:2]
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float64) / 255.0
    gray      = np.dot(img_float[..., :3], [0.2989, 0.5870, 0.1140])
    original  = img_float.copy()

    # Run MATLAB
    ts      = save_folder or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    matlab_out_dir = run_dir / "matlab_data"
    matlab_out_dir.mkdir(exist_ok=True)

    eng = matlab.engine.start_matlab()
    try:
        eng.addpath(str(MATLAB_DIR), nargout=0)        # find run_multi.m
        eng.addpath(str(matlab_out_dir), nargout=0)
        eng.run_multi_out(str(tmp_input), str(matlab_out_dir), nargout=0)
    finally:
        eng.quit()

    # ── Load MATLAB outputs (9 images) ──────────
    def load_mat_img(fname) -> np.ndarray | None:
        p = matlab_out_dir / fname
        if not p.exists():
            return None
        arr = np.array(Image.open(p).convert("L")).astype(np.float64) / 255.0
        return arr

    SR_mat      = load_mat_img("SR.png")
    edges_mat   = load_mat_img("edges.png")
    sobel_mat   = load_mat_img("sobel.png")
    prewitt_mat = load_mat_img("prewitt.png")
    histeq_mat  = load_mat_img("histeq.png")
    bw_mat      = load_mat_img("bw.png")
    morph_mat   = load_mat_img("morph.png")
    fft_mat     = load_mat_img("fft.png")
    heatmap_raw = matlab_out_dir / "heatmap.png"

    # Use MATLAB outputs where available, Python fallbacks where not
    gray_proc = median_filter(gray, size=median_size)
    gray_proc = exposure.equalize_adapthist(gray_proc, clip_limit=clahe_clip)

    lh2, lw2 = int(h * 0.4), int(w * 0.4)
    low_res_py = cv2.resize(
        cv2.resize(gray_proc, (lw2, lh2), interpolation=cv2.INTER_CUBIC),
        (w, h), interpolation=cv2.INTER_CUBIC
    )

    SR_use      = SR_mat      if SR_mat      is not None else low_res_py
    sobel_use   = sobel_mat   if sobel_mat   is not None else normalize(np.hypot(
                    cv2.Sobel(gray_proc, cv2.CV_64F, 1, 0, ksize=3),
                    cv2.Sobel(gray_proc, cv2.CV_64F, 0, 1, ksize=3)))
    prewitt_use = prewitt_mat if prewitt_mat is not None else sobel_use
    canny_use   = edges_mat   if edges_mat   is not None else (
                    cv2.Canny((gray_proc*255).astype(np.uint8), 50, 150).astype(np.float64)/255.0)
    histeq_use  = histeq_mat  if histeq_mat  is not None else exposure.equalize_hist(gray_proc)
    bw_use      = bw_mat      if bw_mat      is not None else (
                    cv2.threshold((gray_proc*255).astype(np.uint8), 0, 255,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1].astype(np.float64)/255.0)
    morph_use   = morph_mat   if morph_mat   is not None else bw_use
    fft_use     = fft_mat     if fft_mat     is not None else normalize(
                    np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray_proc)))))

    # Heatmap: use MATLAB RGB file if it exists
    heatmap_b64 = ""
    if heatmap_raw.exists():
        with open(heatmap_raw, "rb") as f:
            heatmap_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    else:
        heatmap_b64 = arr_to_b64(SR_use, "jet")

    # ── METRICS ──────────────────────────────────
    psnr_val = float(psnr_fn(gray_proc, SR_use, data_range=1.0))
    ssim_val = float(ssim_fn(gray_proc, SR_use, data_range=1.0))
    rmse_val = float(np.sqrt(np.mean((gray_proc - SR_use) ** 2)))

    # ── SAVE 12 panels locally ───────────────────
    jet, bwr = "jet", "bwr"
    save_image(original,      run_dir / "01_original.png")
    save_image(gray_proc,     run_dir / "02_preprocessed.png",    jet)
    save_image(low_res_py,    run_dir / "03_low_resolution.png",  jet)
    save_image(SR_use,        run_dir / "04_super_resolution.png",jet)
    save_image(canny_use,     run_dir / "05_canny_edge.png",      jet)
    save_image(sobel_use,     run_dir / "06_sobel_edge.png",      jet)
    save_image(prewitt_use,   run_dir / "07_prewitt_edge.png",    jet)
    save_image(histeq_use,    run_dir / "08_histogram_eq.png",    jet)
    save_image(bw_use,        run_dir / "09_threshold.png",       bwr)
    save_image(morph_use,     run_dir / "10_morphology.png",      bwr)
    save_image(fft_use,       run_dir / "11_fft.png",             jet)
    save_image(SR_use,        run_dir / "12_heatmap.png",         jet)

    metrics = {"psnr": round(psnr_val, 2), "ssim": round(ssim_val, 4), "rmse": round(rmse_val, 4)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"[MATLAB] Saved run to {run_dir}")

    return {
        "original":     arr_to_b64(original),
        "preprocessed": arr_to_b64(gray_proc,    jet),
        "low_res":      arr_to_b64(low_res_py,   jet),
        "super_res":    arr_to_b64(SR_use,        jet),
        "canny":        arr_to_b64(canny_use,     jet),
        "sobel":        arr_to_b64(sobel_use,     jet),
        "prewitt":      arr_to_b64(prewitt_use,   jet),
        "hist_eq":      arr_to_b64(histeq_use,    jet),
        "threshold":    arr_to_b64(bw_use,        bwr),
        "morphology":   arr_to_b64(morph_use,     bwr),
        "fft":          arr_to_b64(fft_use,       jet),
        "heatmap":      heatmap_b64,
        "save_folder":      str(run_dir),
        "save_folder_name": ts,
        "backend": "MATLAB Engine · run_multi.m",
        "metrics": metrics,
    }


def decode_image(contents: bytes) -> np.ndarray | None:
    """Decode raw bytes into a BGR image. OpenCV first, Pillow fallback.

    Pillow handles formats OpenCV's codec set sometimes misses
    (exotic TIFFs, some WEBP/CMYK JPEGs). Returns None only when
    no decoder can handle the bytes.
    """
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        with Image.open(io.BytesIO(contents)) as pil_img:
            rgb = np.asarray(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# ══════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════

@app.get("/matlab-status")
def matlab_status():
    """Returns whether the MATLAB Engine API is available."""
    return {"available": MATLAB_AVAILABLE}


@app.get("/samples")
def list_samples():
    """Return a list of sample images from the project/ subfolder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    files = [
        f.name for f in sorted(SAMPLES_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in exts
    ]
    return JSONResponse({"samples": files})


@app.post("/process")
async def process(
    file:           UploadFile = File(...),
    edge_weight:    float = Form(0.6),
    detail_weight:  float = Form(0.4),
    sharpen_amount: float = Form(1.8),
    lr_scale:       float = Form(0.4),
    median_size:    int   = Form(5),
    clahe_clip:     float = Form(0.02),
):
    """Python/OpenCV backend — always available."""
    contents = await file.read()
    img = decode_image(contents)
    if img is None:
        return JSONResponse(
            {"error": "Unsupported or corrupted image file. Supported formats: JPG, PNG, WEBP, BMP, TIFF."},
            status_code=400,
        )
    try:
        # Run the CPU-heavy pipeline in a worker thread so the
        # event loop (and therefore /health, /runs, other requests)
        # stays responsive during long processing runs.
        result = await run_in_threadpool(
            run_pipeline, img, edge_weight, detail_weight, sharpen_amount, lr_scale, median_size, clahe_clip
        )
        return JSONResponse(result)
    except Exception as e:
        log.exception("Python pipeline error")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/process/matlab")
async def process_matlab(
    file:        UploadFile = File(...),
    median_size: int        = Form(5),
    clahe_clip:  float      = Form(0.02),
):
    """MATLAB Engine backend — requires MATLAB + matlab.engine installed."""
    if not MATLAB_AVAILABLE:
        return JSONResponse(
            {"error": "MATLAB Engine is not installed. Run: pip install matlabengine"},
            status_code=503,
        )
    contents = await file.read()
    img = decode_image(contents)
    if img is None:
        return JSONResponse(
            {"error": "Unsupported or corrupted image file. Supported formats: JPG, PNG, WEBP, BMP, TIFF."},
            status_code=400,
        )
    try:
        result = await run_in_threadpool(run_matlab_pipeline, img, median_size, clahe_clip)
        return JSONResponse(result)
    except Exception as e:
        log.exception("MATLAB pipeline error")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/runs")
def list_runs():
    """List all saved runs."""
    runs = []
    for d in sorted(OUTPUTS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            metrics_file = d / "metrics.json"
            m     = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
            files = [f.name for f in sorted(d.glob("*.png"))]
            runs.append({"name": d.name, "metrics": m, "files": files})
    return JSONResponse({"runs": runs})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend_python": True,
        "backend_matlab": MATLAB_AVAILABLE,
        "outputs_dir": str(OUTPUTS_DIR),
    }


@app.get("/", include_in_schema=False)
def web_home():
    """Serve the ThermalSR dashboard landing page."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/{page}.html", include_in_schema=False)
def web_page_html(page: str):
    """Serve known dashboard pages via .html extension."""
    if page not in {"index", "process", "architecture", "history"}:
        return JSONResponse({"error": "Page not found"}, status_code=404)
    return FileResponse(WEB_DIR / f"{page}.html")


@app.get("/{page}", include_in_schema=False)
def web_page_clean(page: str):
    """Serve known dashboard pages via clean URLs (no .html extension)."""
    if page not in {"index", "process", "architecture", "history"}:
        return JSONResponse({"error": "Page not found"}, status_code=404)
    return FileResponse(WEB_DIR / f"{page}.html")


@app.get("/download/{run_id}")
def download_run(run_id: str):
    # Security: ensure run_id is a safe alphanumeric/underscore name (no path traversal)
    if not re.match(r'^[\w\-]+$', run_id):
        return JSONResponse({"error": "Invalid run ID"}, status_code=400)
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return JSONResponse({"error": "Run not found"}, status_code=404)

    with tempfile.TemporaryDirectory() as tmp:
        zip_base = Path(tmp) / run_id
        zip_path = Path(tmp) / f"{run_id}.zip"
        shutil.make_archive(str(zip_base), "zip", str(run_dir))
        # Read into memory so the temp dir can be cleaned up safely
        zip_bytes = zip_path.read_bytes()

    import io as _io
    from fastapi.responses import Response
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="ThermalSR_{run_id}.zip"'},
    )
