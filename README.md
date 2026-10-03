# ThermalSR — Full Project
## ISRO Problem Statement #25171 · Optical-Guided Super-Resolution for Thermal IR Imagery
### Description
ThermalSR is a web-based thermal infrared image processing system designed
to enhance and analyze low-resolution thermal imagery using optical-guided
super-resolution and computer vision techniques. 

The system provides a multi-stage image processing pipeline including
preprocessing, low-resolution generation, super-resolution, edge detection,
morphological processing, FFT analysis, histogram analysis, and visualization.

---

## Project Structure

```
thermalsr/
├── frontend/
│   ├── index.html                ← Landing Page
│   ├── css/
│   │   └── global.css            ← Shared design system
│   └── pages/
│       ├── process.html          ← Upload & Run Pipeline
│       ├── architecture.html     ← Pipeline Architecture Explainer
│       └── dashboard.html        ← Metrics Dashboard
│
├── backend/
│   ├── main.py                   ← FastAPI server (MATLAB Engine + Python fallback)
│   └── requirements.txt
│
└── README.md
```

---

## Quick Start

### 1. Frontend (No backend needed)
Just open `frontend/index.html` in any browser.
All pages work standalone with a browser-side Canvas fallback pipeline.

---

### 2. Full Stack (Frontend + FastAPI + MATLAB)

#### Step 1 — Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Step 2 — Install MATLAB Engine for Python
MATLAB R2021b or newer must be installed on your machine.

**Option A (pip, MATLAB R2022b+):**
```bash
pip install matlabengine
```

**Option B (older MATLAB):**
```bash
cd "$(matlab -e MATLAB)/extern/engines/python"
python setup.py install
```

#### Step 3 — Start the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

You'll see:
- `MATLAB engine started successfully.` — full MATLAB pipeline active
- `MATLAB not available. Using Python/OpenCV fallback.` — Python pipeline used instead

#### Step 4 — Open the frontend
Open `frontend/index.html` in your browser.
Click **Run Pipeline** → **Process** page.

---

## Pages

| Page | File | Description |
|------|------|-------------|
| Landing | `index.html` | Overview, features, applications |
| Process | `pages/process.html` | Upload image, configure params, run pipeline |
| Architecture | `pages/architecture.html` | Network modules, loss functions, MATLAB translation |
| Dashboard | `pages/dashboard.html` | PSNR/SSIM/RMSE charts, comparison, run history |

---

## How the MATLAB Integration Works

The backend (`main.py`) calls your MATLAB functions directly via `matlab.engine`:

```
Your MATLAB script          →    Python (MATLAB Engine API)
─────────────────────────────────────────────────────────────
medfilt2(gray, [5 5])       →    eng.medfilt2(mat, matlab.double([[5,5]]))
adapthisteq(gray, ...)      →    eng.adapthisteq(mat, 'ClipLimit', 0.02)
imresize(gray, 0.4, ...)    →    eng.imresize(mat, 0.4, 'bicubic')
imgaussfilt(img, 1)         →    eng.imgaussfilt(mat, 1.0)
edge(gray, 'canny')         →    eng.edge(mat, 'canny')
mat2gray(SR)                →    eng.mat2gray(mat)
imsharpen(SR, ...)          →    eng.imsharpen(mat, 'Radius', 2, 'Amount', 1.8)
psnr(SR, gray)              →    eng.psnr(SR_mat, gray_mat)
ssim(SR, gray)              →    eng.ssim(SR_mat, gray_mat)
```

The `/health` endpoint reports whether MATLAB was detected:
```
GET http://localhost:8000/health
→ {"status":"ok","matlab":true,"backend":"MATLAB Engine API"}
```

---

## API Reference

### POST /process
Processes an uploaded image through the full pipeline.

**Form fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| file | File | required | Any image (JPG, PNG, TIFF) |
| edge_weight | float | 0.6 | Weight of Canny edge guidance |
| detail_weight | float | 0.4 | Weight of multi-scale detail |
| sharpen_amount | float | 1.8 | Unsharp mask strength |
| lr_scale | float | 0.4 | Low-res downscale factor |

**Response:**
```json
{
  "original":  "data:image/png;base64,...",
  "gray":      "data:image/png;base64,...",
  "low_res":   "data:image/png;base64,...",
  "edges":     "data:image/png;base64,...",
  "sr_output": "data:image/png;base64,...",
  "heatmap":   "data:image/png;base64,...",
  "backend":   "MATLAB Engine API",
  "metrics": {
    "psnr": 31.24,
    "ssim": 0.8734,
    "rmse": 0.0612
  }
}
```

---

## Metric Thresholds

| Metric | Excellent | Acceptable | Poor |
|--------|-----------|------------|------|
| PSNR   | > 30 dB   | 20–30 dB   | < 20 |
| SSIM   | > 0.85    | 0.65–0.85  | < 0.65 |
| RMSE   | < 0.05    | 0.05–0.15  | > 0.15 |

---

## Connecting a Deep Learning Model

To plug in a trained PyTorch model, add to `main.py`:
```python
import torch
model = torch.load('weights/thermal_sr.pth', map_location='cpu').eval()

# Replace SR fusion step:
with torch.no_grad():
    t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float()
    SR = model(t).squeeze().numpy()
```
