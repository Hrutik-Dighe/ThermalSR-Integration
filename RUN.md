# Run ThermalSR

The project is structured as a single FastAPI application that serves the
dashboard and the image-processing API from the same address.

## Folder Structure

```
ThermalSR_Integrated/
├── backend/        ← FastAPI server (main.py)
├── frontend/       ← HTML pages + CSS
├── matlab/         ← MATLAB scripts (run_multi.m)
├── samples/        ← Sample images for the Process page
├── outputs/        ← Saved pipeline run results (auto-created)
└── requirements.txt
```

## Install & Run

```powershell
# Install dependencies (from project root)
pip install -r requirements.txt

# Start the server
uvicorn backend.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

| Page | URL |
|---|---|
| Home | http://127.0.0.1:8000 |
| Process | http://127.0.0.1:8000/process |
| Architecture | http://127.0.0.1:8000/architecture |
| History | http://127.0.0.1:8000/history |
| API Docs | http://127.0.0.1:8000/docs |
