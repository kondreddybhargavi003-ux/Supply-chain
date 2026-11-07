# README.md

## StockSense: Supply Chain Intelligence and Inventory Optimization

This repository contains a complete end‑to‑end data‑science project that ingests retail sales, inventory, and procurement data, performs data cleaning, exploratory data analysis (EDA), demand forecasting, stock‑out prediction, and inventory optimisation, and finally presents the results in an interactive Streamlit dashboard and a lightweight FastAPI service.

---

### Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Run the full pipeline](#run-the-full-pipeline)
  - [Launch the dashboard](#launch-the-dashboard)
  - [Start the API server](#start-the-api-server)
  - [Run the test suite](#run-the-test-suite)
- [Outputs](#outputs)
- [References](#references)

---

### Project Structure

```
.
├─ data
│  ├─ raw                 # Original CSV files (sales, inventory, procurement)
│  └─ processed           # Cleaned CSVs produced by the pipeline
│
├─ src                     # Re‑usable Python modules
│  └─ data
│      ├─ __init__.py
│      └─ preprocess.py    # Clean‑ing logic (replaces scripts/07_preprocessing.py)
│
├─ scripts                 # Original analysis scripts (kept for reference)
│  ├─ 01_load_and_inspect.py
│  ├─ ...
│  └─ 23_procurement_tables.py
│
├─ reports                 # Text output files and generated figures
│  ├─ *.txt
│  ├─ figures
│  │   └─ *.png
│  └─ project_audit.md     # Full audit created by the agent
│
├─ dashboard               # Streamlit application
│  ├─ app.py               # Main entry point (multipage)
│  └─ pages                # Individual pages (optional)
│
├─ api                     # FastAPI service exposing summaries & predictions
│  └─ app.py
│
├─ tests                   # Automated pytest suite
│  ├─ conftest.py
│  ├─ test_preprocess.py
│  ├─ test_pipeline.py
│  └─ test_dashboard.py
│
├─ requirements.txt        # Pin‑exact versions of all Python dependencies
├─ run_pipeline.py         # Orchestrates preprocessing, EDA, modeling, final stage
├─ .gitignore
└─ README.md
```

---

### Installation

1. **Clone the repository** (if you haven't already) and navigate to the project root:
   ```
   git clone <repo‑url>
   cd "SS - Copy"
   ```

2. **Create a virtual environment** (recommended) and activate it:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # PowerShell
   # or venv\Scripts\activate.bat for cmd
   ```

3. **Install the pinned dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
   *If you are on a machine without internet access, you can pre‑download the wheels on another computer (`pip download -r requirements.txt -d wheelhouse`) and then install locally with `pip install --no-index --find-links=wheelhouse -r requirements.txt`.*

---

### Usage

#### Run the full pipeline

The pipeline will generate cleaned data, all EDA figures, model artefacts and final reports. It is idempotent – stages are skipped if their outputs already exist.

```powershell
python run_pipeline.py          # runs all stages (preprocess → final)
# or run a specific stage, e.g.:
python run_pipeline.py --stage eda
```

#### Launch the dashboard

The interactive dashboard reads the artefacts produced by the pipeline and visualises them with Plotly.

```powershell
streamlit run dashboard/app.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`). The dashboard contains the following pages:
- **Executive Summary** – high‑level KPIs and back‑testing results.
- **Sales Analytics** – weekly trends, category/store breakdowns, price analysis.
- **Inventory Analytics** – stock‑out rates, excess inventory, inventory‑vs‑sales heatmaps.
- **Demand Forecasting** – model evaluation metrics and forecast plots.
- **Stock‑out Prediction** – ROC/PR curves, confusion matrix.
- **Inventory Optimisation** – reorder point heatmaps, sensitivity analysis.
- **Procurement Analytics** – supplier spending, lead‑time distribution, invoice status.

#### Start the API server

A minimal FastAPI service provides the same summaries and model predictions as JSON endpoints.

```powershell
uvicorn api.app:app --reload
```

Available endpoints (see `api/app.py` for details):
- `GET /summary` – executive KPIs.
- `GET /sales/weekly` – weekly gross units.
- `GET /inventory/stockout` – daily stock‑out share.
- `POST /demand/predict` – forecasting predictions for a given product‑store pair.

#### Run the test suite

The repository includes a comprehensive pytest suite that checks data integrity, preprocessing idempotency, model shapes, and that the dashboard can be imported without error.

```powershell
pytest -q
```

All tests should pass on a clean environment.

---

### Outputs

After a successful pipeline run you will find:
- **Cleaned CSVs** in `data/processed/` (`sales_clean.csv`, `inventory_intervals_clean.csv`).
- **Figures** in `reports/figures/` (weekly demand, stock‑out rates, ROC curves, etc.).
- **Model artefacts** in `models/` (joblib files for demand, stock‑out, and optimisation).
- **Report markdown files** in `reports/` (EDA report, model evaluation, optimisation report, final executive report).
- **Dashboard** (`dashboard/app.py`) and **API** (`api/app.py`).

---

### References
- Original StockSense project code (scripts 01‑23).
- Pandas, NumPy, scikit‑learn, XGBoost, Plotly, Streamlit documentation.
- FastAPI documentation for building lightweight services.

---

**Happy analysing!**

*If you encounter any issues, feel free to open a GitHub issue or contact the repository maintainer.*
