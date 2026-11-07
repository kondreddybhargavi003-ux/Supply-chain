'''run_pipeline.py

Entry point for the full StockSense end‑to‑end workflow.

The pipeline consists of the following logical stages:

1. **preprocess** – Load raw CSVs and produce cleaned files.
2. **eda**        – Exploratory Data Analysis (generates figures & reports).
3. **models**     – Demand forecasting, stock‑out prediction and inventory optimisation.
4. **final**      – Assemble final reports, dashboards and artefacts.

Each stage can be executed independently via the ``--stage`` argument.
If ``--stage all`` (default) the pipeline runs every stage in order.
The script checks for existing artefacts and skips a stage when its
expected output files already exist, allowing incremental reruns.

All paths are resolved relative to the project root so the pipeline
works on Windows, macOS and Linux.
''' 

import argparse
import subprocess
import sys
from pathlib import Path
import logging

# Configure a simple logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Mapping of stages to the scripts that must be executed
STAGE_SCRIPTS = {
    "preprocess": [PROJECT_ROOT / "src" / "data" / "preprocess.py"],
    "eda": [PROJECT_ROOT / "scripts" / "08_eda.py"],
    "models": [
        PROJECT_ROOT / "scripts" / "11_demand_features.py",
        PROJECT_ROOT / "scripts" / "12_demand_models.py",
        PROJECT_ROOT / "scripts" / "14_stockout_model.py",
        PROJECT_ROOT / "scripts" / "17_inventory_optimization.py",
        PROJECT_ROOT / "scripts" / "18_rop_calibration.py",
        PROJECT_ROOT / "scripts" / "19_reorder_final.py",
    ],
    "final": [
        PROJECT_ROOT / "scripts" / "20_action_list.py",
        PROJECT_ROOT / "scripts" / "21_validate_sellout_probability.py",
        PROJECT_ROOT / "scripts" / "22_score_latest.py",
        PROJECT_ROOT / "scripts" / "23_procurement_tables.py",
    ],
}


def _run_script(script_path: Path) -> None:
    """Execute a Python script using the same interpreter.

    The script is run as a subprocess so any global state (e.g. random
    seeds) remains isolated. stdout/stderr are forwarded to the console.
    """
    if not script_path.exists():
        logging.error("Script not found: %s", script_path)
        sys.exit(1)
    cmd = [sys.executable, str(script_path)]
    logging.info("Running %s", script_path.name)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)
    if result.returncode != 0:
        logging.error("Script %s exited with code %s", script_path.name, result.returncode)
        sys.exit(result.returncode)


def _stage_completed(stage: str) -> bool:
    """Very simple heuristic to decide whether a stage can be skipped.

    For ``preprocess`` we look for the two cleaned CSVs.
    For other stages we check for at least one expected output file in the
    ``reports`` directory that matches the stage name.
    """
    if stage == "preprocess":
        return (DATA_PROCESSED / "sales_clean.csv").exists() and (
            DATA_PROCESSED / "inventory_intervals_clean.csv"
        ).exists()
    # generic check – any file whose name contains the stage identifier
    pattern = f"*{stage}*"
    return any(REPORTS_DIR.glob(pattern))


def run_pipeline(stages: list[str]) -> None:
    for stage in stages:
        if _stage_completed(stage):
            logging.info("Stage '%s' already has output – skipping.", stage)
            continue
        logging.info("=== Running stage: %s ===", stage)
        for script in STAGE_SCRIPTS.get(stage, []):
            _run_script(script)
        logging.info("=== Completed stage: %s ===", stage)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the StockSense project pipeline.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Stage(s) to run: preprocess, eda, models, final, or all (comma‑separated).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "all":
        selected = list(STAGE_SCRIPTS.keys())
    else:
        selected = [s.strip() for s in args.stage.split(",")]
    run_pipeline(selected)

