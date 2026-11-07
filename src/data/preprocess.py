'''Preprocessing module for StockSense project.

This module reproduces the functionality of the original
`scripts/07_preprocessing.py` but is structured as importable
functions so that it can be reused by the pipeline and other
scripts.

The steps are:
1. Load raw retail sales and inventory CSVs.
2. Standardise column names (lower‑case, underscores).
3. Clean sales data – remove zero‑quantity rows, separate returns,
   drop unused columns, compute gross and return quantities.
4. Clean inventory data – handle open‑ended end dates (``9999-12-31``),
   fill missing/invalid price fields, flag out‑of‑stock & negative
   stock, drop unused columns, and check for overlapping intervals.
5. Save the cleaned tables to ``data/processed`` as
   ``sales_clean.csv`` and ``inventory_intervals_clean.csv``.

All paths are resolved relative to the project root, enabling the
module to work on any platform (Windows, macOS, Linux).
''' 

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

# Configure a simple logger for debugging / informational messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Project‑relative directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/data/.. -> project root
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase, strip whitespace and replace spaces with underscores."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_raw_sales() -> pd.DataFrame:
    """Load the raw retail sales CSV and standardise column names."""
    path = RAW_DIR / "retail_sales_ml_apl.csv"
    logging.info("Loading sales data from %s", path)
    df = pd.read_csv(path, low_memory=False)
    return _standardise_columns(df)


def load_raw_inventory() -> pd.DataFrame:
    """Load the raw retail inventory CSV and standardise column names."""
    path = RAW_DIR / "retail_inventory_ml_apl.csv"
    logging.info("Loading inventory data from %s", path)
    df = pd.read_csv(path, low_memory=False)
    return _standardise_columns(df)


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Perform the sales‑specific cleaning steps.

    Returns a cleaned ``sales`` DataFrame with the columns listed in the
    original script (``sales_cols``) and sorted chronologically.
    """
    sales = sales.copy()

    # Parse dates
    sales["transaction_date"] = pd.to_datetime(sales["transaction_date"])

    # Remove rows with zero quantity sold
    zero_rows = (sales["qty_sold"] == 0).sum()
    logging.info("Rows with zero quantity removed: %d", zero_rows)
    sales = sales[sales["qty_sold"] != 0]

    # Return vs negative quantity sanity check (informational only)
    logging.info(
        "Is Return vs negative quantity (should line up exactly):\n%s",
        pd.crosstab(sales["is_return"], sales["qty_sold"] < 0).to_string(),
    )

    # Drop columns that are not needed downstream
    sales = sales.drop(columns=["reason_of_return", "sales_channel"], errors="ignore")

    # Separate gross and return quantities
    sales["gross_qty"] = sales["qty_sold"].clip(lower=0)
    sales["return_qty"] = (-sales["qty_sold"]).clip(lower=0)

    # Duplicate rows are kept – they may represent legitimate repeat sales
    dups = sales.duplicated().sum()
    logging.info("Fully identical rows (kept, they can be real repeat sales): %d", dups)
    logging.info("Clean sales rows: %d", len(sales))
    logging.info(
        "Gross units: %,.0f | returned units: %,.0f",
        sales["gross_qty"].sum(),
        sales["return_qty"].sum(),
    )

    # Keep only the columns required for downstream analysis and sort
    sales_cols = [
        "transaction_date",
        "store",
        "product_no",
        "product_description",
        "product_division",
        "product_category",
        "product_subcategory",
        "product_segment",
        "supplier",
        "sales_type",
        "is_return",
        "qty_sold",
        "gross_qty",
        "return_qty",
        "sales_amount",
        "cogs",
        "number_of_transactions",
    ]
    sales = sales[sales_cols].sort_values(["transaction_date", "store", "product_no"]).reset_index(drop=True)
    return sales


def _fill_price(inv: pd.DataFrame, series_name: str, source_col: str, pair: list[str]):
    """Helper to fill missing/invalid price values.

    * Values <= 0 are treated as missing.
    * Forward‑fill then backward‑fill within each product‑store pair.
    * Remaining missing values are filled with the median price for the product.
    """
    inv[series_name] = inv[source_col].where(inv[source_col] > 0)
    blank_before = inv[series_name].isna().sum()
    inv[series_name] = inv.groupby(pair)[series_name].ffill()
    inv[series_name] = inv.groupby(pair)[series_name].bfill()
    inv[series_name] = inv[series_name].fillna(
        inv.groupby("product_no")[series_name].transform("median")
    )
    logging.info(
        "%s: blank/invalid before = %d | still missing after filling = %d",
        series_name,
        blank_before,
        inv[series_name].isna().sum(),
    )
    return inv


def clean_inventory(inv: pd.DataFrame) -> pd.DataFrame:
    """Perform the inventory‑specific cleaning steps.

    Returns a cleaned inventory DataFrame ready for interval‑based analysis.
    """
    inv = inv.copy()

    # Determine the cutoff date from the latest sales transaction (provided later by the caller)
    # The caller must set ``cutoff`` before invoking this function; we keep the logic here for clarity.
    cutoff = inv["transaction_date"].max() if "transaction_date" in inv.columns else None
    if cutoff is None:
        raise ValueError("Cutoff date (max sales transaction date) must be provided externally before cleaning inventory.")

    # Identify open‑ended records (end_date = 9999‑12‑31) and replace with cutoff
    open_mask = inv["end_date"].astype(str).str.startswith("9999")
    inv["end_is_open"] = open_mask.astype(int)
    inv["start_date"] = pd.to_datetime(inv["start_date"], errors="coerce")
    # Parse end_date only when not open‑ended; otherwise set to NaT for now
    inv["end_date"] = pd.to_datetime(inv["end_date"].where(~open_mask), errors="coerce")
    inv.loc[open_mask, "end_date"] = cutoff
    inv["days_valid"] = (inv["end_date"] - inv["start_date"]).dt.days + 1
    logging.info(
        "Open‑ended records (9999‑12‑31) set to %s: %d",
        cutoff.date() if cutoff else "<none>",
        open_mask.sum(),
    )
    logging.info(
        "Records with end before start: %d",
        (inv["end_date"] < inv["start_date"]).sum(),
    )

    # Sort for deterministic filling
    inv = inv.sort_values(["product_no", "store", "start_date"]).reset_index(drop=True)
    pair = ["product_no", "store"]

    # Fill price fields
    inv = _fill_price(inv, "unit_price", "stock_unit_selling_price", pair)
    inv = _fill_price(inv, "unit_cost", "stock_unit_cost_price", pair)

    # Flag out‑of‑stock / negative stock
    inv["out_of_stock"] = (inv["qty_on_hand"] <= 0).astype(int)
    inv["negative_stock"] = (inv["qty_on_hand"] < 0).astype(int)
    logging.info(
        "\nout_of_stock records: %d | negative_stock records: %d",
        inv["out_of_stock"].sum(),
        inv["negative_stock"].sum(),
    )

    # Drop columns that are not needed downstream
    drop_cols = [
        "stocks_selling_amount",
        "cost_of_stocks",
        "stock_unit_selling_price",
        "stock_unit_cost_price",
        "sales_channel",
        "store_type",
    ]
    inv = inv.drop(columns=drop_cols, errors="ignore")
    logging.info("Dropped columns: %s", drop_cols)

    # Check for overlapping intervals within each product‑store pair
    prev_end = inv.groupby(pair)["end_date"].shift()
    overlap_cnt = (inv["start_date"] <= prev_end).sum()
    logging.info("Overlapping records after cleaning: %d", overlap_cnt)
    logging.info("Clean inventory rows: %d", len(inv))
    return inv


def run_preprocessing() -> None:
    """Entry‑point that runs the full preprocessing pipeline.

    The function reads the raw CSVs, cleans them, and writes the cleaned
    versions to ``data/processed``.
    """
    sales_raw = load_raw_sales()
    inventory_raw = load_raw_inventory()

    # Clean sales first – we need the latest transaction date for inventory handling
    sales_clean = clean_sales(sales_raw)
    # Save sales output
    sales_path = PROCESSED_DIR / "sales_clean.csv"
    sales_clean.to_csv(sales_path, index=False)
    logging.info("Saved cleaned sales to %s", sales_path)

    # Determine cutoff date for inventory (max sales transaction date)
    cutoff_date = sales_clean["transaction_date"].max()

    # Inject the cutoff into the inventory cleaning routine via a temporary column
    inventory_raw = inventory_raw.copy()
    inventory_raw["transaction_date"] = cutoff_date  # temporary helper column
    inventory_clean = clean_inventory(inventory_raw)
    # Remove the helper column before saving
    if "transaction_date" in inventory_clean.columns:
        inventory_clean = inventory_clean.drop(columns=["transaction_date"])
    inv_path = PROCESSED_DIR / "inventory_intervals_clean.csv"
    inventory_clean.to_csv(inv_path, index=False)
    logging.info("Saved cleaned inventory to %s", inv_path)


if __name__ == "__main__":
    run_preprocessing()

