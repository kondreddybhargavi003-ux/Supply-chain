from pathlib import Path
import numpy as np
import pandas as pd

RAW, PROC = Path("data/raw"), Path("data/processed")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

names = ["buyers", "goods_receipts", "items", "po_lines", "purchase_orders",
         "requisitions", "suppliers", "supplier_invoices"]
data = {n: pd.read_csv(RAW / f"{n}.csv", low_memory=False) for n in names}

def date_cols(df):
    return [c for c in df.columns if any(k in c.lower() for k in ["date", "_on", "_at"])]

# ------------------------------------------------------------------ A
section("A. COLUMNS AND SAMPLE ROW OF EVERY PROCUREMENT TABLE")
for n, df in data.items():
    print(f"\n{n}: {len(df):,} rows")
    print(f"  columns: {list(df.columns)}")
    print(df.head(2).T.to_string())

# ------------------------------------------------------------------ B
section("B. SHARED COLUMNS BETWEEN TABLES")
keys = list(data)
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        sh = [c for c in data[a].columns if c in data[b].columns]
        if sh:
            print(f"{a} <-> {b}: {sh}")

# ------------------------------------------------------------------ C
section("C. LEAD TIME: PURCHASE ORDER DATES vs GOODS RECEIPT DATES")
po, gr = data["purchase_orders"], data["goods_receipts"]
shared = [c for c in po.columns if c in gr.columns]
cand = [c for c in shared if "po" in c.lower()] or [c for c in shared if c.lower().endswith("id")]
print(f"Shared columns: {shared} | join key used: {cand[0] if cand else 'NONE FOUND'}")
if cand:
    key = cand[0]
    mg = gr.merge(po, on=key, how="left", suffixes=("_gr", "_po"), indicator=True)
    print(f"Goods receipts: {len(gr):,} | matched to a purchase order: {(mg['_merge'] == 'both').sum():,}")
    def nm(c, side, other):
        return c + side if c in other.columns and c != key else c
    po_dates = [nm(c, "_po", gr) for c in date_cols(po)]
    gr_dates = [nm(c, "_gr", po) for c in date_cols(gr)]
    print(f"PO date columns: {po_dates} | receipt date columns: {gr_dates}")
    for pc in po_dates:
        for gc in gr_dates:
            if pc in mg.columns and gc in mg.columns:
                dd = (pd.to_datetime(mg[gc], errors="coerce")
                      - pd.to_datetime(mg[pc], errors="coerce")).dt.days
                if dd.notna().sum() > 0:
                    print(f"\n  days from {pc} to {gc} (n={dd.notna().sum()}):")
                    print(dd.describe().round(1).to_string())
    sup_cols = [c for c in mg.columns if "supplier" in c.lower()]
    print(f"\nSupplier-like columns in merged table: {sup_cols}")

# ------------------------------------------------------------------ D
section("D. CAN PROCUREMENT SUPPLIERS BE LINKED TO RETAIL SUPPLIERS?")
sup = data["suppliers"]
retail_sup = set(pd.read_csv(PROC / "sales_clean.csv", usecols=["supplier"])["supplier"]
                 .dropna().astype(str))
print(f"Retail suppliers: {len(retail_sup)} | sample: {sorted(retail_sup)[:4]}")
for c in sup.columns:
    vals = set(sup[c].dropna().astype(str))
    print(f"suppliers.{c}: {len(vals)} unique | sample {sorted(vals)[:3]} "
          f"| exact matches with retail supplier names: {len(vals & retail_sup)}")

# ------------------------------------------------------------------ E
section("E. CAN PROCUREMENT ITEMS BE LINKED TO RETAIL PRODUCTS?")
retail_prod = set(pd.read_csv(PROC / "sales_clean.csv", usecols=["product_no"])["product_no"].astype(str))
print(f"Retail Product No sample: {sorted(retail_prod)[:3]}")
for n, df in data.items():
    for c in df.columns:
        if any(k in c.lower() for k in ["sku", "item"]):
            vals = set(df[c].dropna().astype(str))
            print(f"{n}.{c}: {len(vals)} unique | sample {sorted(vals)[:3]} "
                  f"| matches in retail Product No: {len(vals & retail_prod)}")

# ------------------------------------------------------------------ F
section("F. SUPPLIER INVOICES: MISSING paid_on")
inv = data["supplier_invoices"]
if "paid_on" in inv.columns:
    miss = inv["paid_on"].isna().rename("paid_on_missing")
    print(f"paid_on missing: {miss.sum()} of {len(inv)}")
    for c in inv.columns:
        if c != "paid_on" and 0 < inv[c].nunique() <= 12:
            print(f"\n{c} vs paid_on missing:")
            print(pd.crosstab(inv[c], miss).to_string())
else:
    print("No paid_on column found. Columns:", list(inv.columns))
