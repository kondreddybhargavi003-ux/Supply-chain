from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path("data/processed")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"])
inv = pd.read_csv(PROC / "inventory_intervals_clean.csv",
                  parse_dates=["start_date", "end_date"])
pair = ["product_no", "store"]
inv = inv.sort_values(pair + ["start_date"]).reset_index(drop=True)
g = inv.groupby(pair)
inv["prev_qty"] = g["qty_on_hand"].shift()
inv["prev_end"] = g["end_date"].shift()
inv["gap_days"] = (inv["start_date"] - inv["prev_end"]).dt.days - 1

# ------------------------------------------------------------------ A
section("A. GAPS BETWEEN CONSECUTIVE INVENTORY RECORDS (same product-store)")
gaps = inv["gap_days"].dropna()
print(f"Consecutive record pairs: {len(gaps):,}")
print(f"Back-to-back (gap = 0 days): {(gaps == 0).mean() * 100:.1f}%")
print(f"Gap > 0 days (untracked days between records): {(gaps > 0).mean() * 100:.1f}%")
print("Size of positive gaps (days):")
print(gaps[gaps > 0].describe().round(1).to_string())

# ------------------------------------------------------------------ B
section("B. WHEN DO PAIRS ENTER THE INVENTORY FILE, AND HOW DO THEY END?")
first = g["start_date"].min()
print("Pairs by month of FIRST inventory record:")
print(first.dt.to_period("M").value_counts().sort_index().to_string())
last_open = g["end_is_open"].last()
print(f"\nPairs whose last record is still open: {(last_open == 1).mean() * 100:.1f}% "
      f"({(last_open == 1).sum():,} of {len(last_open):,})")

# ------------------------------------------------------------------ C
section("C. WHY DO SALES NOT MATCH AN INVENTORY RECORD?")
left = (sales[sales["gross_qty"] > 0][["transaction_date", "product_no", "store", "gross_qty"]]
        .sort_values("transaction_date"))
right = inv[["start_date", "end_date", "product_no", "store", "qty_on_hand",
             "prev_qty", "out_of_stock"]].sort_values("start_date")
m = pd.merge_asof(left, right, left_on="transaction_date", right_on="start_date",
                  by=pair, direction="backward")
m["covered"] = m["qty_on_hand"].notna() & (m["transaction_date"] <= m["end_date"])
no_rec = m["qty_on_hand"].isna()
gap_rec = m["qty_on_hand"].notna() & ~m["covered"]
print(f"Sales rows: {len(m):,}")
print(f"  covered by a record:                 {m['covered'].sum():,} ({m['covered'].mean() * 100:.1f}%)")
print(f"  no earlier record for the pair:      {no_rec.sum():,} ({no_rec.mean() * 100:.1f}%)")
print(f"  earlier record had already ended:    {gap_rec.sum():,} ({gap_rec.mean() * 100:.1f}%)")
print("Days between that ended record and the sale:")
print((m.loc[gap_rec, "transaction_date"] - m.loc[gap_rec, "end_date"]).dt.days.describe().round(1).to_string())

# ------------------------------------------------------------------ D
section("D. IS THE QUANTITY A POST-SALE SNAPSHOT?")
cov = m[m["covered"]].copy()
cov["same_day"] = cov["transaction_date"] == cov["start_date"]
print("Share of matched sales with stock <= 0, by timing of the sale:")
t = cov.groupby("same_day")["qty_on_hand"].agg(
    sales="count", share_zero_or_less=lambda x: (x <= 0).mean() * 100)
t.index = t.index.map({True: "sale on record START date", False: "sale later in the record"})
print(t.round(2).to_string())

sd = cov[cov["same_day"] & cov["prev_qty"].notna()]
print(f"\nSales on a record start date that have a previous record: {len(sd):,}")
print(f"  previous stock > 0: {(sd['prev_qty'] > 0).mean() * 100:.1f}%")
print(f"  previous stock <= 0: {(sd['prev_qty'] <= 0).mean() * 100:.1f}%")
print(f"  mean change in stock (new - previous): {(sd['qty_on_hand'] - sd['prev_qty']).mean():.2f}")

# ------------------------------------------------------------------ E
section("E. DO ZERO-STOCK PERIODS ACTUALLY SELL LESS?")
oos_days = inv.loc[inv["out_of_stock"] == 1, "days_valid"].sum()
in_days = inv.loc[inv["out_of_stock"] == 0, "days_valid"].sum()
u_oos = cov.loc[cov["qty_on_hand"] <= 0, "gross_qty"].sum()
u_in = cov.loc[cov["qty_on_hand"] > 0, "gross_qty"].sum()
print(f"Stock <= 0: {u_oos:,.0f} units over {oos_days:,.0f} record-days "
      f"= {u_oos / oos_days * 1000:.2f} units per 1,000 record-days")
print(f"Stock  > 0: {u_in:,.0f} units over {in_days:,.0f} record-days "
      f"= {u_in / in_days * 1000:.2f} units per 1,000 record-days")
print("-> If these rates are close, 'stock <= 0' does not mean the item cannot sell.")
