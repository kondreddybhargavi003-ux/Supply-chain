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
inv["next_start"] = g["start_date"].shift(-1)
g = inv.groupby(pair)
inv["next_gap0"] = g["gap_days"].shift(-1) == 0
cutoff = inv["end_date"].max()

left = (sales[sales["gross_qty"] > 0][["transaction_date", "product_no", "store", "gross_qty"]]
        .sort_values("transaction_date"))
right = inv[["start_date", "end_date", "product_no", "store", "qty_on_hand", "prev_qty",
             "gap_days", "next_start"]].sort_values("start_date")
m = pd.merge_asof(left, right, left_on="transaction_date", right_on="start_date", by=pair)
m["covered"] = m["qty_on_hand"].notna() & (m["transaction_date"] <= m["end_date"])

# ------------------------------------------------------------------ A
section("A. SALES WITH NO COVERING RECORD: WHAT DID THE ENDED RECORD LOOK LIKE?")
unm = m[m["qty_on_hand"].notna() & ~m["covered"]]
print(f"Unmatched sales that have an earlier (ended) record: {len(unm):,}")
labels = ["<=0", "1", "2", "3", "4+"]
b = pd.cut(unm["qty_on_hand"], bins=[-1000, 0, 1, 2, 3, 100000], labels=labels)
print("\nStock quantity in the ended record (%):")
print((b.value_counts(normalize=True).reindex(labels) * 100).round(1).to_string())
print(f"\nPair never has another record after the sale: {unm['next_start'].isna().mean() * 100:.1f}%")
print(f"Pair reappears later: {unm['next_start'].notna().mean() * 100:.1f}%")
print("-> If most ended records have stock 1, a pair may vanish when its last unit sells.")

# ------------------------------------------------------------------ B
section("B. CORRECTED CHECK: SALES RATE BY STOCK BEFORE THE SALE")
cov = m[m["covered"]].copy()
sd = cov[(cov["transaction_date"] == cov["start_date"])
         & cov["prev_qty"].notna() & (cov["gap_days"] == 0)].copy()
sd["prev_oos"] = (sd["prev_qty"] <= 0).astype(int)
num = sd.groupby("prev_oos")["gross_qty"].agg(["count", "sum"])
exp = inv[inv["next_gap0"]].groupby("out_of_stock")["days_valid"].sum()
res = pd.DataFrame({"sales_rows": num["count"], "units": num["sum"], "record_days": exp})
res["units_per_1000_days"] = (res["units"] / res["record_days"] * 1000).round(2)
res.index = res.index.map({0: "stock before sale > 0", 1: "stock before sale <= 0"})
print(res.to_string())
print("-> A much lower rate at stock <= 0 means recorded zero stock really blocks sales.")

# ------------------------------------------------------------------ C
section("C. STOCKOUT LABEL: IN STOCK AT WEEK END, ZERO RECORD WITHIN H DAYS?")
oos_starts = (inv.loc[inv["out_of_stock"] == 1, pair + ["start_date"]]
              .rename(columns={"start_date": "oos_start"}))
asof_dates = pd.date_range("2025-06-08", cutoff, freq="W-SUN")
parts = []
for t in asof_dates:
    s = inv[(inv["start_date"] <= t) & (inv["end_date"] >= t) & (inv["out_of_stock"] == 0)]
    s = s[pair + ["qty_on_hand", "end_date", "end_is_open", "next_start"]].copy()
    s["asof"] = t
    parts.append(s)
state = pd.concat(parts, ignore_index=True)
print(f"As-of dates (week ends): {len(asof_dates)} | in-stock pair-weeks: {len(state):,}")

j = state[pair + ["asof"]].reset_index().merge(oos_starts, on=pair, how="left")
for H in [7, 14, 28]:
    win = (j["oos_start"] > j["asof"]) & (j["oos_start"] <= j["asof"] + pd.Timedelta(days=H))
    hit = j.loc[win, "index"].unique()
    state[f"y{H}"] = state.index.isin(hit).astype(int)

for H in [7, 14, 28]:
    ok = state[state["asof"] + pd.Timedelta(days=H) <= cutoff]
    cens = ((ok["end_date"] < ok["asof"] + pd.Timedelta(days=H))
            & ok["next_start"].isna() & (ok["end_is_open"] == 0))
    print(f"\nHorizon {H} days:")
    print(f"  pair-weeks: {len(ok):,} | positives: {ok[f'y{H}'].sum():,} "
          f"| positive rate: {ok[f'y{H}'].mean() * 100:.2f}%")
    print(f"  pairs that stop being tracked before the window ends: {cens.mean() * 100:.1f}%")

ok = state[state["asof"] + pd.Timedelta(days=14) <= cutoff].copy()
lab = ["1", "2", "3", "4+"]
ok["q"] = pd.cut(ok["qty_on_hand"], bins=[0, 1, 2, 3, 100000], labels=lab)
t14 = ok.groupby("q", observed=True)["y14"].agg(pair_weeks="count", rate_pct=lambda x: x.mean() * 100)
print("\nH=14: positive rate by stock on hand at week end:")
print(t14.round(2).to_string())
mo = ok.assign(month=ok["asof"].dt.to_period("M")).groupby("month")["y14"].agg(
    pair_weeks="count", rate_pct=lambda x: x.mean() * 100)
print("\nH=14: positive rate by month of the as-of date:")
print(mo.round(2).to_string())
