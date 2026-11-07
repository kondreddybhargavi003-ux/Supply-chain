import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom

PROC, REP, MODELS = Path("data/processed"), Path("reports"), Path("models")
MODELS.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)
PAIR = ["product_no", "store"]
LEADS = [7, 14, 28]
W = 16
Q_STD, Q_HIGH = 0.995, 0.999
HEAD_L = 14
REVIEW_DAYS = 7
MIN_HOLD = 2
BASE = pd.Timestamp("2025-06-01")
NW = 48
A_LIVE = 46
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# ------------------------------------------------------------------ A
section("A. CURRENT STOCK AND RECENT DEMAND")
sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"],
                    usecols=["transaction_date"] + PAIR + ["gross_qty"])
sales = sales[sales["gross_qty"] > 0].copy()
inv = pd.read_csv(PROC / "inventory_intervals_clean.csv",
                  usecols=PAIR + ["product_division", "product_category", "supplier", "start_date",
                                  "end_date", "qty_on_hand", "unit_price", "unit_cost"],
                  parse_dates=["start_date", "end_date"])
cutoff = inv["end_date"].max()
pairs = inv.groupby(PAIR, as_index=False)["product_category"].first()[PAIR]
pairs["pid"] = np.arange(len(pairs))
sales["week_end"] = sales["transaction_date"] + pd.to_timedelta(
    6 - sales["transaction_date"].dt.weekday, unit="D")
sales["widx"] = ((sales["week_end"] - BASE).dt.days // 7).astype(int)
sales = sales.merge(pairs, on=PAIR, how="inner")
S = np.zeros((len(pairs), NW))
np.add.at(S, (sales["pid"].to_numpy(), sales["widx"].to_numpy()), sales["gross_qty"].to_numpy(float))
C = np.concatenate([np.zeros((len(pairs), 1)), S.cumsum(axis=1)], axis=1)

cur = (inv[(inv["start_date"] <= cutoff) & (inv["end_date"] >= cutoff)]
       .sort_values("start_date").drop_duplicates(PAIR, keep="last")
       .merge(pairs, on=PAIR, how="left").reset_index(drop=True))
p = cur["pid"].to_numpy()
cur["units_recent"] = C[p, A_LIVE + 1] - C[p, A_LIVE + 1 - W]
x = cur["units_recent"].to_numpy(float)
oh = cur["qty_on_hand"].to_numpy(float)
g = cur.groupby("product_category")["units_recent"]
m = np.maximum(g.transform("mean").to_numpy(float), 1e-3)
v = g.transform("var").fillna(0).to_numpy(float)
a0 = np.clip(np.where(v - m > 1e-9, m ** 2 / np.maximum(v - m, 1e-9), 20.0), 0.05, 20.0)
print(f"Stock date: {cutoff.date()} | demand window: last {W} full weeks to 2026-04-19")
print(f"Pairs tracked: {len(cur):,} | stock <= 0: {(oh <= 0).sum():,} | stock == 1: {(oh == 1).sum():,} "
      f"| any sale in window: {(x > 0).mean() * 100:.1f}%")

def rop(L, q):
    t = L / (7.0 * W)
    beta = a0 / m + 1.0
    r = nbinom.ppf(q, a0 + x, beta / (beta + t))
    if not np.isfinite(r).all():
        raise ValueError("non-finite quantile")
    return r

def exp_demand(L):
    return (a0 + x) / (a0 / m + 1.0) * (L / (7.0 * W))

# ------------------------------------------------------------------ B
section("B. POLICY, STATUS AND SUGGESTED ORDERS")
rop_s, rop_h = rop(HEAD_L, Q_STD), rop(HEAD_L, Q_HIGH)
order_up_to = rop(HEAD_L + REVIEW_DAYS, Q_STD)
excess_line = np.maximum(rop(28, Q_HIGH), MIN_HOLD)
daily = exp_demand(1)
cover_days = np.where(daily > 0, np.maximum(oh, 0) / daily, np.inf)
cur["expected_demand_14d"] = exp_demand(HEAD_L)
cur["rop_standard"], cur["rop_high"], cur["order_up_to"] = rop_s, rop_h, order_up_to
cur["cover_days"] = np.where(np.isfinite(cover_days), cover_days, np.nan)
cur["status"] = np.select(
    [(oh <= 0) & (x > 0), oh <= 0, (cover_days <= 7) & (oh <= rop_s), oh <= rop_s,
     oh <= rop_h, oh > excess_line],
    ["urgent: out of stock, selling", "out of stock, idle", "urgent: under 7 days of cover",
     "reorder", "watch", "above recent demand cover"], default="ok")
act = cur["status"].isin(["urgent: out of stock, selling", "urgent: under 7 days of cover", "reorder"])
cur["suggested_order"] = np.where(act, np.maximum(order_up_to - np.maximum(oh, 0), 0), 0)
cur["excess_units"] = np.maximum(oh - excess_line, 0)
cur["excess_value_at_cost"] = cur["excess_units"] * cur["unit_cost"]
cur["priority_rank"] = np.nan
order = cur[act].sort_values(["suggested_order", "units_recent"], ascending=False).index
cur.loc[order, "priority_rank"] = np.arange(1, len(order) + 1)

print(f"Policy: gamma-Poisson, {W}-week window, standard quantile {Q_STD}, high quantile {Q_HIGH}, "
      f"lead time {HEAD_L} days (assumption), review {REVIEW_DAYS} days, minimum hold {MIN_HOLD} units")
sc = cur["status"].value_counts()
print(pd.DataFrame({"pairs": sc, "share_%": (sc / len(cur) * 100).round(1)}).to_string())
print(f"\nActionable pairs (urgent + reorder): {act.sum():,} | suggested order units: "
      f"{cur['suggested_order'].sum():,.0f}")
print(f"Pairs above recent demand cover: {(cur['excess_units'] > 0).sum():,} | units: "
      f"{cur['excess_units'].sum():,.0f} | at cost: {cur['excess_value_at_cost'].sum():,.0f}")
print("(excess = stock above the larger of the 28-day high-quantile ROP and the minimum hold; "
      "NOT a savings estimate)")
print(f"Average stock on hand: {oh.mean():.2f} | average standard ROP: {rop_s.mean():.2f} "
      f"| average high ROP: {rop_h.mean():.2f}")

print("\nTop 15 pairs by suggested order:")
cols = PAIR + ["product_category", "qty_on_hand", "units_recent", "rop_standard", "order_up_to",
               "suggested_order", "status"]
print(cur[act].sort_values("priority_rank")[cols].head(15).to_string(index=False))

# ------------------------------------------------------------------ C
section("C. WHERE IS THE WORK? (actionable pairs and excess units)")
cur["actionable"] = act.astype(int)
for key, n in [("product_category", 10), ("store", 10), ("supplier", 10)]:
    t = cur.groupby(key).agg(pairs=("pid", "size"), actionable=("actionable", "sum"),
                             suggested_units=("suggested_order", "sum"),
                             excess_units=("excess_units", "sum"))
    t["actionable_%"] = t["actionable"] / t["pairs"] * 100
    print(f"\nTop {n} by {key} (suggested units):")
    print(t.sort_values("suggested_units", ascending=False).head(n).round(1).to_string())

# ------------------------------------------------------------------ D
section("D. SENSITIVITY: LEAD TIME x POLICY (selling pairs only)")
rows = []
sell = x > 0
for L in LEADS:
    for name, q in [("standard", Q_STD), ("high", Q_HIGH)]:
        r = rop(L, q)
        below = (oh <= r) & sell
        rows.append({"lead_days": L, "policy": name, "avg_rop": r.mean(),
                     "selling_pairs_at_or_below_rop": int(below.sum()),
                     "share_of_selling_pairs_%": below.sum() / max(sell.sum(), 1) * 100,
                     "shortfall_units": np.maximum(r - np.maximum(oh, 0), 0)[sell].sum()})
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("-> Lead time is an assumption. Show the range, not one number.")

keep = (PAIR + ["product_division", "product_category", "supplier", "qty_on_hand", "unit_price",
                "unit_cost", "units_recent", "expected_demand_14d", "cover_days", "rop_standard",
                "rop_high", "order_up_to", "suggested_order", "status", "priority_rank",
                "excess_units", "excess_value_at_cost"])
cur[keep].round(3).to_csv(PROC / "reorder_recommendations.csv", index=False)
with open(MODELS / "inventory_policy.json", "w") as fh:
    json.dump({"method": "gamma-poisson", "window_weeks": W, "q_standard": Q_STD,
               "q_high": Q_HIGH, "headline_lead_days": HEAD_L, "lead_days_scenarios": LEADS,
               "review_days": REVIEW_DAYS, "min_hold_units": MIN_HOLD,
               "stock_date": str(cutoff.date()), "demand_through": "2026-04-19",
               "calibration": "q chosen on 2025-11-30..2026-01-25, tested 2026-02-08..2026-03-22; "
                              "quantile is a tuning level, not a guaranteed service level",
               "notes": "lead times assumed; on-hand only, no open orders known"}, fh, indent=2)
print(f"\nSaved: {PROC / 'reorder_recommendations.csv'} and {MODELS / 'inventory_policy.json'}")
