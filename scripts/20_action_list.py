import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom

PROC, REP, MODELS = Path("data/processed"), Path("reports"), Path("models")
REP.mkdir(exist_ok=True)
PAIR = ["product_no", "store"]
LEADS = [7, 14, 28]
W = 16
Q_STD, Q_HIGH = 0.995, 0.999
HEAD_L, REVIEW_DAYS, MIN_HOLD = 14, 7, 2
P_HIGH, P_MED = 0.30, 0.10
BASE = pd.Timestamp("2025-06-01")
NW, A_LIVE = 48, 46
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"],
                    usecols=["transaction_date"] + PAIR + ["gross_qty"])
sales = sales[sales["gross_qty"] > 0].copy()
inv = pd.read_csv(PROC / "inventory_intervals_clean.csv",
                  usecols=PAIR + ["product_division", "product_category", "supplier", "start_date",
                                  "end_date", "qty_on_hand", "unit_price", "unit_cost"],
                  parse_dates=["start_date", "end_date"])
lab = pd.read_csv(PROC / "stockout_labels.csv.gz", usecols=PAIR + ["asof", "qty_on_hand"],
                  parse_dates=["asof"])
cutoff = inv["end_date"].max()
pairs = inv.groupby(PAIR, as_index=False)["product_category"].first()
pairs["pid"] = np.arange(len(pairs))
sales["week_end"] = sales["transaction_date"] + pd.to_timedelta(
    6 - sales["transaction_date"].dt.weekday, unit="D")
sales["widx"] = ((sales["week_end"] - BASE).dt.days // 7).astype(int)
sales = sales.merge(pairs[PAIR + ["pid"]], on=PAIR, how="inner")
S = np.zeros((len(pairs), NW))
np.add.at(S, (sales["pid"].to_numpy(), sales["widx"].to_numpy()), sales["gross_qty"].to_numpy(float))
C = np.concatenate([np.zeros((len(pairs), 1)), S.cumsum(axis=1)], axis=1)

def prior(df, col, keys):
    g = df.groupby(keys)[col]
    m = np.maximum(g.transform("mean").to_numpy(float), 1e-3)
    v = g.transform("var").fillna(0).to_numpy(float)
    a0 = np.clip(np.where(v - m > 1e-9, m ** 2 / np.maximum(v - m, 1e-9), 20.0), 0.05, 20.0)
    return m, a0

def nb_params(x, m, a0, L):
    t = L / (7.0 * W)
    beta = a0 / m + 1.0
    return a0 + x, beta / (beta + t)

def rop(x, m, a0, L, q):
    r, p = nb_params(x, m, a0, L)
    out = nbinom.ppf(q, r, p)
    if not np.isfinite(out).all():
        raise ValueError("non-finite quantile")
    return out

# ------------------------------------------------------------------ A
section("A. BACKTEST: IS ONE QUANTILE EQUALLY GOOD FOR SLOW AND FAST SELLERS?")
lab = lab.merge(pairs, on=PAIR, how="left")
lab["a"] = ((lab["asof"] - BASE).dt.days // 7).astype(int)
bt = lab[lab["a"].isin(range(26, 43, 2))].reset_index(drop=True)
pid, a = bt["pid"].to_numpy(), bt["a"].to_numpy()
bt["u"] = C[pid, a + 1] - C[pid, a + 1 - W]
bt["f"] = C[pid, a + 1 + HEAD_L // 7] - C[pid, a + 1]
xb = bt["u"].to_numpy(float)
mb, ab = prior(bt, "u", ["a", "product_category"])
bt["tier"] = pd.cut(bt["u"], [-1, 0, 1, 4, 10000], labels=["0 units", "1 unit", "2-4 units", "5+ units"])
f = bt["f"].to_numpy(float)
rows = []
for name, q in [("standard", Q_STD), ("high", Q_HIGH)]:
    r = rop(xb, mb, ab, HEAD_L, q)
    bt["rop"] = r
    bt["covered"] = np.minimum(f, r)
    t = bt.groupby("tier", observed=True).agg(pair_weeks=("u", "size"), mean_future_14d=("f", "mean"),
                                              avg_rop=("rop", "mean"), covered=("covered", "sum"),
                                              demand=("f", "sum"), avg_stock=("qty_on_hand", "mean"))
    t["unit_fill_%"] = t["covered"] / t["demand"] * 100
    t["policy"] = name
    rows.append(t.drop(columns=["covered", "demand"]).reset_index())
print("Pair-weeks by units sold in the previous 16 weeks (lead time 14 days):")
print(pd.concat(rows).set_index(["policy", "tier"]).round(2).to_string())
print("-> If fast sellers show ~99% fill with ROP far above their demand, they are over-protected.")

# ------------------------------------------------------------------ B
section("B. CURRENT SNAPSHOT: PROBABILITY OF SELLING OUT WITHIN THE LEAD TIME")
cur = (inv[(inv["start_date"] <= cutoff) & (inv["end_date"] >= cutoff)]
       .sort_values("start_date").drop_duplicates(PAIR, keep="last")
       .merge(pairs[PAIR + ["pid"]], on=PAIR, how="left").reset_index(drop=True))
pc = cur["pid"].to_numpy()
cur["units_recent"] = C[pc, A_LIVE + 1] - C[pc, A_LIVE + 1 - W]
x = cur["units_recent"].to_numpy(float)
oh = cur["qty_on_hand"].to_numpy(float)
m, a0 = prior(cur, "units_recent", ["product_category"])
r14, p14 = nb_params(x, m, a0, HEAD_L)
oh_eff = np.maximum(oh, 1)
cur["p_sellout_14d"] = nbinom.sf(oh_eff - 1, r14, p14)
rop_s, rop_h = rop(x, m, a0, HEAD_L, Q_STD), rop(x, m, a0, HEAD_L, Q_HIGH)
order_up_to = rop(x, m, a0, HEAD_L + REVIEW_DAYS, Q_STD)
excess_line = np.maximum(rop(x, m, a0, 28, Q_HIGH), MIN_HOLD)
sell, pos = x > 0, oh > 0
pp = cur.loc[sell & pos, "p_sellout_14d"]
print(f"Pairs: {len(cur):,} | selling (sale in last {W} weeks): {sell.sum():,} | "
      f"selling with stock > 0: {(sell & pos).sum():,} | no recent sales and stock > 0: {((~sell) & pos).sum():,}")
print("Probability of selling out within 14 days, selling pairs with stock > 0:")
print(pp.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(3).to_string())
print("\nHow many selling pairs (stock > 0) exceed each probability threshold:")
print(pd.DataFrame({"threshold": [0.05, 0.10, 0.20, 0.30, 0.50],
                    "pairs": [int((pp >= t).sum()) for t in [0.05, 0.10, 0.20, 0.30, 0.50]]})
      .assign(share_of_selling_pct=lambda d: (d["pairs"] / len(pp) * 100).round(1)).to_string(index=False))
print(f"Out of stock and selling: {int((sell & ~pos).sum()):,}")
print(f"Previous rule (stock <= ROP) flagged {int((oh <= rop_s).sum()):,} pairs, "
      f"{int(((oh <= rop_s) & ~sell & pos).sum()):,} of them with NO sale in {W} weeks.")

cur["status"] = np.select(
    [(oh <= 0) & sell, oh <= 0,
     sell & pos & (cur["p_sellout_14d"].to_numpy() >= P_HIGH),
     sell & pos & (cur["p_sellout_14d"].to_numpy() >= P_MED),
     sell & pos & (oh <= rop_h),
     oh > excess_line, ~sell],
    ["urgent: out of stock, selling", "out of stock, idle", "reorder now", "reorder soon",
     "watch", "above recent demand cover", "no recent sales"], default="ok")
act = cur["status"].isin(["urgent: out of stock, selling", "reorder now", "reorder soon"])
cur["rop_standard"], cur["rop_high"], cur["order_up_to"] = rop_s, rop_h, order_up_to
cur["expected_demand_14d"] = (a0 + x) / (a0 / m + 1.0) * (HEAD_L / (7.0 * W))
cur["suggested_order"] = np.where(act, np.maximum(order_up_to - np.maximum(oh, 0), 0), 0)
cur["excess_units"] = np.maximum(oh - excess_line, 0)
cur["excess_value_at_cost"] = cur["excess_units"] * cur["unit_cost"]
cur["priority_rank"] = np.nan
order = cur[act].sort_values(["p_sellout_14d", "suggested_order"], ascending=False).index
cur.loc[order, "priority_rank"] = np.arange(1, len(order) + 1)

print(f"\nStatus counts (reorder now: probability >= {P_HIGH}, reorder soon: >= {P_MED}):")
sc = cur["status"].value_counts()
print(pd.DataFrame({"pairs": sc, "share_%": (sc / len(cur) * 100).round(1)}).to_string())
print(f"\nActionable pairs: {act.sum():,} | suggested order units: {cur['suggested_order'].sum():,.0f}")
print(f"Above recent demand cover: {(cur['excess_units'] > 0).sum():,} pairs | "
      f"{cur['excess_units'].sum():,.0f} units | at cost {cur['excess_value_at_cost'].sum():,.0f} "
      f"(NOT a savings estimate)")
print("\nTop 15 by priority:")
cols = PAIR + ["product_category", "qty_on_hand", "units_recent", "p_sellout_14d", "order_up_to",
               "suggested_order", "status"]
print(cur[act].sort_values("priority_rank")[cols].head(15).round(3).to_string(index=False))

# ------------------------------------------------------------------ C
section("C. WHERE IS THE WORK?")
cur["actionable"] = act.astype(int)
for key in ["product_category", "store", "supplier"]:
    t = cur.groupby(key).agg(pairs=("pid", "size"), actionable=("actionable", "sum"),
                             suggested_units=("suggested_order", "sum"),
                             excess_units=("excess_units", "sum"))
    t["actionable_%"] = t["actionable"] / t["pairs"] * 100
    print(f"\nTop 10 by {key} (suggested units):")
    print(t.sort_values("suggested_units", ascending=False).head(10).round(1).to_string())

srows = []
for L in LEADS:
    for name, q in [("standard", Q_STD), ("high", Q_HIGH)]:
        rr = rop(x, m, a0, L, q)
        below = (oh <= rr) & sell
        srows.append({"lead_days": L, "policy": name, "avg_rop": rr.mean(),
                      "selling_pairs_at_or_below_rop": int(below.sum()),
                      "shortfall_units": np.maximum(rr - np.maximum(oh, 0), 0)[sell].sum()})
pd.DataFrame(srows).round(3).to_csv(REP / "20_sensitivity.csv", index=False)

keep = (PAIR + ["product_division", "product_category", "supplier", "qty_on_hand", "unit_price",
                "unit_cost", "units_recent", "expected_demand_14d", "p_sellout_14d", "rop_standard",
                "rop_high", "order_up_to", "suggested_order", "status", "priority_rank",
                "excess_units", "excess_value_at_cost"])
cur[keep].round(4).to_csv(PROC / "reorder_recommendations.csv", index=False)
with open(MODELS / "inventory_policy.json") as fh:
    pol = json.load(fh)
pol.update({"p_reorder_now": P_HIGH, "p_reorder_soon": P_MED,
            "ranking": "probability of selling out within 14 days (gamma-Poisson), then suggested order",
            "no_recent_sales_rule": f"pairs with no sale in {W} weeks are never flagged for reorder"})
with open(MODELS / "inventory_policy.json", "w") as fh:
    json.dump(pol, fh, indent=2)
print(f"\nSaved: {PROC / 'reorder_recommendations.csv'}, {REP / '20_sensitivity.csv'}, updated {MODELS / 'inventory_policy.json'}")
