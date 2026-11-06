import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

PROC, REP, MODELS = Path("data/processed"), Path("reports"), Path("models")
MODELS.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)
PAIR = ["product_no", "store"]
LEADS = [7, 14, 28]
LEVELS = [0.90, 0.95, 0.99]
WINDOWS = [8, 16, 26]
LIVE_W, LIVE_METHOD = 16, "gamma"
HEAD_L, HEAD_Q = 14, 0.95
REVIEW_DAYS = 7
BASE = pd.Timestamp("2025-06-01")
NW = 48
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def pct(q):
    return int(round(q * 100))

# ------------------------------------------------------------------ A
section("A. WEEKLY UNITS PER PRODUCT-STORE PAIR")
sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"],
                    usecols=["transaction_date"] + PAIR + ["gross_qty"])
sales = sales[sales["gross_qty"] > 0].copy()
inv = pd.read_csv(PROC / "inventory_intervals_clean.csv",
                  usecols=PAIR + ["product_category", "start_date", "end_date",
                                  "qty_on_hand", "unit_cost"],
                  parse_dates=["start_date", "end_date"])
lab = pd.read_csv(PROC / "stockout_labels.csv.gz",
                  usecols=PAIR + ["asof", "qty_on_hand", "obs14", "y14_v2"],
                  parse_dates=["asof"])
cutoff = inv["end_date"].max()

pairs = inv.groupby(PAIR, as_index=False)["product_category"].first()
pairs["pid"] = np.arange(len(pairs))
total_units = sales["gross_qty"].sum()
sales["week_end"] = sales["transaction_date"] + pd.to_timedelta(
    6 - sales["transaction_date"].dt.weekday, unit="D")
sales["widx"] = ((sales["week_end"] - BASE).dt.days // 7).astype(int)
sales = sales.merge(pairs[PAIR + ["pid"]], on=PAIR, how="inner")
print(f"Data cutoff: {cutoff.date()} | pairs in inventory: {len(pairs):,}")
print(f"Gross units in sales: {total_units:,.0f} | units on pairs that exist in inventory: "
      f"{sales['gross_qty'].sum():,.0f} ({sales['gross_qty'].sum() / total_units * 100:.1f}%)")
print(f"Week index range in sales: {sales['widx'].min()} to {sales['widx'].max()} "
      f"(weeks 1 to 46 are full weeks; 0 and 47 are partial and never used)")

S = np.zeros((len(pairs), NW), dtype=np.float64)
np.add.at(S, (sales["pid"].to_numpy(), sales["widx"].to_numpy()), sales["gross_qty"].to_numpy(float))
C = np.concatenate([np.zeros((len(pairs), 1)), S.cumsum(axis=1)], axis=1)

def prior_params(df, col, keys):
    g = df.groupby(keys)[col]
    m = np.maximum(g.transform("mean").to_numpy(float), 1e-3)
    v = g.transform("var").fillna(0).to_numpy(float)
    excess = v - m
    a0 = np.where(excess > 1e-9, m ** 2 / np.maximum(excess, 1e-9), 20.0)
    return m, np.clip(a0, 0.05, 20.0)

def quant(method, x, m, a0, W, L, q):
    t = L / (7.0 * W)
    if method == "poisson":
        r = poisson.ppf(q, x * t)
    else:
        beta = a0 / m + 1.0
        r = nbinom.ppf(q, a0 + x, beta / (beta + t))
    if not np.isfinite(r).all():
        raise ValueError("non-finite quantile")
    return r

def pmean(method, x, m, a0, W, L):
    t = L / (7.0 * W)
    return x * t if method == "poisson" else (a0 + x) / (a0 / m + 1.0) * t

# ------------------------------------------------------------------ B
section("B. BACKTEST: DOES THE RULE COVER DEMAND IN THE NEXT L DAYS?")
lab = lab.merge(pairs[PAIR + ["pid", "product_category"]], on=PAIR, how="left")
lab["a"] = ((lab["asof"] - BASE).dt.days // 7).astype(int)
bt = lab[lab["a"].isin(range(26, 43, 2))].reset_index(drop=True)
pid, a = bt["pid"].to_numpy(), bt["a"].to_numpy()
for W in WINDOWS:
    bt[f"u{W}"] = C[pid, a + 1] - C[pid, a + 1 - W]
for L in LEADS:
    bt[f"f{L}"] = C[pid, a + 1 + L // 7] - C[pid, a + 1]
print(f"As-of Sundays used: {bt['asof'].nunique()} "
      f"({bt['asof'].min().date()} to {bt['asof'].max().date()}, every 2nd week)")
print(f"In-stock pair-weeks: {len(bt):,} | average stock on hand: {bt['qty_on_hand'].mean():.2f}")
for L in LEADS:
    f = bt[f"f{L}"]
    print(f"  next {L:>2} days: {(f > 0).mean() * 100:.1f}% of pair-weeks sell at least 1 unit "
          f"| mean units {f.mean():.3f}")

rows = []
for W in WINDOWS:
    x = bt[f"u{W}"].to_numpy(float)
    m, a0 = prior_params(bt, f"u{W}", ["a", "product_category"])
    for method in ["poisson", "gamma"]:
        for L in LEADS:
            f = bt[f"f{L}"].to_numpy(float)
            pos = f > 0
            for q in LEVELS:
                rop = quant(method, x, m, a0, W, L, q)
                rows.append({"method": method, "window_wks": W, "lead_days": L,
                             "service_%": pct(q),
                             "coverage_all_%": (f <= rop).mean() * 100,
                             "coverage_if_demand_%": (f[pos] <= rop[pos]).mean() * 100,
                             "unit_fill_%": np.minimum(f, rop).sum() / f.sum() * 100,
                             "avg_rop": rop.mean()})
res = pd.DataFrame(rows)
res.round(3).to_csv(REP / "17_rop_backtest.csv", index=False)
for col, title in [("coverage_if_demand_%", "Coverage among pair-weeks that really sold in the window (%)"),
                   ("unit_fill_%", "Share of the window's units that stay within the reorder point (%)"),
                   ("avg_rop", "Average reorder point per pair (units)")]:
    pv = res.pivot_table(index=["method", "window_wks"], columns=["lead_days", "service_%"], values=col)
    print(f"\n{title}")
    print(pv.round(1 if col != "avg_rop" else 2).to_string())
print(f"\nSaved: {REP / '17_rop_backtest.csv'}")
print("Nominal service levels are 90, 95 and 99. Coverage among selling pair-weeks is the tough test.")

# ------------------------------------------------------------------ C
section("C. REORDER POINT AS A STOCKOUT FLAG vs THE 14-DAY SELL-THROUGH LABEL")
ok = (bt["obs14"] == 1).to_numpy()
y = bt["y14_v2"].to_numpy()[ok]
oh = bt["qty_on_hand"].to_numpy()[ok]
x = bt[f"u{LIVE_W}"].to_numpy(float)
m, a0 = prior_params(bt, f"u{LIVE_W}", ["a", "product_category"])
out = []
def flag_row(name, flag):
    out.append({"rule": name, "flagged_%": flag.mean() * 100,
                "precision_%": y[flag].mean() * 100 if flag.any() else np.nan,
                "recall_%": flag[y == 1].mean() * 100,
                "lift": (y[flag].mean() / y.mean()) if flag.any() else np.nan})
flag_row("baseline: stock == 1", oh == 1)
for method in ["poisson", "gamma"]:
    for q in LEVELS:
        rop = quant(method, x, m, a0, LIVE_W, 14, q)[ok]
        flag_row(f"{method} {LIVE_W}wk, stock <= ROP(14d, {pct(q)}%)", oh <= rop)
print(f"Rows: {len(y):,} | 14-day positive rate: {y.mean() * 100:.2f}%")
print(pd.DataFrame(out).set_index("rule").round(2).to_string())

# ------------------------------------------------------------------ D
section(f"D. CURRENT SNAPSHOT ({cutoff.date()}), PROVISIONAL: {LIVE_METHOD}, {LIVE_W}-week window")
cur = (inv[(inv["start_date"] <= cutoff) & (inv["end_date"] >= cutoff)]
       .sort_values("start_date").drop_duplicates(PAIR, keep="last")
       .merge(pairs[PAIR + ["pid"]], on=PAIR, how="left").reset_index(drop=True))
A_LIVE = 46
cur["units_recent"] = C[cur["pid"].to_numpy(), A_LIVE + 1] - C[cur["pid"].to_numpy(), A_LIVE + 1 - LIVE_W]
xl = cur["units_recent"].to_numpy(float)
ml, a0l = prior_params(cur, "units_recent", ["product_category"])
ohl = cur["qty_on_hand"].to_numpy(float)
print(f"Pairs tracked on the cutoff date: {len(cur):,}")
print(f"  stock <= 0: {(ohl <= 0).sum():,} | stock == 1: {(ohl == 1).sum():,} | "
      f"stock >= 4: {(ohl >= 4).sum():,}")
print(f"  with any sale in the last {LIVE_W} weeks: {(xl > 0).mean() * 100:.1f}%")

for L in LEADS:
    for q in LEVELS:
        cur[f"rop_L{L}_s{pct(q)}"] = quant(LIVE_METHOD, xl, ml, a0l, LIVE_W, L, q)
cur["expected_demand_head"] = pmean(LIVE_METHOD, xl, ml, a0l, LIVE_W, HEAD_L)
head = f"rop_L{HEAD_L}_s{pct(HEAD_Q)}"
cur["safety_stock_head"] = np.maximum(cur[head] - cur["expected_demand_head"], 0)
cur["order_up_to"] = quant(LIVE_METHOD, xl, ml, a0l, LIVE_W, HEAD_L + REVIEW_DAYS, HEAD_Q)
rop_h = cur[head].to_numpy()
rop_ex = cur["rop_L28_s99"].to_numpy()
cur["status"] = np.select(
    [(ohl <= 0) & (xl > 0), ohl <= 0, ohl <= rop_h, ohl > rop_ex],
    ["out of stock, selling", "out of stock, idle", "reorder", "excess"], default="ok")
need = cur["status"].isin(["out of stock, selling", "reorder"])
cur["suggested_order"] = np.where(need, np.maximum(cur["order_up_to"] - np.maximum(ohl, 0), 0), 0)
cur["excess_units"] = np.maximum(ohl - rop_ex, 0)
cur["excess_value"] = cur["excess_units"] * cur["unit_cost"]

print(f"\nStatus counts (headline: lead {HEAD_L} days, {pct(HEAD_Q)}% service):")
sc = cur["status"].value_counts()
print(pd.DataFrame({"pairs": sc, "share_%": (sc / len(cur) * 100).round(1)}).to_string())
print(f"Suggested order units in total: {cur['suggested_order'].sum():,.0f}")
print(f"Excess units above the highest scenario (28d, 99%): {cur['excess_units'].sum():,.0f} "
      f"| at cost: {cur['excess_value'].sum():,.0f}")
print("\nTop 10 pairs by suggested order:")
print(cur.sort_values("suggested_order", ascending=False)
      [PAIR + ["product_category", "qty_on_hand", "units_recent", head, "order_up_to",
               "suggested_order", "status"]].head(10).to_string(index=False))

# ------------------------------------------------------------------ E
section("E. SENSITIVITY TO THE ASSUMED LEAD TIME AND SERVICE LEVEL")
srows = []
ohe = np.maximum(ohl, 0)
for L in LEADS:
    for q in LEVELS:
        r = cur[f"rop_L{L}_s{pct(q)}"].to_numpy()
        below = (ohl <= r) & (xl > 0)
        srows.append({"lead_days": L, "service_%": pct(q), "avg_rop": r.mean(),
                      "selling_pairs_at_or_below_rop": int(below.sum()),
                      "share_of_selling_pairs_%": below.sum() / max((xl > 0).sum(), 1) * 100,
                      "shortfall_units": np.maximum(r - ohe, 0)[xl > 0].sum()})
print(pd.DataFrame(srows).round(2).to_string(index=False))
print("-> The lead time is an assumption (no retail order dates exist). Read the spread across rows.")

keep = (PAIR + ["product_category", "qty_on_hand", "units_recent", "expected_demand_head",
                "safety_stock_head", "order_up_to", "suggested_order", "status",
                "excess_units", "excess_value"]
        + [f"rop_L{L}_s{pct(q)}" for L in LEADS for q in LEVELS])
cur[keep].round(3).to_csv(PROC / "reorder_recommendations.csv", index=False)
with open(MODELS / "inventory_policy.json", "w") as fh:
    json.dump({"method": LIVE_METHOD, "window_weeks": LIVE_W, "lead_days": LEADS,
               "service_levels": LEVELS, "headline": [HEAD_L, HEAD_Q],
               "review_days": REVIEW_DAYS, "as_of_stock_date": str(cutoff.date()),
               "demand_through": "2026-04-19",
               "notes": "lead times are assumptions; on-hand stock only, no open orders known"},
              fh, indent=2)
print(f"\nSaved: {PROC / 'reorder_recommendations.csv'} and {MODELS / 'inventory_policy.json'}")
