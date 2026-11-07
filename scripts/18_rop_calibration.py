from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

PROC, REP = Path("data/processed"), Path("reports")
REP.mkdir(exist_ok=True)
PAIR = ["product_no", "store"]
LEADS = [7, 14, 28]
WINDOWS = [16, 26]
METHODS = ["poisson", "gamma"]
Q_GRID = [0.90, 0.95, 0.99, 0.995, 0.999, 0.9999]
TARGETS = [0.90, 0.95]
BASE = pd.Timestamp("2025-06-01")
NW = 48
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# ------------------------------------------------------------------ A
section("A. REBUILD THE BACKTEST TABLE (same rows as script 17)")
sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"],
                    usecols=["transaction_date"] + PAIR + ["gross_qty"])
sales = sales[sales["gross_qty"] > 0].copy()
inv = pd.read_csv(PROC / "inventory_intervals_clean.csv",
                  usecols=PAIR + ["product_category"])
lab = pd.read_csv(PROC / "stockout_labels.csv.gz",
                  usecols=PAIR + ["asof", "qty_on_hand"], parse_dates=["asof"])
pairs = inv.groupby(PAIR, as_index=False)["product_category"].first()
pairs["pid"] = np.arange(len(pairs))
sales["week_end"] = sales["transaction_date"] + pd.to_timedelta(
    6 - sales["transaction_date"].dt.weekday, unit="D")
sales["widx"] = ((sales["week_end"] - BASE).dt.days // 7).astype(int)
sales = sales.merge(pairs[PAIR + ["pid"]], on=PAIR, how="inner")
S = np.zeros((len(pairs), NW))
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

lab = lab.merge(pairs[PAIR + ["pid", "product_category"]], on=PAIR, how="left")
lab["a"] = ((lab["asof"] - BASE).dt.days // 7).astype(int)
bt = lab[lab["a"].isin(range(26, 43, 2))].reset_index(drop=True)
pid, a = bt["pid"].to_numpy(), bt["a"].to_numpy()
for W in WINDOWS:
    bt[f"u{W}"] = C[pid, a + 1] - C[pid, a + 1 - W]
for L in LEADS:
    bt[f"f{L}"] = C[pid, a + 1 + L // 7] - C[pid, a + 1]
early = (bt["a"] <= 34).to_numpy()
periods = {"early": early, "late": ~early}
for p, mk in periods.items():
    d = bt[mk]
    print(f"{p}: {d['asof'].nunique()} as-of Sundays ({d['asof'].min().date()} to "
          f"{d['asof'].max().date()}) | pair-weeks {len(d):,} | average stock on hand {d['qty_on_hand'].mean():.2f}")
print("Rule: pick the quantile level on the EARLY weeks, then test it on the LATE weeks.")

# ------------------------------------------------------------------ B
section("B. FULL GRID (all pair-weeks; fill = share of units within the reorder point)")
rows = []
for W in WINDOWS:
    x = bt[f"u{W}"].to_numpy(float)
    m, a0 = prior_params(bt, f"u{W}", ["a", "product_category"])
    for method in METHODS:
        for L in LEADS:
            f = bt[f"f{L}"].to_numpy(float)
            for q in Q_GRID:
                rop = quant(method, x, m, a0, W, L, q)
                for p, mk in periods.items():
                    rows.append({"method": method, "window_wks": W, "lead_days": L, "q": q,
                                 "period": p,
                                 "coverage_all_%": (f[mk] <= rop[mk]).mean() * 100,
                                 "unit_fill_%": np.minimum(f[mk], rop[mk]).sum() / f[mk].sum() * 100,
                                 "avg_rop": rop[mk].mean()})
res = pd.DataFrame(rows)
res.round(3).to_csv(REP / "18_rop_calibration_grid.csv", index=False)
for L in [14]:
    sub = res[res["lead_days"] == L]
    for col, nd in [("coverage_all_%", 1), ("unit_fill_%", 1), ("avg_rop", 2)]:
        pv = sub.pivot_table(index=["method", "window_wks", "period"], columns="q", values=col)
        print(f"\nLead time {L} days: {col}")
        print(pv.round(nd).to_string())
print(f"\nFull grid for all lead times saved: {REP / '18_rop_calibration_grid.csv'}")

# ------------------------------------------------------------------ C
section("C. CALIBRATION: SMALLEST q THAT REACHES THE TARGET FILL ON THE EARLY WEEKS")
out = []
for (method, W, L), g in res.groupby(["method", "window_wks", "lead_days"]):
    for T in TARGETS:
        e = g[(g["period"] == "early") & (g["unit_fill_%"] >= T * 100)].sort_values("q")
        row = {"method": method, "window_wks": W, "lead_days": L, "target_fill_%": int(T * 100)}
        if len(e) == 0:
            row.update({"q_needed": np.nan})
        else:
            q = e.iloc[0]["q"]
            late = g[(g["period"] == "late") & (g["q"] == q)].iloc[0]
            row.update({"q_needed": q, "early_fill_%": e.iloc[0]["unit_fill_%"],
                        "early_avg_rop": e.iloc[0]["avg_rop"],
                        "late_fill_%": late["unit_fill_%"], "late_avg_rop": late["avg_rop"]})
        out.append(row)
cal = pd.DataFrame(out)
print(cal.round(4).to_string(index=False))
cal.round(4).to_csv(REP / "18_rop_calibration_summary.csv", index=False)
print("\nq_needed empty = the target was not reached even at q = 0.9999.")
print("Compare late_fill_% with the target: that is the honest out-of-sample result.")
print("late_avg_rop is the cost: the average units held as reorder point per pair.")
