from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

PROC, REP = Path("data/processed"), Path("reports")
PAIR = ["product_no", "store"]
W, HEAD_L = 16, 14
BASE = pd.Timestamp("2025-06-01")
NW = 48
SEED = 42
rng = np.random.default_rng(SEED)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# ------------------------------------------------------------------ A
section("A. SELL-OUT PROBABILITY ON THE SAME TEST ROWS AS THE ML MODEL")
sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"],
                    usecols=["transaction_date"] + PAIR + ["gross_qty"])
sales = sales[sales["gross_qty"] > 0].copy()
inv = pd.read_csv(PROC / "inventory_intervals_clean.csv", usecols=PAIR + ["product_category"])
pairs = inv.groupby(PAIR, as_index=False)["product_category"].first()
pairs["pid"] = np.arange(len(pairs))
sales["week_end"] = sales["transaction_date"] + pd.to_timedelta(
    6 - sales["transaction_date"].dt.weekday, unit="D")
sales["widx"] = ((sales["week_end"] - BASE).dt.days // 7).astype(int)
sales = sales.merge(pairs[PAIR + ["pid"]], on=PAIR, how="inner")
S = np.zeros((len(pairs), NW))
np.add.at(S, (sales["pid"].to_numpy(), sales["widx"].to_numpy()), sales["gross_qty"].to_numpy(float))
C = np.concatenate([np.zeros((len(pairs), 1)), S.cumsum(axis=1)], axis=1)

d = pd.read_csv(REP / "14_test_predictions.csv", parse_dates=["asof"])
d = d.merge(pairs, on=PAIR, how="left")
d["a"] = ((d["asof"] - BASE).dt.days // 7).astype(int)
pid, a = d["pid"].to_numpy(), d["a"].to_numpy()
d["u"] = C[pid, a + 1] - C[pid, a + 1 - W]
g = d.groupby(["a", "product_category"])["u"]
m = np.maximum(g.transform("mean").to_numpy(float), 1e-3)
v = g.transform("var").fillna(0).to_numpy(float)
a0 = np.clip(np.where(v - m > 1e-9, m ** 2 / np.maximum(v - m, 1e-9), 20.0), 0.05, 20.0)
t = HEAD_L / (7.0 * W)
beta = a0 / m + 1.0
d["p_sellout"] = nbinom.sf(np.maximum(d["qty_on_hand"].to_numpy(float), 1) - 1,
                           a0 + d["u"].to_numpy(float), beta / (beta + t))
print(f"Rows: {len(d):,} | as-of Sundays: {d['asof'].nunique()} "
      f"({d['asof'].min().date()} to {d['asof'].max().date()})")
print(f"Positive rate v2: {d['y2'].mean() * 100:.2f}% | v1: {d['y1'].mean() * 100:.2f}%")
print(f"Spearman correlation between ML score and sell-out probability: "
      f"{spearmanr(d['score'], d['p_sellout'])[0]:.3f}")

def rank01(s):
    return s.rank(method="average", pct=True).to_numpy()
d["blend"] = rank01(d["score"]) + rank01(d["p_sellout"])

def metrics(y, s, tops=(0.05, 0.20)):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    noisy = s + rng.random(len(s)) * 1e-9
    out = {"PR_AUC": average_precision_score(y, s), "ROC_AUC": roc_auc_score(y, s)}
    order = np.argsort(-noisy)
    for tp in tops:
        k = max(1, int(len(y) * tp))
        hit = y[order[:k]].sum()
        out[f"prec@{int(tp * 100)}%"] = hit / k * 100
        out[f"recall@{int(tp * 100)}%"] = hit / y.sum() * 100
        out[f"lift@{int(tp * 100)}%"] = (hit / k) / y.mean()
    return out

rows = {}
for lbl in ["y2", "y1"]:
    for name, col in [("baseline: 1/stock", "base"), ("ML model (XGBoost)", "score"),
                      ("sell-out probability", "p_sellout"), ("rank blend of both", "blend")]:
        rows[(lbl, name)] = metrics(d[lbl], d[col])
res = pd.DataFrame(rows).T
res.index.names = ["label", "ranking"]
print("\nTop-5% and top-20% lists, same rows, same label (y2 = v2 label, y1 = explicit zero records):")
print(res.round(3).to_string())
res.round(4).to_csv(REP / "21_ranking_comparison.csv")

# ------------------------------------------------------------------ B
section("B. IS THE PROBABILITY CALIBRATED? (predicted vs actual, label v2)")
d["decile"] = pd.qcut(d["p_sellout"].rank(method="first"), 10, labels=False) + 1
cal = d.groupby("decile").agg(rows=("y2", "size"), mean_p_pct=("p_sellout", lambda x: x.mean() * 100),
                              actual_pct=("y2", lambda x: x.mean() * 100))
print(cal.round(2).to_string())
bins = [0, 0.05, 0.10, 0.20, 0.30, 1.01]
d["pbin"] = pd.cut(d["p_sellout"], bins, right=False, labels=["<5%", "5-10%", "10-20%", "20-30%", ">=30%"])
tb = d.groupby("pbin", observed=True).agg(rows=("y2", "size"), mean_p_pct=("p_sellout", lambda x: x.mean() * 100),
                                          actual_pct=("y2", lambda x: x.mean() * 100))
print("\nBy the tier cut-offs used in the action list:")
print(tb.round(2).to_string())

# ------------------------------------------------------------------ C
section("C. STABILITY BY WEEK: PRECISION IN THE TOP 5% OF EACH WEEK")
wk = []
for tt, gdf in d.groupby("asof"):
    k = max(1, int(len(gdf) * 0.05))
    r = {"asof": tt.date(), "rows": len(gdf), "base_rate_%": gdf["y2"].mean() * 100}
    for name, col in [("ML", "score"), ("sellout_p", "p_sellout"), ("baseline", "base")]:
        top = gdf.assign(r=rng.random(len(gdf))).sort_values([col, "r"], ascending=[False, True]).head(k)
        r[f"{name}_prec@5%"] = top["y2"].mean() * 100
    wk.append(r)
print(pd.DataFrame(wk).round(2).to_string(index=False))
print("\nIf ML and sell-out probability are close, the simpler probability is easier to explain and maintain.")
