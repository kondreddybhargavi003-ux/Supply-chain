import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
PROC, MODELS, REP = Path("data/processed"), Path("models"), Path("reports")
FIG = REP / "figures"
MODELS.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
RF_ROWS = 250_000
PAIR = ["product_no", "store"]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
rng = np.random.default_rng(SEED)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def ns(s):
    return pd.to_datetime(s).astype("datetime64[ns]")

# ------------------------------------------------------------------ A
section("A. BUILD FEATURES (ONLY INFORMATION UP TO THE AS-OF SUNDAY)")
lab = pd.read_csv(PROC / "stockout_labels.csv.gz", parse_dates=["asof"])
lab = lab[lab["split14"].isin(["train", "val", "test"])].copy()
lab["asof"] = ns(lab["asof"])
lab = lab.sort_values("asof").reset_index(drop=True)

inv = pd.read_csv(PROC / "inventory_intervals_clean.csv", parse_dates=["start_date", "end_date"])
inv["start_date"] = ns(inv["start_date"])
inv = inv.sort_values(PAIR + ["start_date"]).reset_index(drop=True)
inv["prev_qty"] = inv.groupby(PAIR)["qty_on_hand"].shift()

sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"])
sales = sales[sales["gross_qty"] > 0][["transaction_date"] + PAIR + ["gross_qty"]].copy()
sales["transaction_date"] = ns(sales["transaction_date"])

rec = (inv[PAIR + ["start_date", "qty_on_hand", "prev_qty", "unit_price", "unit_cost",
                   "stock_status", "product_division", "product_category"]]
       .rename(columns={"qty_on_hand": "rec_qty"}).sort_values("start_date"))
df = pd.merge_asof(lab, rec, left_on="asof", right_on="start_date", by=PAIR)
print(f"Rows: {len(df):,}")
print(f"Rows with no matching inventory record: {df['start_date'].isna().sum():,}")
print(f"Rows where record stock differs from label-file stock: {(df['rec_qty'] != df['qty_on_hand']).sum():,}")
print("-> Both numbers should be 0.")

first = inv.groupby(PAIR)["start_date"].min().rename("pair_first").reset_index()
df = df.merge(first, on=PAIR, how="left")
df["days_in_record"] = (df["asof"] - df["start_date"]).dt.days
df["pair_age_days"] = (df["asof"] - df["pair_first"]).dt.days
df["has_prev"] = df["prev_qty"].notna().astype(float)
df["prev_qty_f"] = df["prev_qty"].fillna(df["qty_on_hand"])
df["qty_change"] = df["qty_on_hand"] - df["prev_qty_f"]
df["margin_pct"] = ((df["unit_price"] - df["unit_cost"]) / df["unit_price"] * 100).clip(-100, 100)

pairs = inv[PAIR].drop_duplicates().reset_index(drop=True)
pairs["pid"] = np.arange(len(pairs))
df = df.merge(pairs, on=PAIR, how="left")
base = pd.Timestamp("2025-06-08")
df["widx"] = ((df["asof"] - base).dt.days // 7).astype(int)

sales["week_end"] = sales["transaction_date"] + pd.to_timedelta(
    6 - sales["transaction_date"].dt.weekday, unit="D")
sales["widx"] = (sales["week_end"] - base).dt.days // 7
sales = sales.merge(pairs, on=PAIR, how="inner")
PAD, W = 8, 56
S = np.zeros((len(pairs), W), dtype=np.float32)
np.add.at(S, (sales["pid"].to_numpy(), (sales["widx"] + 1 + PAD).to_numpy()),
          sales["gross_qty"].to_numpy(dtype=np.float32))
C = S.cumsum(axis=1)
A = (S > 0).astype(np.float32).cumsum(axis=1)
p = df["pid"].to_numpy()
c = (df["widx"] + 1 + PAD).to_numpy()
df["units_1w"] = S[p, c]
df["units_4w"] = C[p, c] - C[p, c - 4]
df["units_8w"] = C[p, c] - C[p, c - 8]
df["active_weeks_8"] = A[p, c] - A[p, c - 8]
df["cover_weeks"] = df["qty_on_hand"] / (df["units_4w"] / 4 + 0.25)
print(f"Share of rows with any sale in the last 8 weeks: {(df['units_8w'] > 0).mean() * 100:.1f}%")

num = ["qty_on_hand", "days_in_record", "pair_age_days", "has_prev", "prev_qty_f", "qty_change",
       "unit_price", "unit_cost", "margin_pct", "units_1w", "units_4w", "units_8w",
       "active_weeks_8", "cover_weeks"]
cat = ["product_division", "product_category", "store", "stock_status"]
X = pd.concat([df[num].astype(np.float32), pd.get_dummies(df[cat], dtype=np.float32)], axis=1)
Y2, Y1 = df["y14_v2"].to_numpy(), df["y14_v1"].to_numpy()
tr = (df["split14"] == "train").to_numpy()
va = (df["split14"] == "val").to_numpy()
te = (df["split14"] == "test").to_numpy()
tv = tr | va
print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns | missing values: {int(X.isna().sum().sum())}")
print("\nRows and positive rates (%):")
print(pd.DataFrame({
    "rows": [tr.sum(), va.sum(), te.sum()],
    "v2_rate_%": [Y2[m].mean() * 100 for m in (tr, va, te)],
    "v1_rate_%": [Y1[m].mean() * 100 for m in (tr, va, te)]},
    index=["train", "val", "test"]).round(2).to_string())

def pr_metrics(y, s, top=0.05):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    k = max(1, int(len(y) * top))
    noisy = s + rng.random(len(s)) * 1e-9
    idx = np.argpartition(-noisy, k - 1)[:k]
    hit = y[idx].sum()
    return {"PR_AUC": average_precision_score(y, s), "ROC_AUC": roc_auc_score(y, s),
            "recall@5%": hit / y.sum() * 100, "precision@5%": hit / k * 100,
            "lift@5%": (hit / k) / y.mean(), "base_rate_%": y.mean() * 100}

def base_score(d):
    return 1.0 / d["qty_on_hand"].clip(lower=1).to_numpy()

# ------------------------------------------------------------------ B
section("B. TUNING ON VALIDATION (label v2, fit on train only)")
def make(name, prm, early=True):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=prm["C"], max_iter=300))
    if name == "tree":
        return DecisionTreeClassifier(random_state=SEED, **prm)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=120, max_features="sqrt", n_jobs=-1,
                                      random_state=SEED, **prm)
    return XGBClassifier(learning_rate=0.1, min_child_weight=20, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
                         n_jobs=-1, random_state=SEED,
                         **({"n_estimators": 400, "early_stopping_rounds": 30, **prm}
                            if early else prm))

def sample_idx(mask):
    idx = np.flatnonzero(mask)
    return rng.choice(idx, RF_ROWS, replace=False) if len(idx) > RF_ROWS else idx

def fit(name, prm, train_mask, y, val_mask=None, early=True):
    if name == "random_forest":
        rows = sample_idx(train_mask)
        return make(name, prm).fit(X.iloc[rows], y[rows])
    Xt, yt = X[train_mask], y[train_mask]
    m = make(name, prm, early)
    if name == "xgb" and early:
        return m.fit(Xt, yt, eval_set=[(X[val_mask], y[val_mask])], verbose=False)
    return m.fit(Xt, yt)

grids = {
    "logistic": [dict(C=0.1), dict(C=1.0)],
    "tree": [dict(max_depth=d, min_samples_leaf=l) for d in (4, 6, 8) for l in (200, 1000)],
    "random_forest": [dict(max_depth=d, min_samples_leaf=100) for d in (8, 14)],
    "xgb": [dict(max_depth=d) for d in (3, 5, 7)],
}
best, trials = {}, []
for name, grid in grids.items():
    for prm in grid:
        m = fit(name, prm, tr, Y2, va)
        s = m.predict_proba(X[va])[:, 1]
        mm = pr_metrics(Y2[va], s)
        extra = {"best_n": int(m.best_iteration) + 1} if name == "xgb" else {}
        trials.append({"model": name, **{f"p_{k}": v for k, v in prm.items()},
                       "val_PR_AUC": mm["PR_AUC"], "val_lift@5%": mm["lift@5%"]})
        if name not in best or mm["PR_AUC"] > best[name]["val"]:
            best[name] = {"val": mm["PR_AUC"], "prm": {**prm, **extra}}
    print(f"{name}: best val PR-AUC = {best[name]['val']:.4f} | params = {best[name]['prm']}")
pd.DataFrame(trials).to_csv(REP / "14_tuning_trials.csv", index=False)

# ------------------------------------------------------------------ C
section("C. RESULTS ON LABEL v2 (baseline vs models)")
rows, final, test_scores = [], {}, {}
for sname, mask in [("val", va), ("test", te)]:
    rows.append({"split": sname, "fit_on": "-", "model": "baseline: 1/stock",
                 **pr_metrics(Y2[mask], base_score(df[mask]))})

def final_params(name):
    prm = dict(best[name]["prm"])
    return prm

for name in grids:
    prm = final_params(name)
    m_tr = fit(name, prm if name != "xgb" else {k: v for k, v in prm.items() if k != "best_n"},
               tr, Y2, va)
    if name == "xgb":
        n = prm["best_n"]
        prm_fixed = {"max_depth": prm["max_depth"], "n_estimators": n}
        m_tv = fit(name, prm_fixed, tv, Y2, early=False)
    else:
        m_tv = fit(name, prm, tv, Y2)
    rows.append({"split": "val", "fit_on": "train", "model": name,
                 **pr_metrics(Y2[va], m_tr.predict_proba(X[va])[:, 1])})
    rows.append({"split": "test", "fit_on": "train", "model": name,
                 **pr_metrics(Y2[te], m_tr.predict_proba(X[te])[:, 1])})
    st = m_tv.predict_proba(X[te])[:, 1]
    rows.append({"split": "test", "fit_on": "train+val", "model": name, **pr_metrics(Y2[te], st)})
    final[name], test_scores[name] = m_tv, st
    print(f"finished {name}")

res = pd.DataFrame(rows).set_index(["split", "fit_on", "model"])
print(res.round(3).to_string())
res.round(4).to_csv(REP / "14_model_results_v2.csv")
best_name = max(best, key=lambda k: best[k]["val"])
print(f"\nModel chosen by VALIDATION PR-AUC only: {best_name}")
print("lift@5% = precision in the riskiest 5% of rows divided by the overall positive rate.")

# ------------------------------------------------------------------ D
section("D. CHOSEN MODEL: STABILITY AND CALIBRATION (test, train+val fit)")
d = df[te].copy()
d["score"] = test_scores[best_name]
d["base"] = base_score(d)
d["y2"], d["y1"] = Y2[te], Y1[te]

print("Per as-of week: positive rate and precision in the riskiest 5% of that week")
wk = []
for t, g in d.groupby("asof"):
    k = max(1, int(len(g) * 0.05))
    top_m = g.nlargest(k, "score")["y2"].mean() * 100
    top_b = g.assign(r=rng.random(len(g))).sort_values(["base", "r"], ascending=[False, True]).head(k)["y2"].mean() * 100
    wk.append({"asof": t.date(), "rows": len(g), "base_rate_%": g["y2"].mean() * 100,
               "model_prec@5%": top_m, "baseline_prec@5%": top_b})
print(pd.DataFrame(wk).round(2).to_string(index=False))

q1 = d[d["qty_on_hand"] == 1]
print(f"\nInside rows with stock == 1 ({len(q1):,} rows, the baseline cannot rank these):")
print(f"  base rate {q1['y2'].mean() * 100:.2f}% | model PR-AUC {average_precision_score(q1['y2'], q1['score']):.4f}"
      f" | model ROC-AUC {roc_auc_score(q1['y2'], q1['score']):.3f}")

d["decile"] = pd.qcut(d["score"].rank(method="first"), 10, labels=False) + 1
cal = d.groupby("decile").agg(rows=("y2", "size"), mean_predicted=("score", "mean"),
                               actual_rate=("y2", "mean"))
print("\nCalibration by predicted-risk decile (10 = riskiest):")
print((cal * [1, 100, 100]).round(2).to_string())

# ------------------------------------------------------------------ E
section("E. ROBUSTNESS: SAME MODELS, LABEL v1 (explicit zero records only)")
rows1 = [{"model": "baseline: 1/stock", **pr_metrics(Y1[te], base_score(df[te]))}]
for name in grids:
    prm = dict(best[name]["prm"])
    if name == "xgb":
        prm = {"max_depth": prm["max_depth"], "n_estimators": prm["best_n"]}
        m = fit(name, prm, tr, Y1, early=False)
    else:
        m = fit(name, prm, tr, Y1)
    rows1.append({"model": name, **pr_metrics(Y1[te], m.predict_proba(X[te])[:, 1])})
r1 = pd.DataFrame(rows1).set_index("model")
print("Test set, trained on train only (v1 positive rate is much lower, so compare lift, not PR-AUC):")
print(r1.round(3).to_string())
r1.round(4).to_csv(REP / "14_model_results_v1.csv")

# ------------------------------------------------------------------ F
section("F. XGBOOST FEATURE IMPORTANCE (fitted on train+val)")
imp = pd.Series(final["xgb"].feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(12).round(3).to_string())

pp, rr, _ = precision_recall_curve(d["y2"], d["score"])
pb, rb, _ = precision_recall_curve(d["y2"], d["base"])
plt.figure(figsize=(7, 5))
plt.plot(rr, pp, label=best_name)
plt.plot(rb, pb, label="baseline 1/stock")
plt.axhline(d["y2"].mean(), color="grey", ls="--", label="base rate")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Stockout (14 days): precision-recall on test weeks")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "14_pr_curve.png", dpi=120)
plt.close()
print(f"[saved figure] {FIG / '14_pr_curve.png'}")

# ------------------------------------------------------------------ G
section("G. SAVE ARTIFACTS")
for name, m in final.items():
    joblib.dump(m, MODELS / f"stockout_{name}.joblib", compress=3)
meta = {"feature_columns": list(X.columns), "label": "y14_v2", "horizon_days": 14,
        "best_model_by_validation": best_name,
        "best_params": {k: v["prm"] for k, v in best.items()}}
with open(MODELS / "stockout_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
d[PAIR + ["asof", "qty_on_hand", "y1", "y2", "score", "base"]].to_csv(
    REP / "14_test_predictions.csv", index=False)
print("Saved models:", sorted(p.name for p in MODELS.glob("stockout_*")))
print(f"Saved: {REP / '14_test_predictions.csv'}")
