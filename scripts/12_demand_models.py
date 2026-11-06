import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
PROC, MODELS, REP = Path("data/processed"), Path("models"), Path("reports")
FIG = REP / "figures"
MODELS.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def metrics(y_true, pred):
    err = y_true - pred
    return {"MAE": err.abs().mean(), "RMSE": float(np.sqrt((err ** 2).mean())),
            "WAPE_%": err.abs().sum() / y_true.sum() * 100, "bias": err.mean()}

# ------------------------------------------------------------------ A
section("A. LOAD AND PREPARE")
df = pd.read_csv(PROC / "demand_features.csv", parse_dates=["week_start"])
num_feats = ["lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "roll_mean_4", "roll_mean_8",
             "roll_std_4", "expanding_mean", "promo_share_lag1", "return_rate_lag1",
             "avg_price_lag1", "series_age", "store_roll4", "category_roll4",
             "closed_days", "holiday_season"]
cat_feats = ["product_division", "product_category", "store"]
train_mask = df["split"] == "train"

ratio_cols = ["promo_share_lag1", "return_rate_lag1", "avg_price_lag1"]
medians = {c: float(df.loc[train_mask, c].median()) for c in ratio_cols}
for c in ["lag_2", "lag_3", "lag_4", "lag_8"]:
    df[c] = df[c].fillna(df["expanding_mean"])
df["roll_std_4"] = df["roll_std_4"].fillna(0)
for c in ratio_cols:
    df[c] = df[c].fillna(medians[c])

X_all = pd.concat([df[num_feats], pd.get_dummies(df[cat_feats], dtype=float)], axis=1)
y = df["units"]
tr, va, te = (df["split"] == s for s in ["train", "val", "test"])
tv = tr | va
Xtr, ytr, Xva, yva, Xte, yte = X_all[tr], y[tr], X_all[va], y[va], X_all[te], y[te]
Xtv, ytv = X_all[tv], y[tv]
print(f"Feature matrix: {X_all.shape[1]} columns | missing values left: {int(X_all.isna().sum().sum())}")
print(f"Rows -> train: {tr.sum():,} | val: {va.sum():,} | test: {te.sum():,}")

# ------------------------------------------------------------------ B
section("B. TUNING ON VALIDATION (fit on train only)")
def make_model(name, p):
    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **p)
    obj = "count:poisson" if name == "xgb_poisson" else "reg:squarederror"
    return XGBRegressor(objective=obj, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=10, random_state=SEED, n_jobs=-1, **p)

xgb_grid = [dict(max_depth=d, learning_rate=lr, n_estimators=n)
            for d in (3, 5) for lr in (0.03, 0.1) for n in (200, 500)]
grids = {
    "linear": [{}],
    "random_forest": [dict(max_depth=d, min_samples_leaf=l) for d in (6, 10, None) for l in (5, 20)],
    "xgb_squared": xgb_grid,
    "xgb_poisson": xgb_grid,
}

trials, best = [], {}
for name, grid in grids.items():
    for p in grid:
        m = make_model(name, p).fit(Xtr, ytr)
        pv = np.clip(m.predict(Xva), 0, None)
        mm = metrics(yva, pv)
        trials.append({"model": name, **{f"p_{k}": v for k, v in p.items()},
                       "val_MAE": mm["MAE"], "val_RMSE": mm["RMSE"]})
        if name not in best or mm["MAE"] < best[name][0]:
            best[name] = (mm["MAE"], p)
    print(f"{name}: best val MAE = {best[name][0]:.3f} | params = {best[name][1]}")
pd.DataFrame(trials).to_csv(REP / "12_tuning_trials.csv", index=False)
print(f"All {len(trials)} trials saved to {REP / '12_tuning_trials.csv'}")

# ------------------------------------------------------------------ C
section("C. FINAL RESULTS (baselines vs models)")
rows = []
for s, mask in [("val", va), ("test", te)]:
    d = df[mask]
    for label, col in [("baseline: last_week", "lag_1"),
                       ("baseline: mean_4_weeks", "roll_mean_4"),
                       ("baseline: series_mean", "expanding_mean")]:
        rows.append({"split": s, "fit_on": "-", "model": label, **metrics(d["units"], d[col])})

final, test_preds = {}, {}
for name, (_, p) in best.items():
    m_tr = make_model(name, p).fit(Xtr, ytr)
    m_tv = make_model(name, p).fit(Xtv, ytv)
    pv = np.clip(m_tr.predict(Xva), 0, None)
    pt_tr = np.clip(m_tr.predict(Xte), 0, None)
    pt_tv = np.clip(m_tv.predict(Xte), 0, None)
    rows.append({"split": "val", "fit_on": "train", "model": name, **metrics(yva, pv)})
    rows.append({"split": "test", "fit_on": "train", "model": name, **metrics(yte, pt_tr)})
    rows.append({"split": "test", "fit_on": "train+val", "model": name, **metrics(yte, pt_tv)})
    final[name], test_preds[name] = m_tv, pt_tv

res = pd.DataFrame(rows).set_index(["split", "fit_on", "model"])
print(res.round(3).to_string())
res.round(4).to_csv(REP / "12_model_results.csv")

best_name = min(best, key=lambda k: best[k][0])
print(f"\nModel chosen by VALIDATION MAE only: {best_name}")
print("bias = mean(actual - prediction): negative means over-forecasting.")

# ------------------------------------------------------------------ D
section("D. WHERE DOES THE CHOSEN MODEL WIN OR LOSE? (test, train+val fit)")
d = df[te].copy()
d["pred_best"] = test_preds[best_name]
d["pred_base"] = d["roll_mean_4"]
d["abs_best"] = (d["units"] - d["pred_best"]).abs()
d["abs_base"] = (d["units"] - d["pred_base"]).abs()

print("Mean absolute error by whether the actual week was zero:")
print(d.groupby(np.where(d["units"] == 0, "actual = 0", "actual > 0"))
      [["abs_best", "abs_base"]].mean().round(3).to_string())

tier = pd.qcut(d["expanding_mean"], 3, labels=["low volume", "mid volume", "high volume"])
t = d.groupby(tier, observed=True)[["abs_best", "abs_base"]].mean()
t.insert(0, "rows", d.groupby(tier, observed=True).size())
print("\nMean absolute error by series volume tier:")
print(t.round(3).to_string())

def agg_wape(by, col):
    s = d.groupby(by)[["units", col]].sum()
    return (s["units"] - s[col]).abs().sum() / s["units"].sum() * 100

print("\nWAPE (%) after summing forecasts to a higher level:")
print(pd.DataFrame({
    "chosen model": [agg_wape(["week_start", "product_category"], "pred_best"),
                     agg_wape(["week_start"], "pred_best")],
    "mean_4_weeks": [agg_wape(["week_start", "product_category"], "pred_base"),
                     agg_wape(["week_start"], "pred_base")]},
    index=["category-week (all stores)", "whole chain per week"]).round(1).to_string())

# ------------------------------------------------------------------ E
section("E. FEATURE IMPORTANCE (xgb_squared, fitted on train+val)")
imp = pd.Series(final["xgb_squared"].feature_importances_, index=X_all.columns)
print(imp.sort_values(ascending=False).head(12).round(3).to_string())

w = d.groupby("week_start")[["units", "pred_best", "pred_base"]].sum()
plt.figure(figsize=(11, 4))
plt.plot(w.index, w["units"], marker="o", label="actual")
plt.plot(w.index, w["pred_best"], marker="o", label=f"{best_name}")
plt.plot(w.index, w["pred_base"], marker="o", label="mean_4_weeks baseline")
plt.legend()
plt.title("Test weeks: total units across all category-store series")
plt.tight_layout()
plt.savefig(FIG / "12_test_weekly_total.png", dpi=120)
plt.close()
print(f"[saved figure] {FIG / '12_test_weekly_total.png'}")

# ------------------------------------------------------------------ F
section("F. SAVE ARTIFACTS")
for name, m in final.items():
    joblib.dump(m, MODELS / f"demand_{name}.joblib")
meta = {"feature_columns": list(X_all.columns), "train_medians": medians,
        "best_model_by_validation": best_name,
        "best_params": {k: v[1] for k, v in best.items()}}
with open(MODELS / "demand_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
d[["week_start", "product_category", "store", "units", "pred_best", "pred_base"]].to_csv(
    REP / "12_test_predictions.csv", index=False)
print("Saved models:", sorted(p.name for p in MODELS.glob("demand_*")))
print(f"Saved: {REP / '12_test_predictions.csv'}")
