import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PROC, MODELS, REP = Path("data/processed"), Path("models"), Path("reports")
PAIR = ["product_no", "store"]
LIVE_ASOF = pd.Timestamp("2026-04-19")
CHECK_ASOF = pd.Timestamp("2026-04-05")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def ns(s):
    return pd.to_datetime(s).astype("datetime64[ns]")

# ------------------------------------------------------------------ A
section("A. LOAD DATA, MODEL AND FEATURE LIST")
lab = pd.read_csv(PROC / "stockout_labels.csv.gz", usecols=PAIR + ["asof", "qty_on_hand"],
                  parse_dates=["asof"])
lab["asof"] = ns(lab["asof"])
lab = lab[lab["asof"].isin([LIVE_ASOF, CHECK_ASOF])].sort_values("asof").reset_index(drop=True)
print(f"Rows to score: {lab.groupby('asof').size().to_dict()}")

inv = pd.read_csv(PROC / "inventory_intervals_clean.csv", parse_dates=["start_date", "end_date"])
inv["start_date"] = ns(inv["start_date"])
inv = inv.sort_values(PAIR + ["start_date"]).reset_index(drop=True)
inv["prev_qty"] = inv.groupby(PAIR)["qty_on_hand"].shift()

sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"])
sales = sales[sales["gross_qty"] > 0][["transaction_date"] + PAIR + ["gross_qty"]].copy()
sales["transaction_date"] = ns(sales["transaction_date"])

with open(MODELS / "stockout_meta.json") as fh:
    meta = json.load(fh)
model = joblib.load(MODELS / "stockout_xgb.joblib")
print(f"Model feature columns: {len(meta['feature_columns'])} | label: {meta['label']} "
      f"| horizon: {meta['horizon_days']} days")

# ------------------------------------------------------------------ B
section("B. BUILD THE SAME FEATURES AS SCRIPT 14 (only information up to the as-of Sunday)")
rec = (inv[PAIR + ["start_date", "qty_on_hand", "prev_qty", "unit_price", "unit_cost",
                   "stock_status", "product_division", "product_category"]]
       .rename(columns={"qty_on_hand": "rec_qty"}).sort_values("start_date"))
df = pd.merge_asof(lab, rec, left_on="asof", right_on="start_date", by=PAIR)
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

num = ["qty_on_hand", "days_in_record", "pair_age_days", "has_prev", "prev_qty_f", "qty_change",
       "unit_price", "unit_cost", "margin_pct", "units_1w", "units_4w", "units_8w",
       "active_weeks_8", "cover_weeks"]
cat = ["product_division", "product_category", "store", "stock_status"]
X = pd.concat([df[num].astype(np.float32), pd.get_dummies(df[cat], dtype=np.float32)], axis=1)
extra = [col for col in X.columns if col not in meta["feature_columns"]]
missing = [col for col in meta["feature_columns"] if col not in X.columns]
print(f"Feature columns not known to the model (dropped): {len(extra)} | "
      f"model columns absent here (filled with 0): {len(missing)}")
X = X.reindex(columns=meta["feature_columns"], fill_value=0.0)
print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns | missing values: {int(X.isna().sum().sum())}")
df["score"] = model.predict_proba(X)[:, 1]

# ------------------------------------------------------------------ C
section("C. CHECK: DOES THIS CODE REPRODUCE THE SAVED TEST SCORES?")
saved = pd.read_csv(REP / "14_test_predictions.csv", parse_dates=["asof"])
saved["asof"] = ns(saved["asof"])
chk = df[df["asof"] == CHECK_ASOF][PAIR + ["asof", "score"]].merge(
    saved[PAIR + ["asof", "score"]], on=PAIR + ["asof"], suffixes=("_new", "_saved"))
print(f"Rows compared for {CHECK_ASOF.date()}: {len(chk):,}")
if len(chk):
    diff = (chk["score_new"] - chk["score_saved"]).abs()
    print(f"Max absolute difference: {diff.max():.6f} | mean: {diff.mean():.6f} | "
          f"correlation: {chk['score_new'].corr(chk['score_saved']):.6f}")
    print("-> A max difference near 0 means the live scoring matches the validated pipeline.")

# ------------------------------------------------------------------ D
section(f"D. LATEST SCORES (as of {LIVE_ASOF.date()}): RISK TIERS BY RANK")
live = df[df["asof"] == LIVE_ASOF].copy()
live["rank_pct"] = live["score"].rank(method="first", ascending=False) / len(live)
live["risk_tier"] = np.select([live["rank_pct"] <= 0.05, live["rank_pct"] <= 0.20],
                              ["High", "Medium"], default="Low")
print(f"In-stock pairs scored: {len(live):,}")
print("Score distribution (not literal probabilities; the training rate was 5.9%, the test rate 8.1%):")
print(live["score"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(4).to_string())
tt = live["risk_tier"].value_counts().reindex(["High", "Medium", "Low"])
print("\nTier sizes:")
print(pd.DataFrame({"pairs": tt, "share_%": (tt / len(live) * 100).round(1)}).to_string())
print("\nTier by stock on hand:")
live["stock_band"] = pd.cut(live["qty_on_hand"], [0, 1, 2, 3, 10000], labels=["1", "2", "3", "4+"])
print(pd.crosstab(live["stock_band"], live["risk_tier"]).reindex(columns=["High", "Medium", "Low"]).to_string())
print("\nHigh tier: top 5 categories and top 5 stores")
hi = live[live["risk_tier"] == "High"]
print(hi["product_category"].value_counts().head(5).to_string())
print(hi["store"].value_counts().head(5).to_string())

rr = PROC / "reorder_recommendations.csv"
if rr.exists():
    rec_list = pd.read_csv(rr, usecols=PAIR + ["status", "priority_rank"])
    j = live.merge(rec_list, on=PAIR, how="left")
    print("\nML risk tier vs reorder-list status (stock dates differ by 5 days: 19 vs 24 April):")
    print(pd.crosstab(j["status"].fillna("not tracked on 24 Apr"), j["risk_tier"])
          .reindex(columns=["High", "Medium", "Low"]).to_string())

out = live[PAIR + ["product_division", "product_category", "asof", "qty_on_hand", "units_4w",
                   "units_8w", "cover_weeks", "score", "rank_pct", "risk_tier"]]
out.round({"score": 5, "rank_pct": 5, "cover_weeks": 3}).to_csv(PROC / "stockout_risk_latest.csv", index=False)
print(f"\nSaved: {PROC / 'stockout_risk_latest.csv'} ({len(out):,} rows)")
