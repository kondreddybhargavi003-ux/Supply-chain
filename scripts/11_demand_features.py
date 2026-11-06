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

KEYS = ["product_category", "store"]

sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"])
sales["week_start"] = sales["transaction_date"] - pd.to_timedelta(
    sales["transaction_date"].dt.weekday, unit="D")

# ------------------------------------------------------------------ A
section("A. WEEKS AND CLOSED DAYS")
all_weeks = sorted(sales["week_start"].unique())
weeks = all_weeks[1:-1]
print(f"All weeks: {len(all_weeks)} | full weeks used: {len(weeks)}")
print(f"First full week starts {pd.Timestamp(weeks[0]).date()} | "
      f"last full week starts {pd.Timestamp(weeks[-1]).date()}")

all_days = pd.date_range(sales["transaction_date"].min(), sales["transaction_date"].max())
closed = all_days.difference(sales["transaction_date"].unique())
closed_week = pd.Series(closed).dt.to_period("W").dt.start_time.value_counts()
print(f"Closed days (no sales at all): {[d.date().isoformat() for d in closed]}")

sales = sales[sales["week_start"].isin(weeks)].copy()
sales["promo_qty"] = np.where(sales["sales_type"] != "Full Price", sales["gross_qty"], 0)
sales["gross_amount"] = np.where(sales["gross_qty"] > 0, sales["sales_amount"], 0)

# ------------------------------------------------------------------ B
section("B. WEEKLY PANEL (CATEGORY x STORE), ZERO-FILLED FROM FIRST SALE")
agg = (sales.groupby(KEYS + ["week_start"])
       .agg(units=("gross_qty", "sum"), returns=("return_qty", "sum"),
            promo_units=("promo_qty", "sum"), amount=("gross_amount", "sum"))
       .reset_index())

first = (agg[agg["units"] > 0].groupby(KEYS)["week_start"].min()
         .rename("first_week").reset_index())
wk = pd.DataFrame({"week_start": weeks})
wk["week_idx"] = range(len(wk))

panel = first.merge(wk, how="cross")
panel = panel[panel["week_start"] >= panel["first_week"]]
panel = panel.merge(agg, on=KEYS + ["week_start"], how="left")
num_cols = ["units", "returns", "promo_units", "amount"]
panel[num_cols] = panel[num_cols].fillna(0)
panel = panel.sort_values(KEYS + ["week_start"]).reset_index(drop=True)

print(f"Series (category x store): {len(first):,}")
print(f"Panel rows: {len(panel):,}")
print(f"Share of series-weeks with zero demand: {(panel['units'] == 0).mean() * 100:.1f}%")
print("Weekly units per series-week:")
print(panel["units"].describe().round(2).to_string())

# ------------------------------------------------------------------ C
section("C. FEATURES (ONLY PAST INFORMATION)")
g = panel.groupby(KEYS)
for k in [1, 2, 3, 4, 8]:
    panel[f"lag_{k}"] = g["units"].shift(k)
panel["u_prev"] = g["units"].shift(1)

g = panel.groupby(KEYS)
panel["roll_mean_4"] = g["u_prev"].transform(lambda s: s.rolling(4, min_periods=1).mean())
panel["roll_mean_8"] = g["u_prev"].transform(lambda s: s.rolling(8, min_periods=1).mean())
panel["roll_std_4"] = g["u_prev"].transform(lambda s: s.rolling(4, min_periods=2).std())
panel["expanding_mean"] = g["u_prev"].transform(lambda s: s.expanding(min_periods=1).mean())

units_prev = g["units"].shift(1).replace(0, np.nan)
panel["promo_share_lag1"] = g["promo_units"].shift(1) / units_prev
panel["return_rate_lag1"] = g["returns"].shift(1) / units_prev
panel["avg_price_lag1"] = g["amount"].shift(1) / units_prev
panel["series_age"] = (panel["week_start"] - panel["first_week"]).dt.days // 7

def past_roll_long(entity, value_name):
    table = agg.pivot_table(index="week_start", columns=entity, values="units",
                            aggfunc="sum", fill_value=0)
    table = table.reindex(weeks, fill_value=0)
    rolled = table.shift(1).rolling(4, min_periods=1).mean()
    return (rolled.reset_index()
            .melt(id_vars="week_start", var_name=entity, value_name=value_name))

panel = panel.merge(past_roll_long("store", "store_roll4"), on=["week_start", "store"], how="left")
panel = panel.merge(past_roll_long("product_category", "category_roll4"),
                    on=["week_start", "product_category"], how="left")

panel["closed_days"] = panel["week_start"].map(closed_week).fillna(0).astype(int)
panel["holiday_season"] = ((panel["week_start"] >= "2025-11-24")
                           & (panel["week_start"] <= "2025-12-22")).astype(int)

div_map = sales.drop_duplicates("product_category").set_index("product_category")["product_division"]
panel["product_division"] = panel["product_category"].map(div_map)

# first row of each series is always a sale week by construction, so drop it
before = len(panel)
panel = panel[panel["series_age"] > 0].copy()
print(f"Dropped first-sale-week rows (selection bias): {before - len(panel):,}")

panel["split"] = np.where(panel["week_idx"] < 30, "train",
                  np.where(panel["week_idx"] < 38, "val", "test"))
split_info = panel.groupby("split").agg(
    rows=("units", "size"), series=("store", "size"),
    first=("week_start", "min"), last=("week_start", "max"),
    mean_units=("units", "mean"), zero_share=("units", lambda x: (x == 0).mean() * 100))
split_info["series"] = panel.groupby("split").apply(
    lambda d: d.groupby(KEYS).ngroups, include_groups=False)
print("\nTime-based split:")
print(split_info.round(2).to_string())

feature_cols = ["lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "roll_mean_4", "roll_mean_8",
                "roll_std_4", "expanding_mean", "promo_share_lag1", "return_rate_lag1",
                "avg_price_lag1", "series_age", "store_roll4", "category_roll4",
                "closed_days", "holiday_season"]
print("\nMissing values in features:")
print(panel[feature_cols].isna().sum().to_string())

out_cols = (["week_start", "week_idx", "split", "product_division"] + KEYS
            + feature_cols + ["units"])
panel = panel[out_cols].drop(columns=[])
panel.to_csv(PROC / "demand_features.csv", index=False)
print(f"\nSaved: {PROC / 'demand_features.csv'}  ({len(panel):,} rows)")

# ------------------------------------------------------------------ D
section("D. BASELINES (THE NUMBERS A MODEL MUST BEAT)")
def metrics(df, pred):
    err = df["units"] - pred
    return pd.Series({
        "rows": len(df),
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "WAPE_%": err.abs().sum() / df["units"].sum() * 100,
        "bias": err.mean()})

rows = {}
for s in ["val", "test"]:
    d = panel[panel["split"] == s]
    for name, pred in [("zero", 0 * d["units"]),
                       ("last_week (lag_1)", d["lag_1"]),
                       ("mean_4_weeks", d["roll_mean_4"]),
                       ("mean_8_weeks", d["roll_mean_8"]),
                       ("series_mean", d["expanding_mean"])]:
        rows[(s, name)] = metrics(d, pred)
res = pd.DataFrame(rows).T
res.index.names = ["split", "baseline"]
print(res.round(3).to_string())
print("\nLower MAE, RMSE and WAPE are better. 'bias' near 0 means no systematic over or under forecast.")
