from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC = Path("data/processed")
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=120)
    plt.close()
    print(f"[saved figure] {FIG / name}")

sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"])
inv = pd.read_csv(PROC / "inventory_intervals_clean.csv",
                  parse_dates=["start_date", "end_date"])
sales["week"] = sales["transaction_date"].dt.to_period("W")
weeks = sorted(sales["week"].unique())
inner = weeks[1:-1]
full = sales[sales["week"].isin(inner)]

# ------------------------------------------------------------------ A
section("A. WEEKLY GROSS DEMAND")
wk = full.groupby("week")["gross_qty"].sum()
wk.index = wk.index.start_time
print(wk.describe().round(0).to_string())
print("\nTop 5 weeks:\n", wk.sort_values(ascending=False).head(5).to_string())
print("\nBottom 5 weeks:\n", wk.sort_values().head(5).to_string())
plt.figure(figsize=(11, 4))
plt.plot(wk.index, wk.values, marker="o", ms=3)
plt.title("Weekly gross units sold (all stores)")
plt.ylabel("Units")
save("A_weekly_total.png")

div = full.groupby(["week", "product_division"])["gross_qty"].sum().unstack()
div.index = div.index.start_time
plt.figure(figsize=(11, 4))
for c in div.columns:
    plt.plot(div.index, div[c], label=c)
plt.legend()
plt.title("Weekly gross units by division")
save("A_weekly_by_division.png")
print("\nUnits share by division (%):")
print((full.groupby("product_division")["gross_qty"].sum()
       / full["gross_qty"].sum() * 100).round(1).to_string())

# ------------------------------------------------------------------ B
section("B. WEEKDAY PATTERN")
d = sales.groupby("transaction_date")["gross_qty"].sum().reset_index()
d["dow"] = d["transaction_date"].dt.day_name()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = d.groupby("dow")["gross_qty"].mean().reindex(order)
print("Average units per day by weekday:")
print(dow.round(0).to_string())
dow.plot(kind="bar", figsize=(8, 4), title="Average daily units by weekday")
save("B_weekday.png")

# ------------------------------------------------------------------ C
section("C. CATEGORIES AND STORES")
cat = full.groupby("product_category")["gross_qty"].sum().sort_values(ascending=False)
print("Top 10 categories (units):")
print(cat.head(10).to_string())
cat.head(10).sort_values().plot(kind="barh", figsize=(8, 5), title="Top 10 categories by units")
save("C_top_categories.png")

st = full.groupby("store")["gross_qty"].sum().sort_values(ascending=False)
print(f"\nStores: {len(st)} | top store {st.iloc[0]:,.0f} units | "
      f"bottom store {st.iloc[-1]:,.0f} units")
print(f"Top 5 stores share of units: {st.head(5).sum() / st.sum() * 100:.1f}%")
st.plot(kind="bar", figsize=(12, 4), title="Units by store")
save("C_units_by_store.png")

# ------------------------------------------------------------------ D
section("D. SALES TYPE AND PRICE")
mix = full.groupby(["week", "sales_type"])["gross_qty"].sum().unstack().fillna(0)
share = mix.div(mix.sum(axis=1), axis=0) * 100
share.index = share.index.start_time
print("Average weekly share of units by Sales Type (%):")
print(share.mean().round(1).to_string())
share.plot(figsize=(11, 4), title="Weekly share of units by Sales Type (%)")
save("D_sales_type_share.png")

pos = sales[sales["gross_qty"] > 0].copy()
pos["unit_price_sold"] = pos["sales_amount"] / pos["gross_qty"]
print("\nAverage selling price per unit by Sales Type:")
print(pos.groupby("sales_type")["unit_price_sold"].agg(["mean", "median"]).round(1).to_string())

# ------------------------------------------------------------------ E
section("E. RETURNS")
ret = full.groupby("week")[["gross_qty", "return_qty"]].sum()
ret["rate"] = ret["return_qty"] / ret["gross_qty"] * 100
print(f"Weekly return rate (%): mean={ret['rate'].mean():.1f} "
      f"min={ret['rate'].min():.1f} max={ret['rate'].max():.1f}")
rc = (full.groupby("product_category")[["gross_qty", "return_qty"]].sum()
      .assign(rate=lambda x: x.return_qty / x.gross_qty * 100)
      .query("gross_qty >= 500").sort_values("rate", ascending=False))
print("\nHighest return rates by category (categories with 500+ units):")
print(rc.head(5).round(1).to_string())
print("Lowest:")
print(rc.tail(3).round(1).to_string())

# ------------------------------------------------------------------ F
section("F. DAILY STOCKOUT RATE (rebuilt from intervals)")
cutoff = sales["transaction_date"].max()
days = pd.date_range(inv["start_date"].min(), cutoff)
def daily_count(frame):
    s = (frame["start_date"] - days[0]).dt.days.to_numpy()
    e = (frame["end_date"] - days[0]).dt.days.to_numpy() + 1
    arr = np.zeros(len(days) + 1)
    np.add.at(arr, s, 1)
    np.add.at(arr, e, -1)
    return np.cumsum(arr)[:-1]
tracked = daily_count(inv)
oos = daily_count(inv[inv["out_of_stock"] == 1])
rate = pd.Series(oos / tracked * 100, index=days)
print(f"Pairs tracked per day: min={tracked.min():,.0f} max={tracked.max():,.0f}")
print(f"Share of tracked pairs with stock <= 0: mean={rate.mean():.2f}% "
      f"min={rate.min():.2f}% max={rate.max():.2f}%")
print("Monthly average out-of-stock share (%):")
print(rate.resample("MS").mean().round(2).to_string())
rate.plot(figsize=(11, 4), title="Daily share of tracked product-store pairs with stock <= 0 (%)")
save("F_daily_stockout_rate.png")

inv["oos_days"] = inv["days_valid"] * inv["out_of_stock"]
for col, n in [("product_category", 27), ("supplier", 10), ("store", 10)]:
    g = inv.groupby(col)[["oos_days", "days_valid"]].sum()
    g["oos_pct"] = g["oos_days"] / g["days_valid"] * 100
    g = g.sort_values("oos_pct", ascending=False)
    print(f"\nHighest day-weighted stockout % by {col} (top {min(n, 8)}):")
    print(g["oos_pct"].head(8).round(2).to_string())
    print(f"Lowest 3:\n{g['oos_pct'].tail(3).round(2).to_string()}")

# ------------------------------------------------------------------ G
section("G. DO SALES HAPPEN WHILE INVENTORY SAYS STOCK <= 0?")
left = (sales[sales["gross_qty"] > 0][["transaction_date", "product_no", "store"]]
        .sort_values("transaction_date"))
right = (inv[["start_date", "end_date", "product_no", "store", "qty_on_hand"]]
         .sort_values("start_date"))
m = pd.merge_asof(left, right, left_on="transaction_date", right_on="start_date",
                  by=["product_no", "store"], direction="backward")
m["covered"] = m["qty_on_hand"].notna() & (m["transaction_date"] <= m["end_date"])
print(f"Sales rows: {len(m):,}")
print(f"Sales matched to an inventory interval covering that day: "
      f"{m['covered'].mean() * 100:.1f}%")
cov = m[m["covered"]]
print(f"Of matched sales, stock on hand <= 0 that day: {(cov['qty_on_hand'] <= 0).mean() * 100:.2f}%")
print(f"Of matched sales, stock on hand == 1 that day: {(cov['qty_on_hand'] == 1).mean() * 100:.2f}%")
print("Stock on hand on the day of a sale:")
print(cov["qty_on_hand"].describe().round(2).to_string())
