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

def ns(s):
    return pd.to_datetime(s).astype("datetime64[ns]")

PAIR = ["product_no", "store"]
HORIZONS = [7, 14, 28]

sales = pd.read_csv(PROC / "sales_clean.csv", parse_dates=["transaction_date"])
inv = pd.read_csv(PROC / "inventory_intervals_clean.csv",
                  parse_dates=["start_date", "end_date"])
sales = sales[sales["gross_qty"] > 0][["transaction_date"] + PAIR + ["gross_qty"]].copy()
sales["transaction_date"] = ns(sales["transaction_date"])
inv["start_date"] = ns(inv["start_date"])
inv["end_date"] = ns(inv["end_date"])
inv = inv.sort_values(PAIR + ["start_date"]).reset_index(drop=True)
cutoff = inv["end_date"].max()
g = inv.groupby(PAIR)
inv["next_start"] = g["start_date"].shift(-1)
inv["next_qty"] = g["qty_on_hand"].shift(-1)
inv["prev_qty"] = g["qty_on_hand"].shift(1)
print(f"Data cutoff: {cutoff.date()} | inventory records: {len(inv):,} | sale rows: {len(sales):,}")

# ------------------------------------------------------------------ A
section("A. IMPLIED STOCKOUTS: LAST UNIT SOLD, BUT NO ZERO RECORD WRITTEN")
explicit = inv[inv["out_of_stock"] == 1]
print(f"Explicit zero/negative records: {len(explicit):,}")
print(f"  with stock > 0 in the previous record (real in-stock -> zero step): "
      f"{(explicit['prev_qty'] > 0).sum():,}")

right = inv[PAIR + ["start_date", "end_date", "qty_on_hand", "next_start", "next_qty"]].rename(
    columns={"start_date": "rec_start", "end_date": "rec_end", "qty_on_hand": "rec_qty"})
left = sales.sort_values("transaction_date")
m = pd.merge_asof(left, right.sort_values("rec_start"), left_on="transaction_date",
                  right_on="rec_start", by=PAIR)
unc = m[m["rec_start"].notna() & (m["transaction_date"] > m["rec_end"])
        & (m["rec_qty"] > 0)].copy()
print(f"Sales after an ended in-stock record with no covering record: {len(unc):,}")

unc = unc.sort_values("transaction_date")
unc["cum_units"] = unc.groupby(PAIR + ["rec_start"])["gross_qty"].cumsum()
hit = unc[unc["cum_units"] >= unc["rec_qty"]]
implied = (hit.groupby(PAIR + ["rec_start"])
           .agg(next_imp=("transaction_date", "min"), rec_qty=("rec_qty", "first"),
                rec_end=("rec_end", "first"), next_start=("next_start", "first"),
                next_qty=("next_qty", "first"))
           .reset_index())

nofollow = ((inv["qty_on_hand"] > 0) & (inv["end_is_open"] == 0)
            & (inv["next_start"].isna()
               | (inv["next_start"] > inv["end_date"] + pd.Timedelta(days=1))))
print(f"\nIn-stock records that ended with no record starting the next day: {nofollow.sum():,}")
print(f"Implied stockout events (later sales reach the record's stock): {len(implied):,} "
      f"({len(implied) / nofollow.sum() * 100:.1f}% of those)")
print(f"Total events: explicit {len(explicit):,} + implied {len(implied):,}")

lab = ["1", "2", "3", "4+"]
b = pd.cut(implied["rec_qty"], bins=[0, 1, 2, 3, 100000], labels=lab)
print("\nStock of the record that was emptied (%):")
print((b.value_counts(normalize=True).reindex(lab) * 100).round(1).to_string())

delay = (implied["next_imp"] - implied["rec_end"]).dt.days
print("\nDays from the record's end to the sale that empties it:")
print(delay.describe().round(1).to_string())

re = implied["next_start"].notna()
print(f"\nPair gets a later record after the implied stockout: {re.mean() * 100:.1f}% "
      f"| never gets one: {(~re).mean() * 100:.1f}%")
rq = implied.loc[re, "next_qty"]
print("Stock in that next record (a jump up suggests a restock):")
print(rq.describe().round(2).to_string())
print(f"  next record is zero/negative: {(rq <= 0).mean() * 100:.1f}% | stock >= 3: {(rq >= 3).mean() * 100:.1f}%")
print("-> A restock after the event supports 'real stockout'. No later record leaves delisting possible.")

# ------------------------------------------------------------------ B
section("B. BUILD PAIR-WEEKS AND LABELS (as of each Sunday, in stock > 0)")
asof_dates = pd.date_range("2025-06-08", cutoff, freq="W-SUN")
instock = inv[inv["out_of_stock"] == 0]
parts = []
for t in asof_dates:
    s = instock[(instock["start_date"] <= t) & (instock["end_date"] >= t)]
    s = s[PAIR + ["qty_on_hand"]].copy()
    s["asof"] = t
    parts.append(s)
state = pd.concat(parts, ignore_index=True)
state["asof"] = ns(state["asof"])
state = state.sort_values("asof").reset_index(drop=True)
print(f"As-of dates: {len(asof_dates)} | in-stock pair-weeks: {len(state):,}")

exp = (explicit[PAIR + ["start_date"]].rename(columns={"start_date": "next_exp"})
       .sort_values("next_exp"))
imp = implied[PAIR + ["next_imp"]].sort_values("next_imp")
state = pd.merge_asof(state, exp, left_on="asof", right_on="next_exp",
                      by=PAIR, direction="forward", allow_exact_matches=False)
state = pd.merge_asof(state, imp, left_on="asof", right_on="next_imp",
                      by=PAIR, direction="forward", allow_exact_matches=False)
state["next_any"] = state[["next_exp", "next_imp"]].min(axis=1)

for H in HORIZONS:
    lim = state["asof"] + pd.Timedelta(days=H)
    state[f"obs{H}"] = (lim <= cutoff).astype(int)
    state[f"y{H}_v1"] = (state["next_exp"] <= lim).astype(int)
    state[f"y{H}_v2"] = (state["next_any"] <= lim).astype(int)

rows = []
for H in HORIZONS:
    ok = state[state[f"obs{H}"] == 1]
    p1, p2 = int(ok[f"y{H}_v1"].sum()), int(ok[f"y{H}_v2"].sum())
    rows.append({"horizon_days": H, "pair_weeks": len(ok),
                 "v1_positives": p1, "v1_rate_%": p1 / len(ok) * 100,
                 "v2_positives": p2, "v2_rate_%": p2 / len(ok) * 100,
                 "implied_only_share_of_v2_%": (p2 - p1) / p2 * 100})
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("Reference: script 10 gave v1 rates of 0.91%, 1.76% and 3.29% for 7, 14 and 28 days.")

# ------------------------------------------------------------------ C
section("C. HORIZON 14: WHAT DOES EACH LABEL LOOK LIKE?")
ok = state[state["obs14"] == 1].copy()
ok["q"] = pd.cut(ok["qty_on_hand"], bins=[0, 1, 2, 3, 100000], labels=lab)
t = ok.groupby("q", observed=True).agg(
    pair_weeks=("y14_v1", "size"),
    v1_rate=("y14_v1", lambda x: x.mean() * 100),
    v2_rate=("y14_v2", lambda x: x.mean() * 100))
print("Positive rate (%) by stock on hand at week end:")
print(t.round(2).to_string())

print("\nSimple rule 'stock on hand == 1 means high risk' (the baseline to beat):")
for v in ["v1", "v2"]:
    y = ok[f"y14_{v}"]
    flag = ok["qty_on_hand"] == 1
    prec = y[flag].mean()
    rec = flag[y == 1].mean()
    print(f"  {v}: flags {flag.mean() * 100:.1f}% of rows | precision {prec * 100:.2f}% "
          f"| recall {rec * 100:.1f}% | lift {prec / y.mean():.2f}x")

mo = ok.assign(month=ok["asof"].dt.to_period("M")).groupby("month").agg(
    pair_weeks=("y14_v1", "size"),
    v1_rate=("y14_v1", lambda x: x.mean() * 100),
    v2_rate=("y14_v2", lambda x: x.mean() * 100))
print("\nPositive rate (%) by month of the as-of date:")
print(mo.round(2).to_string())

# ------------------------------------------------------------------ D
section("D. TIME SPLIT FOR HORIZON 14 (label windows never cross a boundary)")
VAL_START = pd.Timestamp("2026-01-04")
TEST_START = pd.Timestamp("2026-03-01")
end14 = state["asof"] + pd.Timedelta(days=14)
state["split14"] = np.select(
    [end14 > cutoff,
     end14 < VAL_START,
     (state["asof"] >= VAL_START) & (end14 < TEST_START),
     state["asof"] >= TEST_START],
    ["unobserved", "train", "val", "test"], default="purged")
sp = state.groupby("split14").agg(
    pair_weeks=("y14_v1", "size"), asof_dates=("asof", "nunique"),
    first_asof=("asof", "min"), last_asof=("asof", "max"),
    v1_pos=("y14_v1", "sum"), v2_pos=("y14_v2", "sum"))
sp["v1_rate_%"] = sp["v1_pos"] / sp["pair_weeks"] * 100
sp["v2_rate_%"] = sp["v2_pos"] / sp["pair_weeks"] * 100
sp.loc[sp.index == "unobserved", ["v1_pos", "v2_pos", "v1_rate_%", "v2_rate_%"]] = np.nan
print(sp.round(2).to_string())
print("'purged' rows are skipped so label windows cannot overlap the next period.")
print("'unobserved' rows have a window that runs past the data and are not used.")

out_cols = (PAIR + ["asof", "qty_on_hand", "split14"]
            + [f"obs{H}" for H in HORIZONS]
            + [f"y{H}_{v}" for H in HORIZONS for v in ("v1", "v2")])
state[out_cols].to_csv(PROC / "stockout_labels.csv.gz", index=False, compression="gzip")
print(f"\nSaved: {PROC / 'stockout_labels.csv.gz'}  ({len(state):,} rows)")
