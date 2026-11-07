from pathlib import Path
import numpy as np
import pandas as pd

RAW, PROC, REP = Path("data/raw"), Path("data/processed"), Path("reports")
PROC.mkdir(parents=True, exist_ok=True)
REP.mkdir(exist_ok=True)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

names = ["buyers", "goods_receipts", "items", "po_lines", "purchase_orders",
         "requisitions", "suppliers", "supplier_invoices"]
T = {n: pd.read_csv(RAW / f"{n}.csv") for n in names}
buyers, gr, items, pl, po, req, sup, inv = (T[n] for n in names)
gr["received_on"] = pd.to_datetime(gr["received_on"])
pl["ordered_on"] = pd.to_datetime(pl["ordered_on"])
po["ordered_on"] = pd.to_datetime(po["ordered_on"])
req["requested_on"] = pd.to_datetime(req["requested_on"])
inv["invoice_date"] = pd.to_datetime(inv["invoice_date"])
inv["paid_on"] = pd.to_datetime(inv["paid_on"])

lines = (pl.rename(columns={"id": "po_line_id", "ordered_on": "line_ordered_on"})
         .merge(po.rename(columns={"id": "purchase_order_id", "ordered_on": "po_ordered_on",
                                   "payment_terms": "po_payment_terms"}),
                on="purchase_order_id", how="left")
         .merge(sup.rename(columns={"id": "supplier_id", "payment_terms": "supplier_terms"}),
                on="supplier_id", how="left")
         .merge(items.rename(columns={"id": "item_id", "list_price": "item_list_price"}),
                on="item_id", how="left")
         .merge(buyers[["id", "team"]].rename(columns={"id": "buyer_id", "team": "buyer_team"}),
                on="buyer_id", how="left"))
lines["line_amount"] = lines["quantity"] * lines["unit_price"]

gr = gr.sort_values(["po_line_id", "received_on", "id"]).reset_index(drop=True)
gr["receipt_order"] = gr.groupby("po_line_id").cumcount() + 1
gr["cum_received_qty"] = gr.groupby("po_line_id")["received_qty"].cumsum()
rc = gr.groupby("po_line_id").agg(
    n_receipts=("id", "size"), received_qty_total=("received_qty", "sum"),
    first_received=("received_on", "min"), last_received=("received_on", "max"),
    n_not_good=("condition", lambda s: int((s != "good").sum())))
lines = lines.merge(rc, on="po_line_id", how="left")
lines["n_receipts"] = lines["n_receipts"].fillna(0).astype(int)
lines["n_not_good"] = lines["n_not_good"].fillna(0).astype(int)
lines["received_qty_total"] = lines["received_qty_total"].fillna(0)
lines["received_ratio"] = lines["received_qty_total"] / lines["quantity"]
lines["lead_days_first"] = (lines["first_received"] - lines["po_ordered_on"]).dt.days

# ------------------------------------------------------------------ A
section("A. DO RECEIPTS ADD UP? (cumulative received quantity vs ordered quantity per PO line)")
print(f"PO lines: {len(lines)} | with receipts: {(lines['n_receipts'] > 0).sum()} | "
      f"with 1 receipt: {(lines['n_receipts'] == 1).sum()} | with 2 receipts: {(lines['n_receipts'] == 2).sum()}")
print("\nreceived_share by receipt order (share of the ordered quantity):")
print(gr.groupby("receipt_order")["received_share"].agg(["count", "mean", "min", "max"]).round(3).to_string())
r = lines[lines["n_receipts"] > 0].copy()
print("\nCumulative received quantity / ordered quantity, by number of receipts:")
print(r.groupby("n_receipts")["received_ratio"].describe().round(3).to_string())
cls = pd.cut(r["received_ratio"], [-0.01, 0.95, 1.05, 100], labels=["short (<95%)", "complete (95-105%)", "over (>105%)"])
print("\nPO lines by cumulative receipt result:")
print(pd.DataFrame({"lines": cls.value_counts().reindex(["short (<95%)", "complete (95-105%)", "over (>105%)"])}).to_string())
print(f"Units received beyond the ordered quantity: "
      f"{(r['received_qty_total'] - r['quantity']).clip(lower=0).sum():,.0f} of {r['quantity'].sum():,.0f} ordered on those lines")
print("-> Two receipts that each look like most of the order would be over-delivery, duplicate records, or shares measured differently.")

# ------------------------------------------------------------------ B
section("B. LEAD TIME WITHOUT THE LATE-DATA BIAS (only lines old enough to have arrived)")
horizon = gr["received_on"].max()
max_lead = lines["lead_days_first"].max()
lines["age_at_horizon"] = (horizon - lines["po_ordered_on"]).dt.days
print(f"Latest receipt date in the data: {horizon.date()} | longest observed first-receipt lead time: {max_lead:.0f} days")
nr = lines[lines["n_receipts"] == 0]
print(f"Lines with no receipt: {len(nr)} | older than {max_lead:.0f} days at the horizon "
      f"(past anything observed): {(nr['age_at_horizon'] > max_lead).sum()} | "
      f"possibly still pending: {(nr['age_at_horizon'] <= max_lead).sum()}")
mature = lines[lines["age_at_horizon"] >= max_lead]
print(f"\nMature cohort (ordered at least {max_lead:.0f} days before the horizon): {len(mature)} lines")
rows = []
for d in [7, 14, 21, 30, 40]:
    rows.append({"received_within_days": d,
                 "share_of_lines_%": ((mature["lead_days_first"] <= d)).mean() * 100})
print(pd.DataFrame(rows).round(1).to_string(index=False))
print(f"Never received in the mature cohort: {(mature['n_receipts'] == 0).mean() * 100:.1f}%")
print("-> The measured lead time (median 11 days) only covers lines that arrived. This table shows how many never did.")

# ------------------------------------------------------------------ C
section("C. INVOICE CHECKS (three-way match: order, receipt, invoice)")
inv_tot = inv.groupby("po_line_id").agg(invoices_on_line=("id", "size"),
                                         invoiced_qty_total=("invoiced_qty", "sum")).reset_index()
ii = (inv.rename(columns={"id": "invoice_id"})
      .merge(lines[["po_line_id", "po_number", "supplier_name", "item_label", "quantity",
                    "unit_price", "po_payment_terms", "n_receipts", "received_qty_total",
                    "first_received"]], on="po_line_id", how="left")
      .merge(inv_tot, on="po_line_id", how="left"))
ii["billed_amount"] = ii["invoiced_qty"] * ii["billed_unit_price"]
ii["is_paid"] = ii["paid_on"].notna()
ii["no_receipt"] = ii["n_receipts"] == 0
ii["before_receipt"] = ii["first_received"].notna() & (ii["invoice_date"] < ii["first_received"])
ii["price_over_2pct"] = ii["billed_unit_price"] / ii["unit_price"] > 1.02
ii["line_billed_over_ordered"] = ii["invoiced_qty_total"] > ii["quantity"] * 1.05
ii["line_billed_over_received"] = (ii["n_receipts"] > 0) & (ii["invoiced_qty_total"] > ii["received_qty_total"] * 1.05)
flags = ["no_receipt", "before_receipt", "price_over_2pct", "line_billed_over_ordered", "line_billed_over_received"]
ii["flag_count"] = ii[flags].sum(axis=1)
print(f"Invoices: {len(ii)} | on lines with 2 invoices: {(ii['invoices_on_line'] == 2).sum()}")
print("\nInvoiced quantity on a line / ordered quantity (lines with at least one invoice):")
ln = ii.drop_duplicates("po_line_id")
print((ln["invoiced_qty_total"] / ln["quantity"]).describe().round(3).to_string())
dbl = ln[ln["line_billed_over_ordered"]]
excess_qty = (dbl["invoiced_qty_total"] - dbl["quantity"]).clip(lower=0)
excess_amt = (excess_qty * dbl["unit_price"]).sum()
print(f"Lines billed for more than 105% of the ordered quantity: {len(dbl)} | "
      f"excess units: {excess_qty.sum():,.0f} | excess at PO price: {excess_amt:,.0f} "
      f"(of {ii['billed_amount'].sum():,.0f} billed)")
print("\nFlag counts by invoice status:")
t = ii.groupby("status")[flags].sum()
t.insert(0, "invoices", ii.groupby("status").size())
print(t.to_string())
print("\nShare of invoices with at least one flag, by status (%):")
print((ii.groupby("status")["flag_count"].apply(lambda s: (s > 0).mean() * 100)).round(1).to_string())

# ------------------------------------------------------------------ D
section("D. BUILD CLEAN TABLES FOR THE APP")
pay_horizon = max(inv["paid_on"].max(), inv["invoice_date"].max())
ii["days_unpaid_at_horizon"] = np.where(ii["is_paid"], np.nan, (pay_horizon - ii["invoice_date"]).dt.days)
ii["terms_days"] = ii["po_payment_terms"].str.extract(r"(\d+)")[0].astype(float)
ii["days_to_pay"] = (ii["paid_on"] - ii["invoice_date"]).dt.days
ii["paid_late"] = ii["is_paid"] & (ii["days_to_pay"] > ii["terms_days"])
print(f"Payment horizon used for unpaid ages: {pay_horizon.date()} (latest payment/invoice date in the data; indicative)")

lagg = ii.groupby("po_line_id").agg(invoiced_amount=("billed_amount", "sum"),
                                    paid_amount=("billed_amount", lambda s: s[ii.loc[s.index, "is_paid"]].sum()),
                                    n_invoices=("invoice_id", "size"))
lines = lines.merge(lagg, on="po_line_id", how="left")
for c in ["invoiced_amount", "paid_amount", "n_invoices"]:
    lines[c] = lines[c].fillna(0)
lines["unpaid_amount"] = lines["invoiced_amount"] - lines["paid_amount"]
lines["receipt_status"] = np.select(
    [lines["n_receipts"] == 0, lines["received_ratio"] < 0.95, lines["received_ratio"] <= 1.05],
    ["not received", "short", "complete"], default="over")

sc = lines.groupby("supplier_name").agg(
    pos=("purchase_order_id", "nunique"), lines=("po_line_id", "size"), spend=("line_amount", "sum"),
    lines_received=("n_receipts", lambda s: int((s > 0).sum())),
    median_lead_days=("lead_days_first", "median"),
    receipts_not_good=("n_not_good", "sum"), n_receipts=("n_receipts", "sum"))
iv_s = ii.groupby("supplier_name").agg(
    invoices=("invoice_id", "size"), disputed=("status", lambda s: int((s == "disputed").sum())),
    unpaid=("is_paid", lambda s: int((~s).sum())), billed=("billed_amount", "sum"),
    flagged=("flag_count", lambda s: int((s > 0).sum())))
sc = sc.join(iv_s).fillna({"invoices": 0, "disputed": 0, "unpaid": 0, "billed": 0, "flagged": 0})
sc["spend_share_%"] = sc["spend"] / sc["spend"].sum() * 100
sc["sample_note"] = np.where((sc["n_receipts"] < 10) | (sc["invoices"] < 10), "small sample: read counts only", "")
sc = sc.sort_values("spend", ascending=False)
print("\nSupplier table (counts, not rates):")
print(sc[["pos", "lines", "spend", "spend_share_%", "n_receipts", "receipts_not_good", "median_lead_days",
          "invoices", "disputed", "unpaid", "flagged", "sample_note"]].round(1).to_string())

rq = req.rename(columns={"id": "requisition_id"})
conv = pl[["requisition_id", "ordered_on"]].rename(columns={"ordered_on": "po_line_ordered_on"})
rq = rq.merge(conv, on="requisition_id", how="left")
rq["converted"] = rq["po_line_ordered_on"].notna()
rq["days_to_po"] = (rq["po_line_ordered_on"] - rq["requested_on"]).dt.days
print(f"\nRequisitions: {len(rq)} | converted: {rq['converted'].sum()} ({rq['converted'].mean() * 100:.1f}%)")
print("Requested between", rq["requested_on"].min().date(), "and", rq["requested_on"].max().date(),
      "| earliest PO line:", pl["ordered_on"].min().date())

monthly = (lines.assign(month=lines["po_ordered_on"].dt.to_period("M").astype(str))
           .groupby("month").agg(pos=("purchase_order_id", "nunique"), lines=("po_line_id", "size"),
                                 spend=("line_amount", "sum")).reset_index())
print("\nSpend by PO month:")
print(monthly.round(0).to_string(index=False))

lines.to_csv(PROC / "procurement_lines_final.csv", index=False, float_format="%.4f")
ii.to_csv(PROC / "procurement_invoices_final.csv", index=False, float_format="%.4f")
sc.round(3).reset_index().to_csv(PROC / "procurement_suppliers_final.csv", index=False)
rq.to_csv(PROC / "procurement_requisitions_final.csv", index=False)
monthly.round(2).to_csv(PROC / "procurement_monthly.csv", index=False)
print("\nSaved: procurement_lines_final, procurement_invoices_final, procurement_suppliers_final, "
      "procurement_requisitions_final, procurement_monthly (all in data/processed)")
