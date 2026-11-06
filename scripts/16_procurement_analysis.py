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
                on="item_id", how="left"))
lines["line_amount"] = lines["quantity"] * lines["unit_price"]

# ------------------------------------------------------------------ A
section("A. RELATIONSHIP CHAIN, DATE RANGES AND CARDINALITY")
links = [
    ("goods_receipts.po_line_id -> po_lines.id", gr["po_line_id"], pl["id"]),
    ("supplier_invoices.po_line_id -> po_lines.id", inv["po_line_id"], pl["id"]),
    ("po_lines.purchase_order_id -> purchase_orders.id", pl["purchase_order_id"], po["id"]),
    ("po_lines.item_id -> items.id", pl["item_id"], items["id"]),
    ("po_lines.requisition_id -> requisitions.id", pl["requisition_id"], req["id"]),
    ("purchase_orders.supplier_id -> suppliers.id", po["supplier_id"], sup["id"]),
    ("purchase_orders.buyer_id -> buyers.id", po["buyer_id"], buyers["id"]),
    ("requisitions.item_id -> items.id", req["item_id"], items["id"]),
]
for label, child, parent in links:
    print(f"{label}: {child.isin(parent).mean() * 100:.0f}% of {len(child)} rows found")

print("\nDate ranges:")
for label, s in [("requisitions.requested_on", req["requested_on"]),
                 ("purchase_orders.ordered_on", po["ordered_on"]),
                 ("po_lines.ordered_on", pl["ordered_on"]),
                 ("goods_receipts.received_on", gr["received_on"]),
                 ("supplier_invoices.invoice_date", inv["invoice_date"]),
                 ("supplier_invoices.paid_on", inv["paid_on"])]:
    print(f"  {label}: {s.min().date()} to {s.max().date()} (missing {s.isna().sum()})")

print(f"\nSuppliers with at least one PO: {sup['id'].isin(po['supplier_id']).sum()} of {len(sup)}")
print(f"PO lines with at least one receipt: {pl['id'].isin(gr['po_line_id']).sum()} of {len(pl)}")
print("Receipts per PO line:")
print(gr.groupby("po_line_id").size().value_counts().sort_index().to_string())
print(f"PO lines with at least one invoice: {pl['id'].isin(inv['po_line_id']).sum()} of {len(pl)}")
print("Invoices per PO line:")
print(inv.groupby("po_line_id").size().value_counts().sort_index().to_string())
d = (lines["line_ordered_on"] - lines["po_ordered_on"]).dt.days
print("\nPO line order date minus PO order date (days):")
print(d.describe().round(1).to_string())
print("Receipt condition:", gr["condition"].value_counts().to_dict())

# ------------------------------------------------------------------ B
section("B. LEAD TIME WITH THE CORRECT JOIN (receipt -> PO line -> PO)")
r = gr.rename(columns={"id": "receipt_id"}).merge(lines, on="po_line_id", how="left")
r["lead_days"] = (r["received_on"] - r["po_ordered_on"]).dt.days
r["lead_valid"] = r["lead_days"].where(r["lead_days"] >= 0)
print(f"Receipts: {len(r)} | matched to a PO: {r['purchase_order_id'].notna().sum()}")
print("Days from PO order date to receipt date, all receipts:")
print(r["lead_days"].describe().round(1).to_string())
print(f"Receipts dated BEFORE the PO order date: {(r['lead_days'] < 0).sum()} "
      f"({(r['lead_days'] < 0).mean() * 100:.1f}%)")
first = r.sort_values("received_on").drop_duplicates("po_line_id")
valid = first[first["lead_days"] >= 0]
print(f"\nFirst receipt per PO line: {len(first)} | with non-negative lead time: {len(valid)}")
print(valid["lead_days"].describe().round(1).to_string())
print("Percentiles (first receipts, non-negative):")
print(valid["lead_days"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
chk = (r["received_qty"] / r["quantity"] - r["received_share"]).abs()
print(f"\nreceived_qty / ordered quantity vs received_share: mean abs difference = {chk.mean():.3f}")
print("-> These lead times describe office and maintenance supplies only. They are NOT used for retail products.")

# ------------------------------------------------------------------ C
section("C. SUPPLIER SCORECARD (small samples: read the counts)")
lk = lines[["po_line_id", "supplier_name", "po_payment_terms", "quantity", "unit_price"]]
iv = inv.rename(columns={"id": "invoice_id"}).merge(lk, on="po_line_id", how="left")
iv["billed_amount"] = iv["invoiced_qty"] * iv["billed_unit_price"]

sc = lines.groupby("supplier_name").agg(
    pos=("purchase_order_id", "nunique"), lines=("po_line_id", "size"),
    spend=("line_amount", "sum"))
rcs = r.groupby("supplier_name").agg(
    receipts=("receipt_id", "size"), median_lead_days=("lead_valid", "median"),
    avg_received_share=("received_share", "mean"),
    damaged_pct=("condition", lambda s: (s != "good").mean() * 100))
ivs = iv.groupby("supplier_name").agg(
    invoices=("invoice_id", "size"),
    disputed_pct=("status", lambda s: (s == "disputed").mean() * 100),
    unpaid_pct=("paid_on", lambda s: s.isna().mean() * 100),
    avg_abs_price_drift_pct=("billing_drift", lambda s: (s - 1).abs().mean() * 100))
sc = sc.join(rcs).join(ivs).sort_values("spend", ascending=False)
print(sc.round(1).to_string())
sc.round(2).to_csv(REP / "16_supplier_scorecard.csv")
print(f"\nSaved: {REP / '16_supplier_scorecard.csv'}")

# ------------------------------------------------------------------ D
section("D. INVOICES AND PAYMENTS")
print("Invoice status:", inv["status"].value_counts().to_dict())
fr = gr.groupby("po_line_id")["received_on"].min().rename("first_received").reset_index()
iv = iv.merge(fr, on="po_line_id", how="left")
iv["days_after_receipt"] = (iv["invoice_date"] - iv["first_received"]).dt.days
print(f"\nInvoices whose PO line has no receipt: {iv['first_received'].isna().sum()}")
print("Invoice date minus first receipt date (days):")
print(iv["days_after_receipt"].describe().round(1).to_string())
print(f"Invoices dated before the goods arrived: {(iv['days_after_receipt'] < 0).sum()}")

iv["terms_days"] = iv["po_payment_terms"].str.extract(r"(\d+)")[0].astype(float)
paid = iv[iv["paid_on"].notna()].copy()
paid["days_to_pay"] = (paid["paid_on"] - paid["invoice_date"]).dt.days
paid["days_over_terms"] = paid["days_to_pay"] - paid["terms_days"]
print(f"\nPaid invoices: {len(paid)}")
print("Days from invoice to payment:")
print(paid["days_to_pay"].describe().round(1).to_string())
print(f"Paid later than the PO payment terms: {(paid['days_over_terms'] > 0).mean() * 100:.1f}%")

asof = max(gr["received_on"].max(), pl["ordered_on"].max(), po["ordered_on"].max(),
           req["requested_on"].max(), inv["invoice_date"].max(), inv["paid_on"].max())
un = iv[iv["paid_on"].isna()].copy()
un["age_days"] = (asof - un["invoice_date"]).dt.days
print(f"\nUnpaid invoices as of the latest date in the data ({asof.date()}):")
print(un.groupby("status").agg(invoices=("invoice_id", "size"), amount=("billed_amount", "sum"),
                               median_age_days=("age_days", "median"),
                               max_age_days=("age_days", "max")).round(1).to_string())

iv["price_ratio"] = iv["billed_unit_price"] / iv["unit_price"]
iv["qty_ratio_check"] = iv["invoiced_qty"] / iv["quantity"]
print(f"\nbilled_unit_price / PO unit_price vs billing_drift: mean abs diff = "
      f"{(iv['price_ratio'] - iv['billing_drift']).abs().mean():.3f}")
print(f"invoiced_qty / PO quantity vs qty_ratio: mean abs diff = "
      f"{(iv['qty_ratio_check'] - iv['qty_ratio']).abs().mean():.3f}")
print("billing_drift distribution:")
print(iv["billing_drift"].describe().round(3).to_string())
print(f"Invoices billed more than 2% above the PO price: {(iv['price_ratio'] > 1.02).sum()} of {len(iv)}")
over = ((iv["billed_unit_price"] - iv["unit_price"]) * iv["invoiced_qty"]).clip(lower=0)
print(f"Price excess on those invoices: {over.sum():,.0f} of {iv['billed_amount'].sum():,.0f} billed")
print("Status of invoices billed above 2% over price:")
print(iv[iv["price_ratio"] > 1.02]["status"].value_counts().to_string())

# ------------------------------------------------------------------ E
section("E. REQUISITIONS, PRICES AND SPEND")
rq = req.rename(columns={"id": "requisition_id", "item_id": "req_item_id"})
conv = rq["requisition_id"].isin(pl["requisition_id"])
print(f"Requisitions: {len(rq)} | converted into a PO line: {conv.sum()} ({conv.mean() * 100:.1f}%)")
print(rq.assign(converted=conv).groupby("requester_dept")["converted"]
      .agg(requisitions="size", converted_pct=lambda x: x.mean() * 100).round(1).to_string())

pr = (pl.rename(columns={"id": "po_line_id", "ordered_on": "line_ordered_on"})
      .merge(rq[["requisition_id", "requested_on", "requester_dept", "quantity_requested",
                 "list_price", "req_item_id"]], on="requisition_id", how="left"))
print(f"\nPO lines whose item equals the requisition item: "
      f"{(pr['item_id'] == pr['req_item_id']).mean() * 100:.1f}%")
dd = (pr["line_ordered_on"] - pr["requested_on"]).dt.days
print("Days from requisition to PO line order date:")
print(dd.describe().round(1).to_string())
print(f"PO lines ordered BEFORE the requisition date: {(dd < 0).sum()}")
print(f"Ordered quantity / requested quantity: mean = "
      f"{(pr['quantity'] / pr['quantity_requested']).mean():.2f}")
nf = (pr["unit_price"] / pr["list_price"] - pr["negotiated_factor"]).abs()
print(f"unit_price / list_price vs negotiated_factor: mean abs diff = {nf.mean():.3f}")
print(f"Average negotiated factor: {pr['negotiated_factor'].mean():.3f} (below 1 = below list price)")

print("\nSpend by catalogue:")
print(lines.groupby("catalogue")["line_amount"].agg(lines="size", spend="sum")
      .round(0).sort_values("spend", ascending=False).to_string())
print("\nTop 5 items by spend:")
print(lines.groupby("item_label")["line_amount"].sum().sort_values(ascending=False)
      .head(5).round(0).to_string())
bt = lines.merge(buyers[["id", "team"]].rename(columns={"id": "buyer_id"}), on="buyer_id", how="left")
print("\nSpend by buyer team:")
print(bt.groupby("team")["line_amount"].agg(lines="size", spend="sum").round(0)
      .sort_values("spend", ascending=False).to_string())

lines.to_csv(PROC / "procurement_lines_enriched.csv", index=False)
print(f"\nSaved: {PROC / 'procurement_lines_enriched.csv'}")
