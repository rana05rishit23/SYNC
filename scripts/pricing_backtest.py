"""Backtest: re-price the historical sales that are rule-valid with our policy and compare to what was charged."""
import sys, pandas as pd
sys.path.insert(0, ".")
from backend.app.catalogue import BusinessProfile, load_catalogue, validate_bundle
from backend.app.pricing import build_quote, term_options
cat = load_catalogue()
L = pd.read_csv("data/smb_leads.csv"); G = pd.read_csv("data/mock_gstin_registry.csv").set_index("gstin").status
rows = []
for _, r in L[L.outcome == "Activated"].iterrows():
    items = {x.split("x")[0]: int(x.split("x")[1]) for x in r.bundle_products.split(";")}
    st = G.get(r.gstin) if r.has_gstin == "Y" else None
    prof = BusinessProfile(int(r.city_tier), int(r.employee_count), int(r.num_sites), st)
    if validate_bundle(prof, items, cat): continue
    t = 24 if r.contract_months >= 24 else 12
    if t not in term_options(items, cat): continue
    q = build_quote(prof, items, t, cat)
    rows.append((r.segment, t, r.monthly_bundle_value_inr, q.monthly_total))
d = pd.DataFrame(rows, columns=["segment", "term", "historical", "ours"])
d["ratio"] = d.ours / d.historical
print("rule-valid historical bundles re-priced:", len(d))
print("overall: historical total", d.historical.sum(), "| ours", d.ours.sum(), "| ratio", round(d.ours.sum() / d.historical.sum(), 3))
print(d.groupby("term").ratio.mean().round(3).to_string())
print(d.groupby("segment").ratio.mean().round(3).to_string())
