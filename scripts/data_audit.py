"""Data audit: verify the README's claims and find traps BEFORE designing the product.
Run from repo root: python scripts/data_audit.py
"""
import pandas as pd, re
D = "data/"
leads = pd.read_csv(D+"smb_leads.csv"); cat = pd.read_csv(D+"product_catalogue.csv")
gst = pd.read_csv(D+"mock_gstin_registry.csv"); udy = pd.read_csv(D+"mock_udyam_registry.csv")

CH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def gstin_checksum_ok(g):
    if not isinstance(g, str) or not re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]", g): return False
    tot = 0
    for i, c in enumerate(g[:14]):
        p = CH.index(c) * (1 if i % 2 == 0 else 2)
        tot += p // 36 + p % 36
    return CH[(36 - tot % 36) % 36] == g[14]

print("1. GSTIN format+checksum valid across registry:", round(gst.gstin.map(gstin_checksum_ok).mean(), 4))
print("2. Registry status:", gst.status.value_counts().to_dict())
lg = leads[leads.has_gstin == "Y"]
m = lg.merge(gst, on="gstin", how="left", indicator=True)
print("3. Lead GSTINs found in registry:", round((m._merge == "both").mean(), 4), "of", len(lg))
print("   status of matched leads:", m.status.value_counts(dropna=False).to_dict())
ok = m[m._merge == "both"]
print("   lead name == registry trade_name:", round((ok.business_name == ok.trade_name).mean(), 3))
lu = leads[leads.has_udyam == "Y"].merge(udy, on="udyam_number", how="left", indicator=True)
print("4. Lead Udyam found in registry:", round((lu._merge == "both").mean(), 4), "of", len(lu))
print("   dup gstins in registry:", gst.gstin.duplicated().sum(), "| dup udyam:", udy.udyam_number.duplicated().sum())

c = cat.set_index("product_id")
act = leads[leads.outcome == "Activated"]
viol = dict.fromkeys(["gstin_required_but_missing", "below_min_employees", "below_min_sites", "dependency_missing", "tier_not_available"], 0)
for _, r in act.iterrows():
    items = {x.split("x")[0]: int(x.split("x")[1]) for x in r.bundle_products.split(";")}
    for pid in items:
        p = c.loc[pid]
        if p.requires_gstin == "Y" and r.has_gstin != "Y": viol["gstin_required_but_missing"] += 1
        if r.employee_count < p.min_employees: viol["below_min_employees"] += 1
        if r.num_sites < p.min_sites: viol["below_min_sites"] += 1
        if str(r.city_tier) not in str(p.available_city_tiers).split(";"): viol["tier_not_available"] += 1
        if isinstance(p.requires_one_of, str) and not (set(p.requires_one_of.split("|")) & set(items)): viol["dependency_missing"] += 1
print("5. Rule violations in the", len(act), "historical activated bundles:", viol)
print("6. Most-requested products:", leads.products_of_interest.str.split(";").explode().value_counts().head(8).to_dict())
print("7. Languages:", leads.preferred_language.value_counts().to_dict())
print("8. City tiers:", leads.city_tier.value_counts().to_dict())
# 9: bundle value check -- does monthly_bundle_value reconcile with catalogue price ranges?
def price_range(bundle):
    lo = hi = 0
    for x in bundle.split(";"):
        pid, q = x.split("x"); p = c.loc[pid]
        lo += p.price_min_inr_per_month * int(q); hi += p.price_max_inr_per_month * int(q)
    return lo, hi
inr = sum(lo <= r.monthly_bundle_value_inr <= hi for _, r in act.iterrows() for lo, hi in [price_range(r.bundle_products)])
print("9. Historical bundle values within catalogue min-max range:", inr, "of", len(act))
