"""Trace where role distortion actually happens."""
from collections import Counter
from gbgsynth import GbgSynth

city = GbgSynth(2024)
r = city.synthesize('107')  # Haga

# 1. Roles after synthesis
roles = Counter(ind.hh_role for ind in r.individuals)
print("Roles after synthesis (individual level):")
for role, cnt in roles.most_common():
    print(f"  {role:<20} {cnt}")

# 2. Other individuals
other_inds = [ind for ind in r.individuals if ind.hh_role == 'other']
print(f"\nOther individuals: {len(other_inds)}")
print(f"  with household_id: {sum(1 for i in other_inds if i.household_id is not None)}")

# 3. Household composition
print(f"\nTotal households: {len(r.households)}")
hh_sizes = Counter(hh.size for hh in r.households)
print(f"HH by size: {dict(sorted(hh_sizes.items()))}")

# 4. Role comparison — what does the evaluator compare?
comp = r.compare_to_marginals(print_report=False)
if 'role' in comp:
    print(f"\nRole comparison (what the evaluator sees):")
    for row in comp['role']['comparison']:
        print(f"  {row}")

# 5. What is the census data being compared against?
pos = r._marginals.get('household_position')
if pos is not None:
    role_col = 'Hushållsställning'
    cnt_col = 'Antal'
    if role_col in pos.columns:
        print(f"\nCensus position data (individual-level, 7 categories):")
        census = pos.groupby(role_col)[cnt_col].sum()
        for role, cnt in census.items():
            if cnt > 0:
                print(f"  {role:<55} {cnt}")
