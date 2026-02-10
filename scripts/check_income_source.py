"""Quick check of income source age conditioning results."""
from gbgsynth import GbgArea
from collections import Counter

area = GbgArea('107', '107 Haga', year=2024)
area.generate()

# Age distribution of adults 20+
age_bands = Counter()
for ind in area.individuals:
    if ind.age < 20:
        continue
    if ind.age <= 24: age_bands['20-24'] += 1
    elif ind.age <= 34: age_bands['25-34'] += 1
    elif ind.age <= 44: age_bands['35-44'] += 1
    elif ind.age <= 54: age_bands['45-54'] += 1
    elif ind.age <= 64: age_bands['55-64'] += 1
    elif ind.age <= 74: age_bands['65-74'] += 1
    else: age_bands['75+'] += 1

total_20plus = sum(age_bands.values())
print("Age distribution of adults 20+:")
for band in ['20-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']:
    c = age_bands.get(band, 0)
    print(f"  {band}: {c:5d} ({c/total_20plus*100:.1f}%)")
print(f"  Total 20+: {total_20plus}")
n_65plus = age_bands.get('65-74', 0) + age_bands.get('75+', 0)
print(f"  65+: {n_65plus} ({n_65plus/total_20plus*100:.1f}%)")

# What fraction of 65+ get pension?
seniors = [i for i in area.individuals if i.age >= 65]
pension_seniors = sum(1 for i in seniors if i.income_source == 'pension')
print(f"\n  Seniors (65+): {len(seniors)}, pension: {pension_seniors} ({pension_seniors/len(seniors)*100:.1f}%)")

# What fraction of 55-64 get pension?
mid = [i for i in area.individuals if 55 <= i.age <= 64]
pension_mid = sum(1 for i in mid if i.income_source == 'pension')
print(f"  55-64: {len(mid)}, pension: {pension_mid} ({pension_mid/len(mid)*100:.1f}%)")

# Per-dimension MAPE
c = area.compare_to_marginals(print_report=False)
print("\nPer-dimension MAPE:")
for key, data in c.items():
    if key == 'overall' or not data or 'comparison' not in data:
        continue
    errors = [abs(r['error_pct']) for r in data['comparison'] if r['actual'] > 0]
    dim_mape = sum(errors) / len(errors) if errors else 0
    print(f"  {key:25s}: {dim_mape:6.1f}%  ({len(errors)} categories)")

print(f"\nOverall MAPE: {c['overall']['mape']:.1f}%")

# Income source detail
print("\nIncome source detail:")
for r in c['income_source']['comparison']:
    cat = r['category'][:35]
    print(f"  {cat:35s}  census={r['actual']:5d}  synth={r['synth']:5d}  err={r['error_pct']:+.1f}%")
