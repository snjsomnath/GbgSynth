"""Check which agents 20+ have income_source=None."""
import sys
sys.path.insert(0, '/Users/ssanjay/GitHub/GbgSynth')
from gbgsynth import GbgSynth

gs = GbgSynth()
area = gs.synthesize('101')

agents_20plus = [a for a in area.individuals if a.age >= 20]
has_source = [a for a in agents_20plus if getattr(a, 'income_source', None) is not None]
no_source = [a for a in agents_20plus if getattr(a, 'income_source', None) is None]

print(f'Total agents 20+: {len(agents_20plus)}')
print(f'With income_source: {len(has_source)}')
print(f'Without income_source (None): {len(no_source)}')

# Check ages of those without source  
if no_source:
    ages = [a.age for a in no_source]
    print(f'\nAge range of None agents: {min(ages)}-{max(ages)}')
    # Age distribution
    from collections import Counter
    age_bins = Counter()
    for a in ages:
        if a < 25: age_bins['20-24'] += 1
        elif a < 35: age_bins['25-34'] += 1
        elif a < 45: age_bins['35-44'] += 1
        elif a < 55: age_bins['45-54'] += 1
        elif a < 65: age_bins['55-64'] += 1
        elif a < 75: age_bins['65-74'] += 1
        else: age_bins['75+'] += 1
    print('Age breakdown of None agents:')
    for band in ['20-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']:
        print(f'  {band}: {age_bins.get(band, 0)}')

# Also check the 16-19 group specifically
teens = [a for a in area.individuals if 16 <= a.age <= 19]
print(f'\nAgents 16-19: {len(teens)}')
teens_src = [a for a in teens if getattr(a, 'income_source', None) is not None]
print(f'  with income_source: {len(teens_src)}')
