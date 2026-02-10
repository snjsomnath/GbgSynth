#!/usr/bin/env python3
"""Check what the synthesizer assigns vs what the new income table provides."""
from gbgsynth import GbgSynth

city = GbgSynth(year=2023, log_level='WARNING')
area = city.get_area('107')
area.generate()

# Individual attributes
ind = area.individuals[0]
attrs = [a for a in dir(ind) if not a.startswith('_')]
print('Individual attributes:', attrs)
print()

# Sample adults
adults = [i for i in area.individuals if i.age >= 18][:5]
for a in adults:
    print(f'  age={a.age}, sex={a.sex}, income={getattr(a,"income",None)}, '
          f'income_decile={getattr(a,"income_decile",None)}, '
          f'income_standard={getattr(a,"income_standard",None)}, '
          f'role={getattr(a,"household_role",None)}')

# Census totals
print()
print('=== Census (20_HuvudInk) for Haga ===')
census_men = 918+22+58+361+49+18+0+12+40
census_women = 1068+25+63+454+56+38+13+17+40
print(f'Total adults in census: Men={census_men}, Women={census_women}, Total={census_men+census_women}')

synth_adults = [i for i in area.individuals if i.age >= 18]
synth_men = [i for i in synth_adults if i.sex == 'M']
synth_women = [i for i in synth_adults if i.sex == 'F']
print(f'Synth adults: Men={len(synth_men)}, Women={len(synth_women)}, Total={len(synth_adults)}')
print()

# Current income standard comparison
print('=== Current Income Standard Comparison ===')
comp = area._compare_income_distribution()
for row in comp['comparison']:
    print(f"  {row['category'][:55]}: census={row['actual']}, synth={row['synth']}, err={row['error_pct']}%")

# What the synth knows about adults by age
print()
print('=== Synth adult age breakdown (for mapping to HuvudInk) ===')
import collections
age_bins = {'18-24': 0, '25-44': 0, '45-64': 0, '65+': 0}
for a in synth_adults:
    if a.age < 25: age_bins['18-24'] += 1
    elif a.age < 45: age_bins['25-44'] += 1
    elif a.age < 65: age_bins['45-64'] += 1
    else: age_bins['65+'] += 1
for k, v in age_bins.items():
    print(f'  {k}: {v}')
