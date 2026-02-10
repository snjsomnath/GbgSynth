"""Check total population alignment for income source."""
import pandas as pd

df = pd.read_csv('output/detailed_comparisons.csv')
inc = df[df['dimension'] == 'Income Source Distribution']

# Per area: census total vs synth total
areas = inc.groupby(['area_code', 'area_name']).agg(
    census_total=('census_count', 'sum'),
    synth_total=('synth_count', 'sum')
).reset_index()
areas['diff_pct'] = (areas['synth_total'] - areas['census_total']) / areas['census_total'] * 100

print('=== Income Source total population (census vs synth) ===')
print(f'{"Area":40s} {"Census":>8s} {"Synth":>8s} {"Diff%":>8s}')
for _, row in areas.head(15).iterrows():
    print(f'{row["area_name"]:40s} {row["census_total"]:8.0f} {row["synth_total"]:8.0f} {row["diff_pct"]:+7.1f}%')

print(f'\nMean diff: {areas["diff_pct"].mean():+.1f}%')
print(f'Median diff: {areas["diff_pct"].median():+.1f}%')
print(f'Areas with >5% diff: {(areas["diff_pct"].abs() > 5).sum()}')

# The census counts adults 20+. The synth assigns to agents age 20+.
# Are these the same number of people?
# Check via age distribution
age = df[df['dimension'] == 'Age Distribution']
age101 = age[age['area_code'] == 101]
print('\n=== Age distribution for 101 Kungsladugård ===')
for _, row in age101.iterrows():
    print(f'  {row["category"]:20s} census={row["census_count"]:5.0f}  synth={row["synth_count"]:5.0f}')

# Sum adults 20+ from age distribution
adult_cats = [c for c in age101['category'].unique() if not any(x in c for x in ['0-5', '6-15', '16-19'])]
adult_census = age101[age101['category'].isin(adult_cats)]['census_count'].sum()
adult_synth = age101[age101['category'].isin(adult_cats)]['synth_count'].sum()
print(f'\nAdults 20+ from age dist: census={adult_census:.0f}, synth={adult_synth:.0f}')

inc101 = inc[inc['area_code'] == 101]
inc_census = inc101['census_count'].sum()
inc_synth = inc101['synth_count'].sum()
print(f'Adults 20+ from income source: census={inc_census:.0f}, synth={inc_synth:.0f}')
print(f'Difference: age says {adult_synth:.0f} agents 20+, income source assigned to {inc_synth:.0f}')
