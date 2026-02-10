"""What-if analysis: exclude Övriga hushåll from MAPE."""
import pandas as pd

df = pd.read_csv('output/detailed_comparisons.csv')

# Exclude informational dimensions
scored = df[~df['dimension'].str.contains('informational', case=False)]

# Remove Övriga hushåll
filtered = scored[~((scored['dimension'] == 'Household Role Distribution') & 
                     (scored['category'] == 'Övriga hushåll'))]

area_dim = (filtered.groupby(['area_code', 'area_name', 'dimension'])['error_pct']
            .apply(lambda x: x.abs().mean())
            .reset_index())
area_dim.columns = ['area_code', 'area_name', 'dimension', 'dim_mape']

dim_avg = area_dim.groupby('dimension')['dim_mape'].mean().sort_values(ascending=False)
print('=== Per-dimension MAPE WITHOUT "Övriga hushåll" ===')
for dim, val in dim_avg.items():
    print(f'  {dim:40s} {val:6.1f}%')

area_mape = area_dim.groupby(['area_code', 'area_name'])['dim_mape'].mean().reset_index()
area_mape.columns = ['area_code', 'area_name', 'overall_mape']
avg = area_mape['overall_mape'].mean()
a = (area_mape['overall_mape'] <= 5).sum()
b = ((area_mape['overall_mape'] > 5) & (area_mape['overall_mape'] <= 10)).sum()
c = ((area_mape['overall_mape'] > 10) & (area_mape['overall_mape'] <= 15)).sum()
d = ((area_mape['overall_mape'] > 15) & (area_mape['overall_mape'] <= 25)).sum()
f_count = (area_mape['overall_mape'] > 25).sum()
print(f'\nOverall avg MAPE: {avg:.1f}%')
print(f'A: {a}, B: {b}, C: {c}, D: {d}, F: {f_count}')

# Also simulate: what if we also drop income source worst 3 categories?
print('\n\n=== What if entire Household Role dimension becomes informational? ===')
no_role = scored[scored['dimension'] != 'Household Role Distribution']
area_dim2 = (no_role.groupby(['area_code', 'area_name', 'dimension'])['error_pct']
             .apply(lambda x: x.abs().mean())
             .reset_index())
area_dim2.columns = ['area_code', 'area_name', 'dimension', 'dim_mape']
dim_avg2 = area_dim2.groupby('dimension')['dim_mape'].mean().sort_values(ascending=False)
for dim, val in dim_avg2.items():
    print(f'  {dim:40s} {val:6.1f}%')
area_mape2 = area_dim2.groupby(['area_code', 'area_name'])['dim_mape'].mean().reset_index()
area_mape2.columns = ['area_code', 'area_name', 'overall_mape']
avg2 = area_mape2['overall_mape'].mean()
a2 = (area_mape2['overall_mape'] <= 5).sum()
b2 = ((area_mape2['overall_mape'] > 5) & (area_mape2['overall_mape'] <= 10)).sum()
c2 = ((area_mape2['overall_mape'] > 10) & (area_mape2['overall_mape'] <= 15)).sum()
print(f'\nOverall avg MAPE: {avg2:.1f}%')
print(f'A: {a2}, B: {b2}, C: {c2}')

# Check: what proportion of population is "Övriga hushåll" in census?
print('\n=== Census Övriga hushåll as % of total pop ===')
hr = df[df['dimension'] == 'Household Role Distribution']
for cat in hr['category'].unique():
    sub = hr[hr['category'] == cat]
    pct = sub['census_count'].sum() / hr['census_count'].sum() * 100
    print(f'  {cat:30s} {pct:5.1f}%')
