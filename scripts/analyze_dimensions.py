"""Analyze per-dimension MAPE from batch results."""
import pandas as pd

df = pd.read_csv('output/detailed_comparisons.csv')

# Exclude informational dimensions (same as the grading code)
scored = df[~df['dimension'].str.contains('informational', case=False)]

# Per area, per dimension: mean absolute error
area_dim = (scored.groupby(['area_code', 'area_name', 'dimension'])['error_pct']
            .apply(lambda x: x.abs().mean())
            .reset_index())
area_dim.columns = ['area_code', 'area_name', 'dimension', 'dim_mape']

# Per dimension: mean across all areas
dim_avg = area_dim.groupby('dimension')['dim_mape'].mean().sort_values(ascending=False)
print('=== Per-dimension MAPE (averaged across all 96 areas) ===')
for dim, val in dim_avg.items():
    print(f'  {dim:40s} {val:6.1f}%')

# Overall MAPE = mean of per-dimension MAPEs per area
area_mape = area_dim.groupby(['area_code', 'area_name'])['dim_mape'].mean().reset_index()
area_mape.columns = ['area_code', 'area_name', 'overall_mape']
print(f'\nOverall avg MAPE: {area_mape["overall_mape"].mean():.1f}%')

a = (area_mape['overall_mape'] <= 5).sum()
b = ((area_mape['overall_mape'] > 5) & (area_mape['overall_mape'] <= 10)).sum()
c = ((area_mape['overall_mape'] > 10) & (area_mape['overall_mape'] <= 15)).sum()
d = ((area_mape['overall_mape'] > 15) & (area_mape['overall_mape'] <= 25)).sum()
f_count = (area_mape['overall_mape'] > 25).sum()
print(f'A: {a}, B: {b}, C: {c}, D: {d}, F: {f_count}')

# Show income source category breakdown
print('\n=== Income Source per-category error (all areas) ===')
inc = scored[scored['dimension'] == 'Income Source Distribution']
cat_avg = inc.groupby('category')['error_pct'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
for cat, val in cat_avg.items():
    print(f'  {cat:40s} {val:6.1f}%')

# Show household role breakdown
print('\n=== Household Role per-category error (all areas) ===')
hr = scored[scored['dimension'] == 'Household Role Distribution']
cat_avg2 = hr.groupby('category')['error_pct'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
for cat, val in cat_avg2.items():
    print(f'  {cat:40s} {val:6.1f}%')

# Show top 10 worst areas
print('\n=== Top 10 worst areas by MAPE ===')
worst = area_mape.sort_values('overall_mape', ascending=False).head(10)
for _, row in worst.iterrows():
    print(f"  {row['area_name']:40s} {row['overall_mape']:6.1f}%")

# Show top 10 best areas
print('\n=== Top 10 best areas by MAPE ===')
best = area_mape.sort_values('overall_mape', ascending=True).head(10)
for _, row in best.iterrows():
    print(f"  {row['area_name']:40s} {row['overall_mape']:6.1f}%")

# Dimension breakdown for best area (Heden)
print('\n=== Dimension breakdown for top-5 best areas ===')
for _, row in best.head(5).iterrows():
    code = row['area_code']
    name = row['area_name']
    dims = area_dim[area_dim['area_code'] == code].sort_values('dim_mape', ascending=False)
    print(f'\n  {name} (overall: {row["overall_mape"]:.1f}%):')
    for _, d in dims.iterrows():
        print(f'    {d["dimension"]:40s} {d["dim_mape"]:6.1f}%')
