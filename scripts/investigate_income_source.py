"""Investigate why income source sex-level totals don't match census.

The age weights redistribute P(source|sex) into P(source|age,sex).
If we sum synth(source,sex) across all ages, does it match census(source,sex)?
If not, the age-band normalization is distorting the marginal.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('output/detailed_comparisons.csv')

# Focus on income source for Kungsladugård (101) as example
inc = df[(df['dimension'] == 'Income Source Distribution') & (df['area_code'] == 101)]
print('=== Income Source: 101 Kungsladugård ===')
print(f'{"Category":50s} {"Census":>7s} {"Synth":>7s} {"Err%":>7s}')
for _, row in inc.iterrows():
    print(f'{row["category"]:50s} {row["census_count"]:7.0f} {row["synth_count"]:7.0f} {row["error_pct"]:7.1f}')
census_total = inc['census_count'].sum()
synth_total = inc['synth_count'].sum()
print(f'{"TOTAL":50s} {census_total:7.0f} {synth_total:7.0f} {(synth_total-census_total)/census_total*100:7.1f}')

print('\n\n=== Mathematical explanation of the distortion ===')
print("""
The census gives P(source | sex). Age weights give us:
  P(source | age, sex) ∝ P(source | sex) × w(source, age)

After normalization within each age band:
  P(source | age, sex) = P(source|sex) × w(source,age) / Σ_s [P(s|sex) × w(s,age)]

The aggregate synth proportion for a source is:
  P_synth(source | sex) = Σ_age [ N(age,sex)/N(sex) × P(source|age,sex) ]

This equals P_census(source|sex) ONLY IF the normalization denominator
Σ_s [P(s|sex) × w(s,age)] is the same for every age band. But it's not,
because each age band has different weight profiles.

So the age weights STRUCTURALLY distort the sex-level marginal.
""")

# Let's quantify: simulate what happens with uniform age distribution
# to show the distortion is from the weighting, not population skew

# Use Kungsladugård census proportions as baseline
baseline_male = {
    'work': 0.62, 'unemployment': 0.025, 'studies': 0.075,
    'pension': 0.17, 'disability': 0.03, 'sickness': 0.024,
    'parental_leave': 0.007, 'financial_support': 0.023, 'no_income': 0.025
}

age_weights = {
    (20, 24): {'work': 0.55, 'unemployment': 1.0, 'studies': 8.0, 'pension': 0.0,
               'disability': 0.4, 'sickness': 0.3, 'parental_leave': 0.8,
               'financial_support': 2.5, 'no_income': 1.5},
    (25, 34): {'work': 1.1, 'unemployment': 1.0, 'studies': 1.8, 'pension': 0.0,
               'disability': 0.5, 'sickness': 0.7, 'parental_leave': 3.5,
               'financial_support': 1.2, 'no_income': 1.0},
    (35, 44): {'work': 1.2, 'unemployment': 1.0, 'studies': 0.4, 'pension': 0.0,
               'disability': 0.8, 'sickness': 1.0, 'parental_leave': 2.0,
               'financial_support': 1.0, 'no_income': 0.8},
    (45, 54): {'work': 1.2, 'unemployment': 1.0, 'studies': 0.1, 'pension': 0.0,
               'disability': 1.5, 'sickness': 1.3, 'parental_leave': 0.1,
               'financial_support': 0.8, 'no_income': 0.8},
    (55, 64): {'work': 1.0, 'unemployment': 0.8, 'studies': 0.05, 'pension': 0.2,
               'disability': 2.0, 'sickness': 1.5, 'parental_leave': 0.01,
               'financial_support': 0.6, 'no_income': 0.8},
    (65, 74): {'work': 0.25, 'unemployment': 0.01, 'studies': 0.0, 'pension': 2.8,
               'disability': 0.3, 'sickness': 0.1, 'parental_leave': 0.0,
               'financial_support': 0.2, 'no_income': 0.5},
    (75, 200): {'work': 0.05, 'unemployment': 0.0, 'studies': 0.0, 'pension': 4.0,
                'disability': 0.15, 'sickness': 0.05, 'parental_leave': 0.0,
                'financial_support': 0.2, 'no_income': 0.4},
}

# Assume roughly equal population per age band (simplified)
# Real distortion depends on actual age distribution
n_bands = len(age_weights)
print(f'\n=== Distortion with uniform age distribution ({n_bands} bands) ===')
print(f'{"Source":50s} {"Baseline":>10s} {"Weighted":>10s} {"Shift":>10s}')

aggregate = {}
for src in baseline_male:
    weighted_sum = 0
    for (amin, amax), weights in age_weights.items():
        w = weights.get(src, 1.0)
        adjusted_p = baseline_male[src] * w
        # Normalization denominator for this age band
        denom = sum(baseline_male[s] * weights.get(s, 1.0) for s in baseline_male)
        normalized_p = adjusted_p / denom
        weighted_sum += normalized_p / n_bands  # uniform population per band
    aggregate[src] = weighted_sum

for src in baseline_male:
    base = baseline_male[src]
    agg = aggregate[src]
    shift = (agg - base) / base * 100 if base > 0 else float('inf')
    print(f'{src:50s} {base:10.4f} {agg:10.4f} {shift:+9.1f}%')

# Now show with REALISTIC age distribution (typical Swedish city)
# Roughly: 20-24: 8%, 25-34: 18%, 35-44: 16%, 45-54: 15%, 55-64: 14%, 65-74: 16%, 75+: 13%
age_pop_frac = {
    (20, 24): 0.08, (25, 34): 0.18, (35, 44): 0.16,
    (45, 54): 0.15, (55, 64): 0.14, (65, 74): 0.16, (75, 200): 0.13
}

print(f'\n=== Distortion with realistic age distribution ===')
print(f'{"Source":50s} {"Baseline":>10s} {"Weighted":>10s} {"Shift":>10s}')

aggregate2 = {}
for src in baseline_male:
    weighted_sum = 0
    for (amin, amax), weights in age_weights.items():
        pop_frac = age_pop_frac[(amin, amax)]
        w = weights.get(src, 1.0)
        adjusted_p = baseline_male[src] * w
        denom = sum(baseline_male[s] * weights.get(s, 1.0) for s in baseline_male)
        normalized_p = adjusted_p / denom
        weighted_sum += normalized_p * pop_frac
    aggregate2[src] = weighted_sum

for src in baseline_male:
    base = baseline_male[src]
    agg = aggregate2[src]
    shift = (agg - base) / base * 100 if base > 0 else float('inf')
    print(f'{src:50s} {base:10.4f} {agg:10.4f} {shift:+9.1f}%')

total = sum(aggregate2.values())
print(f'{"SUM":50s} {"1.0000":>10s} {total:10.4f}')
