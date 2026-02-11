"""Summarise per-dimension MAPE across all 96 areas (topdown engine)."""
import csv
import numpy as np

dims = [
    'sex', 'age', 'role', 'household_size',
    'housing_type', 'education', 'income_source',
]

with open('output/summary_report.csv') as f:
    rows = list(csv.DictReader(f))

print(f"Total areas: {len(rows)}")
grades = {}
for r in rows:
    g = r.get('quality_grade', '?')
    grades[g] = grades.get(g, 0) + 1
print(f"Quality grades: {grades}\n")

hdr = f"{'Dimension':<20} {'Median':>8} {'Mean':>8} {'P90':>8} {'Max':>8} {'> 10%':>10}"
print(hdr)
print("-" * len(hdr))

for d in dims:
    col = f"{d}_mean_error_pct"
    vals = []
    for r in rows:
        v = r.get(col, '')
        if v:
            try:
                vals.append(float(v))
            except ValueError:
                pass
    if vals:
        a = np.array(vals)
        n_bad = int(np.sum(a > 10))
        print(f"{d:<20} {np.median(a):>7.1f}% {np.mean(a):>7.1f}% "
              f"{np.percentile(a, 90):>7.1f}% {np.max(a):>7.1f}% "
              f"{n_bad:>6}/{len(vals)}")
    else:
        print(f"{d:<20} (no data)")

# Overall MAPE per area
print()
area_mapes = []
for r in rows:
    errs = []
    for d in dims:
        v = r.get(f"{d}_mean_error_pct", '')
        if v:
            try:
                errs.append(float(v))
            except ValueError:
                pass
    if errs:
        area_mapes.append(np.mean(errs))

a = np.array(area_mapes)
print(f"Overall MAPE across {len(a)} areas:")
print(f"  Median: {np.median(a):.1f}%")
print(f"  Mean:   {np.mean(a):.1f}%")
print(f"  P10:    {np.percentile(a, 10):.1f}%")
print(f"  P90:    {np.percentile(a, 90):.1f}%")
print(f"  Min:    {np.min(a):.1f}%")
print(f"  Max:    {np.max(a):.1f}%")

# Worst areas
print(f"\nWorst 5 areas:")
labels = []
for r in rows:
    errs = []
    for d in dims:
        v = r.get(f"{d}_mean_error_pct", '')
        if v:
            try:
                errs.append(float(v))
            except ValueError:
                pass
    if errs:
        labels.append((r.get('area_name', '?'), np.mean(errs)))
labels.sort(key=lambda x: -x[1])
for name, mape in labels[:5]:
    print(f"  {name:<30} {mape:.1f}%")

print(f"\nBest 5 areas:")
for name, mape in labels[-5:]:
    print(f"  {name:<30} {mape:.1f}%")
