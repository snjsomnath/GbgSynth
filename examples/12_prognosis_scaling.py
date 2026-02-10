"""
12 · Prognosis Scaling
=======================
Scale a neighbourhood's population to a future year (2025-2032)
using the official Gothenburg population prognosis, and compare
demographic shifts.
"""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.ERROR)

city = GbgSynth(year=2023)

# Preview scaling factors before committing
preview = city.get_area("Haga").get_prognosis_summary(
    base_year=2025, target_year=2030
)
print("Prognosis preview (Haga, 2025→2030):")
for k, v in preview.items():
    print(f"  {k}: {v}")

# Generate base and future populations
base = city.synthesize("Haga")
future = city.synthesize_future("Haga", target_year=2030)

# Compare
def age_bucket(age):
    if age < 18: return "0-17"
    if age < 65: return "18-64"
    return "65+"

from collections import Counter

base_ages = Counter(age_bucket(a.age) for a in base.individuals)
future_ages = Counter(age_bucket(a.age) for a in future.individuals)

print(f"\n{'Age group':<12} {'2023':>8} {'2030':>8} {'Change':>8}")
print("-" * 38)
for grp in ["0-17", "18-64", "65+"]:
    b, f = base_ages[grp], future_ages[grp]
    pct = (f - b) / b * 100 if b else 0
    print(f"{grp:<12} {b:>8,} {f:>8,} {pct:>+7.1f}%")

print(f"{'Total':<12} {len(base.individuals):>8,} "
      f"{len(future.individuals):>8,} "
      f"{(len(future.individuals) - len(base.individuals)) / len(base.individuals) * 100:>+7.1f}%")
