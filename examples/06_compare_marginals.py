"""
06 · Compare to Census Marginals
==================================
Check how well the synthetic population matches the real census
distributions (age, sex, household size, housing type, etc.).
"""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.ERROR, format="%(message)s")

city = GbgSynth(year=2023)
haga = city.synthesize("Haga")

# Detailed comparison prints a table and returns a dict
comparison = haga.compare_to_marginals(print_report=True)

# Programmatic access to overall fit
ov = comparison["overall"]
print(f"\nOverall fit:")
print(f"  RMSE:        {ov['rmse']:.2f}")
print(f"  MAE:         {ov['mae']:.2f}")
print(f"  MAPE:        {ov.get('mape', 0):.1f}%")
print(f"  Correlation: {ov['correlation']:.4f}")
