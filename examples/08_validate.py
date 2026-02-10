"""
08 · Full Validation
=====================
Run out-of-sample validation that compares the synthetic population
against census tables *not* used during fitting, plus sanity checks.
"""

import logging
from gbgsynth import GbgSynth
from gbgsynth.validation import Validator

logging.basicConfig(level=logging.ERROR, format="%(message)s")

city = GbgSynth(year=2023)
haga = city.synthesize("Haga")

validator = Validator(haga, tolerance=0.15)
report = validator.run_all_validations()

# Show summary
print(report.summary())

# Derived metrics (avg HH size, dependency ratio, …)
print("\nDerived metrics:")
for name, value in validator.compute_derived_metrics().items():
    if isinstance(value, float):
        print(f"  {name}: {value:.3f}")
    else:
        print(f"  {name}: {value}")
