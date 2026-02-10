"""
05 · Car Ownership Analysis
============================
Analyse household-level car ownership using the synthesised data.
"""

import logging
logging.basicConfig(level=logging.ERROR)

from gbgsynth import GbgSynth

city = GbgSynth(year=2023)
area = city.synthesize("Haga")

hh = area.households_df
stats = area.get_summary_statistics()
pop = stats["total_population"]
cars = stats["total_cars"]

print("CAR OWNERSHIP — Haga")
print(f"  Total cars:           {cars}")
print(f"  Cars per capita:      {cars / pop:.3f}")
print(f"  HH with ≥1 car:      {(hh['cars'] > 0).sum()} / {len(hh)}"
      f"  ({100 * (hh['cars'] > 0).mean():.1f}%)")

print("\nMean cars by household size:")
print(hh.groupby("size")["cars"].mean().to_string())
