"""
09 · Batch Processing
======================
Synthesise several neighbourhoods and aggregate statistics.
"""

import logging
logging.basicConfig(level=logging.ERROR)

from pathlib import Path
from gbgsynth import GbgSynth

OUTPUT_DIR = str(Path(__file__).parent.parent / "output")

city = GbgSynth(year=2023)
targets = ["Haga", "Annedal", "Olivedal"]

print(f"{'Area':<12} {'Pop':>6} {'HH':>6} {'Cars':>5} {'Corr':>7}")
print("-" * 42)

totals = {"pop": 0, "hh": 0, "cars": 0}

for name in targets:
    area = city.synthesize(name)
    area.save(output_dir=OUTPUT_DIR)
    s = area.get_summary_statistics()
    c = area.compare_to_marginals(print_report=False)
    corr = c["overall"]["correlation"]

    print(f"{name:<12} {s['total_population']:>6,} {s['total_households']:>6,}"
          f" {s['total_cars']:>5,} {corr:>7.4f}")

    totals["pop"] += s["total_population"]
    totals["hh"]  += s["total_households"]
    totals["cars"] += s["total_cars"]

print("-" * 42)
print(f"{'TOTAL':<12} {totals['pop']:>6,} {totals['hh']:>6,} {totals['cars']:>5,}")
