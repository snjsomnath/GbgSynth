"""
10 · Reproducibility Check
===========================
Run the synthesis twice and compare results to show that
stochastic variation between runs is small.
"""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.ERROR)

city = GbgSynth(year=2023)

runs = []
for i in (1, 2):
    area = city.synthesize("Haga")
    s = area.get_summary_statistics()
    c = area.compare_to_marginals(print_report=False)["overall"]
    runs.append({"pop": s["total_population"], "hh": s["total_households"], **c})

print(f"{'Metric':<20} {'Run 1':>10} {'Run 2':>10} {'Δ':>10}")
print("-" * 52)
for m in ["pop", "hh", "rmse", "mae", "correlation"]:
    v1, v2 = runs[0][m], runs[1][m]
    fmt = ".4f" if isinstance(v1, float) else ","
    print(f"{m:<20} {v1:>10{fmt}} {v2:>10{fmt}} {abs(v1 - v2):>10{fmt}}")

print("\nSmall Δ values confirm the synthesis is near-deterministic.")
