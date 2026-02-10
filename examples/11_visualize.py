"""
11 · Visualise Population
==========================
Create publication-ready plots using the built-in plotting module.
All PNGs are saved to <project_root>/plots/.

Requires: pip install matplotlib  (optional: seaborn)
"""

import logging
from pathlib import Path
from gbgsynth import GbgSynth, plotting

logging.basicConfig(level=logging.ERROR, format="%(message)s")

OUT = Path(__file__).parent.parent / "plots"
OUT.mkdir(parents=True, exist_ok=True)

city = GbgSynth(year=2023)
haga = city.synthesize("Haga")

plotting.set_style("seaborn")

plots = {
    "age_distribution":    plotting.plot_age_distribution(haga, show_marginals=True),
    "household_size":      plotting.plot_household_size(haga, show_marginals=True),
    "population_pyramid":  plotting.plot_population_pyramid(haga),
    "car_ownership":       plotting.plot_car_ownership(haga, by="household_size"),
    "marginal_comparison": plotting.plot_marginal_comparison(haga),
    "error_analysis":      plotting.plot_error_analysis(haga),
    "scatter_comparison":  plotting.plot_scatter_comparison(haga),
}

for name, fig in plots.items():
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓ {path.name}")

# Compare three areas side-by-side
areas = [haga] + [city.synthesize(n) for n in ["Annedal", "Olivedal"]]

for metric in ("population", "cars_per_capita"):
    fig = plotting.compare_areas(areas, metric=metric)
    path = OUT / f"areas_{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓ {path.name}")

print(f"\n{len(plots) + 2} plots saved to {OUT}")
