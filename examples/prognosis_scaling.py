"""
Prognosis Scaling Example
=========================

Scale a synthesised neighbourhood population to a future year using
official population prognosis data from Gothenburg City, and visualise
the demographic shifts across age, sex, household structure, and more.

The prognosis is published at the mellanområde (intermediate area)
level for years 2025–2032. GbgSynth automatically maps each
primärområde to its parent mellanområde, fetches the age-specific
prognosis, and scales the census marginals before re-synthesising.

Usage:
    python examples/prognosis_scaling.py
"""

import logging
import os
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from gbgsynth import GbgSynth, PrognosisScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ── colour palette ────────────────────────────────────────────────────
C_BASE = "#5B8DB8"       # steel blue  – base year
C_FUTURE = "#E8734A"     # warm coral  – future year
C_MALE = "#5B8DB8"
C_FEMALE = "#E8734A"
C_MALE_F = "#89B4D4"     # lighter versions for future overlay
C_FEMALE_F = "#F2A88A"
YEAR_COLOURS = plt.cm.viridis(np.linspace(0.15, 0.95, 8))


def _age_group(age: int) -> str:
    """Bin a single year of age into the census age groups."""
    if age <= 17:
        return "0–17"
    elif age <= 24:
        return "18–24"
    elif age <= 44:
        return "25–44"
    elif age <= 64:
        return "45–64"
    elif age <= 79:
        return "65–79"
    else:
        return "80+"


AGE_GROUP_ORDER = ["0–17", "18–24", "25–44", "45–64", "65–79", "80+"]
HH_SIZE_LABELS = ["1", "2", "3", "4", "5", "6+"]


# ======================================================================
# Individual plot helpers
# ======================================================================

def plot_age_group_comparison(ax, base, future, base_year, target_year):
    """Side-by-side bar chart of census age groups."""
    base_counts = Counter(_age_group(a.age) for a in base.individuals)
    future_counts = Counter(_age_group(a.age) for a in future.individuals)

    x = np.arange(len(AGE_GROUP_ORDER))
    w = 0.35
    b_vals = [base_counts.get(g, 0) for g in AGE_GROUP_ORDER]
    f_vals = [future_counts.get(g, 0) for g in AGE_GROUP_ORDER]

    ax.bar(x - w / 2, b_vals, w, label=str(base_year), color=C_BASE, alpha=0.85)
    ax.bar(x + w / 2, f_vals, w, label=str(target_year), color=C_FUTURE, alpha=0.85)

    # % change annotations
    for i, (bv, fv) in enumerate(zip(b_vals, f_vals)):
        if bv > 0:
            pct = (fv - bv) / bv * 100
            colour = "#2d8a2d" if pct >= 0 else "#c0392b"
            ax.text(x[i] + w / 2, fv + max(b_vals) * 0.02,
                    f"{pct:+.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=colour)

    ax.set_xticks(x)
    ax.set_xticklabels(AGE_GROUP_ORDER)
    ax.set_ylabel("Population")
    ax.set_title("Age group distribution")
    ax.legend()


def plot_population_pyramids(ax, base, future, base_year, target_year):
    """Overlaid population pyramids for base and future."""
    bins_5y = list(range(0, 101, 5))
    labels_5y = [f"{lo}–{lo+4}" for lo in range(0, 100, 5)]

    def _pyramid_data(area):
        m = [0] * 20
        f = [0] * 20
        for ind in area.individuals:
            idx = min(ind.age // 5, 19)
            if ind.sex == "male":
                m[idx] += 1
            else:
                f[idx] += 1
        return m, f

    bm, bf = _pyramid_data(base)
    fm, ff = _pyramid_data(future)
    y = np.arange(20)

    # Future bars (behind, wider)
    ax.barh(y, [-v for v in fm], height=0.8, color=C_MALE_F, alpha=0.55,
            label=f"Male {target_year}")
    ax.barh(y, ff, height=0.8, color=C_FEMALE_F, alpha=0.55,
            label=f"Female {target_year}")
    # Base bars (front, narrower)
    ax.barh(y, [-v for v in bm], height=0.55, color=C_MALE, alpha=0.9,
            label=f"Male {base_year}")
    ax.barh(y, bf, height=0.55, color=C_FEMALE, alpha=0.9,
            label=f"Female {base_year}")

    max_v = max(max(bm + fm), max(bf + ff)) * 1.15
    ax.set_xlim(-max_v, max_v)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{abs(int(v))}"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels_5y, fontsize=7)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Count")
    ax.set_title("Population pyramid overlay")
    ax.legend(fontsize=7, loc="upper right")


def plot_household_size(ax, base, future, base_year, target_year):
    """Household-size distribution comparison."""
    def _hh_counts(area):
        c = Counter(min(hh.size, 6) for hh in area.households)
        return [c.get(int(l) if l != "6+" else 6, 0) for l in HH_SIZE_LABELS]

    x = np.arange(len(HH_SIZE_LABELS))
    w = 0.35
    ax.bar(x - w / 2, _hh_counts(base), w, label=str(base_year),
           color=C_BASE, alpha=0.85)
    ax.bar(x + w / 2, _hh_counts(future), w, label=str(target_year),
           color=C_FUTURE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(HH_SIZE_LABELS)
    ax.set_xlabel("Household size")
    ax.set_ylabel("Households")
    ax.set_title("Household size distribution")
    ax.legend()


def plot_household_type(ax, base, future, base_year, target_year):
    """Household-type composition (single, couple, single-parent, other)."""
    def _hh_type_counts(area):
        types = Counter()
        for hh in area.households:
            if hh.is_single():
                types["Single person"] += 1
            elif hh.is_couple() and hh.num_children == 0:
                types["Couple (no child)"] += 1
            elif hh.is_couple() and hh.num_children > 0:
                types["Couple + children"] += 1
            elif hh.is_single_parent():
                types["Single parent"] += 1
            else:
                types["Other"] += 1
        return types

    type_labels = ["Single person", "Couple (no child)",
                   "Couple + children", "Single parent", "Other"]
    bc = _hh_type_counts(base)
    fc = _hh_type_counts(future)
    x = np.arange(len(type_labels))
    w = 0.35

    ax.bar(x - w / 2, [bc.get(t, 0) for t in type_labels], w,
           label=str(base_year), color=C_BASE, alpha=0.85)
    ax.bar(x + w / 2, [fc.get(t, 0) for t in type_labels], w,
           label=str(target_year), color=C_FUTURE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Households")
    ax.set_title("Household type")
    ax.legend(fontsize=8)


def plot_sex_ratio(ax, base, future, base_year, target_year):
    """Sex-ratio comparison across age groups."""
    def _sex_ratios(area):
        m = Counter(_age_group(a.age) for a in area.individuals if a.sex == "male")
        f = Counter(_age_group(a.age) for a in area.individuals if a.sex == "female")
        return [m.get(g, 0) / max(f.get(g, 0), 1) for g in AGE_GROUP_ORDER]

    x = np.arange(len(AGE_GROUP_ORDER))
    w = 0.35
    ax.bar(x - w / 2, _sex_ratios(base), w, label=str(base_year),
           color=C_BASE, alpha=0.85)
    ax.bar(x + w / 2, _sex_ratios(future), w, label=str(target_year),
           color=C_FUTURE, alpha=0.85)
    ax.axhline(1.0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_GROUP_ORDER)
    ax.set_ylabel("Male / Female ratio")
    ax.set_title("Sex ratio by age group")
    ax.legend(fontsize=8)


def plot_dependency_and_cars(ax, base, future, base_year, target_year):
    """Key summary indicators: dependency ratio, avg HH size, cars/capita."""
    def _indicators(area):
        pop = len(area.individuals)
        working = sum(1 for a in area.individuals if 18 <= a.age <= 64)
        dependent = pop - working
        hh = len(area.households)
        cars = sum(h.cars for h in area.households)
        return {
            "Dependency\nratio": dependent / max(working, 1),
            "Avg HH\nsize": pop / max(hh, 1),
            "Cars per\ncapita": cars / max(pop, 1),
            "Children\nshare (%)": sum(1 for a in area.individuals if a.age < 18) / max(pop, 1) * 100,
            "Elderly\nshare (%)": sum(1 for a in area.individuals if a.age >= 65) / max(pop, 1) * 100,
        }

    bi = _indicators(base)
    fi = _indicators(future)
    labels = list(bi.keys())
    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w / 2, [bi[l] for l in labels], w, label=str(base_year),
           color=C_BASE, alpha=0.85)
    ax.bar(x + w / 2, [fi[l] for l in labels], w, label=str(target_year),
           color=C_FUTURE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Key demographic indicators")
    ax.legend(fontsize=8)


def plot_prognosis_trajectory(ax, pri_code):
    """Line chart of projected total population 2025→2032."""
    years = list(range(2025, 2033))
    totals = []
    for yr in years:
        s = PrognosisScaler(base_year=2025, target_year=yr)
        _, tdf = s.get_prognosis(pri_code)
        totals.append(int(tdf["count"].sum()))

    ax.plot(years, totals, "o-", color=C_FUTURE, lw=2, markersize=6)
    ax.fill_between(years, totals, alpha=0.12, color=C_FUTURE)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mellanområde population")
    ax.set_title("Population prognosis trajectory")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xticks(years)


def plot_age_heatmap(ax, pri_code):
    """Heatmap of per-census-bin scale factors across all prognosis years.

    Uses the single-year prognosis data to compute exact factors for
    each census age bin (0-5, 6-15, 16-18, 19-24, 25-34, 35-44,
    45-54, 55-64, 65-74, 75-84, 85+).
    """
    from gbgsynth.prognosis import _parse_census_age_range

    census_bins = [
        "0-5 år", "6-15 år", "16-18 år", "19-24 år",
        "25-34 år", "35-44 år", "45-54 år", "55-64 år",
        "65-74 år", "75-84 år", "85- år",
    ]
    years = list(range(2026, 2033))
    base_year = 2025

    # Fetch base prognosis once
    s0 = PrognosisScaler(base_year=base_year, target_year=base_year)
    base_df, _ = s0.get_prognosis(pri_code)
    base_by_age = base_df.set_index("age")["count"]

    matrix = []  # rows=years, cols=census bins
    for yr in years:
        s = PrognosisScaler(base_year=base_year, target_year=yr)
        _, target_df = s.get_prognosis(pri_code)
        target_by_age = target_df.set_index("age")["count"]

        row = []
        for bin_label in census_bins:
            lo, hi = _parse_census_age_range(bin_label)
            bsum = sum(base_by_age.get(a, 0) for a in range(lo, hi + 1))
            tsum = sum(target_by_age.get(a, 0) for a in range(lo, hi + 1))
            row.append(tsum / bsum if bsum > 0 else 1.0)
        matrix.append(row)

    matrix = np.array(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.8, vmax=1.4)
    ax.set_xticks(range(len(census_bins)))
    ax.set_xticklabels(census_bins, fontsize=6.5, rotation=45, ha="right")
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.set_ylabel("Target year")
    ax.set_title("Scale factors by census age bin & year")

    # Annotate cells
    for i in range(len(years)):
        for j in range(len(census_bins)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if 0.9 < val < 1.15 else "white",
                    fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Scale factor")


# ======================================================================
# Main
# ======================================================================

def main():
    city = GbgSynth(year=2024)
    area_name = "Haga"
    target_year = 2032

    # ── Synthesise base and future populations ────────────────────────
    print(f"\n{'='*60}")
    print(f"  Synthesising {area_name}: base year {city.year}")
    print(f"{'='*60}")
    base = city.synthesize(area_name)

    print(f"\n{'='*60}")
    print(f"  Scaling {area_name} → {target_year} via prognosis")
    print(f"{'='*60}")
    future = base.scale_to_year(target_year)

    # ── Print summary ─────────────────────────────────────────────────
    pop_b, pop_f = len(base.individuals), len(future.individuals)
    hh_b, hh_f = len(base.households), len(future.households)
    pct = (pop_f - pop_b) / pop_b * 100 if pop_b else 0
    summary = future.stats.get("prognosis", {})

    print(f"\n  Area:              {base.area_name}")
    print(f"  Mellanområde:      {summary.get('mel_name', '?')}")
    print(f"  Base population:   {pop_b:>6}")
    print(f"  Future population: {pop_f:>6}  ({pct:+.1f}%)")
    print(f"  Base households:   {hh_b:>6}")
    print(f"  Future households: {hh_f:>6}")

    # ── Build 8-panel figure ──────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), constrained_layout=True)
    fig.suptitle(
        f"{base.area_name}  —  Prognosis scaling  "
        f"{city.year} → {target_year}\n"
        f"(Mellanområde: {summary.get('mel_name', '?')},  "
        f"overall growth {summary.get('overall_growth', '?')})",
        fontsize=15, fontweight="bold", y=1.01,
    )
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.28)

    plot_age_group_comparison(fig.add_subplot(gs[0, 0]),
                              base, future, city.year, target_year)
    plot_population_pyramids(fig.add_subplot(gs[0, 1]),
                             base, future, city.year, target_year)
    plot_household_size(fig.add_subplot(gs[1, 0]),
                        base, future, city.year, target_year)
    plot_household_type(fig.add_subplot(gs[1, 1]),
                        base, future, city.year, target_year)
    plot_sex_ratio(fig.add_subplot(gs[2, 0]),
                   base, future, city.year, target_year)
    plot_dependency_and_cars(fig.add_subplot(gs[2, 1]),
                             base, future, city.year, target_year)
    plot_prognosis_trajectory(fig.add_subplot(gs[3, 0]),
                              base.area_code)
    plot_age_heatmap(fig.add_subplot(gs[3, 1]),
                     base.area_code)

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base.area_code}_prognosis_{target_year}.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\n  ✓ Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
