#!/usr/bin/env python
"""
Visualize synthetic population with built-in plotting functions.

This example demonstrates the plotting module for creating
standard visualizations of the synthetic population.

Requires: pip install matplotlib
Optional: pip install seaborn (for enhanced styling)
"""

import logging
from gbgsynth import GbgSynth, plotting

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    # Generate population
    city = GbgSynth(year=2023)
    haga = city.synthesize("Haga")
    
    # Set nice plot style (uses seaborn if available)
    plotting.set_style("seaborn")
    
    # Individual plots
    print("Creating individual plots...")
    
    # Age distribution with census comparison
    fig1 = plotting.plot_age_distribution(haga, show_marginals=True)
    fig1.savefig("age_distribution.png", dpi=150)
    print("  ✓ age_distribution.png")
    
    # Household size comparison
    fig2 = plotting.plot_household_size(haga, show_marginals=True)
    fig2.savefig("household_size.png", dpi=150)
    print("  ✓ household_size.png")
    
    # Population pyramid
    fig3 = plotting.plot_population_pyramid(haga)
    fig3.savefig("population_pyramid.png", dpi=150)
    print("  ✓ population_pyramid.png")
    
    # Car ownership by household size
    fig4 = plotting.plot_car_ownership(haga, by="household_size")
    fig4.savefig("car_ownership.png", dpi=150)
    print("  ✓ car_ownership.png")
    
    # Multi-panel marginal comparison
    fig5 = plotting.plot_marginal_comparison(haga)
    fig5.savefig("marginal_comparison.png", dpi=150)
    print("  ✓ marginal_comparison.png")
    
    # Error analysis with census data
    fig6 = plotting.plot_error_analysis(haga)
    fig6.savefig("error_analysis.png", dpi=150)
    print("  ✓ error_analysis.png")
    
    # Scatter plot: census vs synthesized
    fig7 = plotting.plot_scatter_comparison(haga)
    fig7.savefig("scatter_comparison.png", dpi=150)
    print("  ✓ scatter_comparison.png")
    
    # Or save all standard plots at once
    print("\nSaving all plots to ./plots/...")
    saved = plotting.save_all_plots(haga, output_dir="./plots")
    print(f"  ✓ Saved {len(saved)} plots")
    
    # Compare multiple areas
    print("\nComparing multiple areas...")
    areas = [city.synthesize(name) for name in ["Haga", "Annedal", "Olivedal"]]
    
    fig8 = plotting.compare_areas(areas, metric="population")
    fig8.savefig("areas_population.png", dpi=150)
    print("  ✓ areas_population.png")
    
    fig9 = plotting.compare_areas(areas, metric="cars_per_capita")
    fig9.savefig("areas_cars.png", dpi=150)
    print("  ✓ areas_cars.png")
    
    print("\nDone! Check the generated PNG files.")


if __name__ == "__main__":
    main()
