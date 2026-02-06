#!/usr/bin/env python
"""
Generate populations for multiple areas in batch.

This example shows how to process several neighbourhoods
and aggregate the results.
"""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    city = GbgSynth(year=2023)
    
    # Use area names instead of codes for clarity
    target_areas = ["Haga", "Annedal", "Olivedal"]
    
    all_stats = []
    
    for area_name in target_areas:
        # One-liner synthesis
        area = city.synthesize(area_name)
        
        # Save each area's output
        area.save(output_dir="./output")
        
        # Collect stats
        stats = area.get_summary_statistics()
        all_stats.append(stats)
        
        # Log summary for each area
        area.log_statistics(include_marginal_comparison=False)
    
    # Aggregate totals
    total_pop = sum(s['total_population'] for s in all_stats)
    total_hh = sum(s['total_households'] for s in all_stats)
    total_cars = sum(s['total_cars'] for s in all_stats)
    
    print(f"\nTotal across {len(target_areas)} areas:")
    print(f"  Individuals: {total_pop:,}")
    print(f"  Households: {total_hh:,}")
    print(f"  Cars: {total_cars:,}")


if __name__ == "__main__":
    main()
