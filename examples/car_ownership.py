#!/usr/bin/env python
"""
Analyze car ownership in a synthetic population.

This example shows how to access household-level car data
and compute ownership statistics using DataFrame properties.
"""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    city = GbgSynth(year=2023)
    area = city.synthesize("Haga")
    
    # Use DataFrame property for analysis
    households = area.households_df
    
    # Car ownership statistics
    stats = area.get_summary_statistics()
    pop = stats['total_population']
    total_cars = stats['total_cars']
    hh_with_cars = (households['cars'] > 0).sum()
    
    print("Car Ownership Analysis:")
    print(f"  Total cars: {total_cars}")
    print(f"  Cars per person: {total_cars/pop:.3f}")
    print(f"  Households with cars: {100*hh_with_cars/len(households):.1f}%")
    
    print("\nCars by Household Size:")
    print(households.groupby('size')['cars'].agg(['mean', 'sum']))


if __name__ == "__main__":
    main()
