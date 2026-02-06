#!/usr/bin/env python
"""
Custom analysis on generated population using pandas.

This example shows how to use the built-in DataFrame properties
for analysis without manual conversion.
"""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.WARNING)


def main():
    city = GbgSynth(year=2023)
    haga = city.synthesize("Haga")
    
    # Access DataFrames directly via properties
    individuals = haga.individuals_df
    households = haga.households_df
    
    print("Age Distribution:")
    print(individuals['age'].describe())
    
    print("\nSex Distribution:")
    print(individuals['sex'].value_counts())
    
    print("\nHousehold Role Distribution:")
    print(individuals['hh_role'].value_counts())
    
    print("\nHousehold Size Distribution:")
    print(households['size'].value_counts().sort_index())
    
    print("\nCar Ownership by Household Size:")
    print(households.groupby('size')['cars'].mean())


if __name__ == "__main__":
    main()
