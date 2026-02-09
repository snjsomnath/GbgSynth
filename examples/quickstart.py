"""
Quick start script for GbgSynth.

Run this to verify the installation and generate your first synthetic population.
"""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    print("GbgSynth - Quick Start")
    print("=" * 50)
    
    # Initialize
    city = GbgSynth(year=2023)
    print(f"\n{city}")  # Shows repr
    
    # List available areas
    print(f"\nAvailable areas: {len(city.list_areas())} neighbourhoods")
    print(f"First few: {city.list_areas()[:5]}")
    
    # One-liner synthesis (use name or code)
    print("\nGenerating population for Haga...")
    haga = city.synthesize("Haga")
    
    # Show result
    print(f"\n{haga}")  # Shows repr with counts
    
    # Access data as DataFrames
    print(f"\nAge distribution:")
    print(haga.individuals_df['age'].describe())
    
    # Save to files
    haga.save(output_dir="./output")
    print("\n✓ Saved to ./output/")
    
    print("\n" + "=" * 50)
    print("Success! See other examples for more features.")


if __name__ == "__main__":
    main()
