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
    
    # List available areas
    print(f"\nAvailable areas: {len(city.list_areas())} neighbourhoods")
    print(f"First few: {city.list_areas()[:5]}")
    
    # One-liner synthesis (use name or code)
    print("\nGenerating population for Haga...")
    haga = city.synthesize("Haga")
    
    # Log statistics
    haga.log_statistics()
    
    # Save to files
    haga.save(output_dir="./output")
    print("\n✓ Saved to ./output/")
    
    print("\n" + "=" * 50)
    print("Success! See other examples for more features.")


if __name__ == "__main__":
    main()
    print("  - Read the README.md for full documentation")

if __name__ == "__main__":
    main()
