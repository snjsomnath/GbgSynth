#!/usr/bin/env python
"""
Discover all available areas in Gothenburg.

This example shows how to list all primary areas
that can be synthesized.
"""

from gbgsynth import GbgSynth


def main():
    city = GbgSynth(year=2023)
    
    # Get list of area names (no API call needed)
    area_names = city.list_areas()
    print(f"Found {len(area_names)} primary areas in Gothenburg:\n")
    
    # Show first 10
    for i, name in enumerate(area_names[:10], 1):
        # Look up the code if needed
        code = city.get_area_code(name)
        print(f"{i:2d}. {code}: {name}")
    
    print(f"\n... and {len(area_names) - 10} more areas")
    
    # Or get full dict with codes
    all_areas = city.get_all_areas()
    print(f"\nExample: area code '107' -> '{all_areas['107']}'")


if __name__ == "__main__":
    main()
