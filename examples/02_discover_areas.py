"""
02 · Discover Areas
===================
Browse all 96 primary areas (primärområden) in Gothenburg
without making any API calls.
"""

import logging
logging.basicConfig(level=logging.WARNING)

from gbgsynth import GbgSynth

city = GbgSynth(year=2023)

# List area names
names = city.list_areas()
print(f"Gothenburg has {len(names)} primary areas.\n")

# Show first 15 with their codes
for i, name in enumerate(names[:15], 1):
    code = city.get_area_code(name)
    print(f"  {code}  {name}")

print(f"  ...  ({len(names) - 15} more)\n")

# Look up by name → code or code → name
print(f"Code for 'Haga': {city.get_area_code('Haga')}")
all_areas = city.get_all_areas()
print(f"Name for '107':  {all_areas['107']}")
