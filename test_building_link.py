#!/usr/bin/env python3
"""Test dwelling-to-building linkage."""
import sys
sys.path.insert(0, '/private/tmp/GbgSynth')
from gbgsynth import GbgSynth
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

city = GbgSynth(year=2023, log_level='INFO')
area = city.get_area('107')  # Haga
print(f'Testing area: {area.area_name}')

area.generate(use_topdown=True)

print(f'\n=== RESULTS ===')
print(f'Population: {len(area.individuals):,}')
print(f'Households: {len(area.households):,}')
print(f'Dwellings: {len(area.dwellings):,}')

# Check building linkage
linked = sum(1 for d in area.dwellings if d.building_id is not None)
print(f'Dwellings linked to buildings: {linked}/{len(area.dwellings)} ({100*linked/len(area.dwellings):.1f}%)')

# Sample dwellings with building info
print('\nSample dwellings with building locations:')
for d in area.dwellings[:5]:
    bid = d.building_id[:8] + '...' if d.building_id else 'None'
    x = f'{d.centroid_x:.0f}' if d.centroid_x else 'N/A'
    y = f'{d.centroid_y:.0f}' if d.centroid_y else 'N/A'
    print(f'  Dwelling {d.dwelling_id}: {d.floor_area}m², floor {d.floor_number}, building={bid}, coords=({x}, {y})')

# Building distribution stats
if linked > 0:
    from collections import Counter
    building_counts = Counter(d.building_id for d in area.dwellings if d.building_id)
    print(f'\nBuildings with dwellings: {len(building_counts)}')
    print(f'Avg dwellings per building: {sum(building_counts.values()) / len(building_counts):.1f}')
    print(f'Max dwellings in one building: {max(building_counts.values())}')
