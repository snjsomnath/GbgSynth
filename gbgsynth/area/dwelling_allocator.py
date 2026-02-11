"""
Dwelling allocation pipeline for synthesised populations.

Creates dwelling units from SCB size distributions, optionally loads
building footprints, links dwellings to buildings, and matches
households to appropriately-sized dwellings.
"""

import logging
import os
import random
import unicodedata
from typing import Dict, List, Optional

import pandas as pd

from gbgsynth.models import Dwelling, Household

logger = logging.getLogger(__name__)

__all__ = ['DwellingAllocator']


class DwellingAllocator:
    """Allocates households to dwelling units for a single area.

    Parameters
    ----------
    config : Config
        Floor-area mappings and house-type translation helpers.
    area_name : str
        Display name used in log messages.
    area_code : str
        Numeric area code.
    """

    def __init__(self, config, area_name: str, area_code: str):
        self.config = config
        self.area_name = area_name
        self.area_code = area_code

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def allocate(
        self,
        households: List[Household],
        dwelling_data: Optional[pd.DataFrame],
    ) -> List[Dwelling]:
        """Run the full dwelling-allocation pipeline.

        1. Create dwelling units from SCB data.
        2. Load building footprints (if available) and link.
        3. Match households to dwellings.

        The *households* list is mutated in-place (``dwelling_id`` set).

        Returns
        -------
        list[Dwelling]
            Created dwelling objects.
        """
        logger.info("Allocating households to dwellings...")

        if dwelling_data is None or dwelling_data.empty:
            logger.warning("No dwelling data available, skipping allocation")
            return []

        dwellings = self._create_dwellings(dwelling_data)
        logger.info(f"Created {len(dwellings)} dwelling units")

        buildings = self._load_building_footprints()
        if buildings is not None and len(buildings) > 0:
            self._link_dwellings_to_buildings(dwellings, buildings)
        else:
            logger.warning(
                f"⚠️  No building footprints available for {self.area_name}. "
                "Dwellings will NOT be geo-located to buildings. "
                "Run 'generate_neighbourhood_heights()' from data_utils "
                "to download building data."
            )

        self._match_households_to_dwellings(households, dwellings)

        # Report
        allocated = sum(1 for hh in households
                        if hh.dwelling_id is not None)
        linked = sum(1 for d in dwellings if d.building_id is not None)
        logger.info(
            f"Allocated {allocated}/{len(households)} households to dwellings")
        if linked > 0:
            logger.info(
                f"Linked {linked}/{len(dwellings)} dwellings to buildings")
        elif len(dwellings) > 0:
            logger.warning(
                f"⚠️  0/{len(dwellings)} dwellings linked to buildings "
                f"for {self.area_name}. "
                "Synthetic population lacks realistic spatial distribution!"
            )

        return dwellings

    # ------------------------------------------------------------------
    # Dwelling creation
    # ------------------------------------------------------------------

    def _create_dwellings(
        self, dwelling_data: pd.DataFrame
    ) -> List[Dwelling]:
        """Create ``Dwelling`` objects from SCB dwelling size distribution."""
        dwellings: List[Dwelling] = []
        dwelling_id = 1

        floor_area_map = self.config._config.get('floor_area_mappings', {})
        house_type_map_fn = self.config.translate_house_type

        # Identify columns
        hustyp_col = area_col = None
        antal_col = 'Antal'
        for col in dwelling_data.columns:
            if 'Hustyp' in col or col == 'Hustyp':
                hustyp_col = col
            if 'Bostadsarea' in col or 'area' in col.lower():
                area_col = col

        if hustyp_col is None or area_col is None:
            logger.warning(
                "Could not find required columns. Available: "
                f"{list(dwelling_data.columns)}")
            return dwellings

        for _, row in dwelling_data.iterrows():
            house_type_sv = row[hustyp_col]
            floor_area_range = row[area_col]
            count = int(row.get(antal_col, 0))
            if count <= 0:
                continue

            area_info = floor_area_map.get(
                floor_area_range,
                {'min': 50, 'max': 80, 'midpoint': 65})
            house_type = house_type_map_fn(house_type_sv)

            for _ in range(count):
                floor_area = random.uniform(area_info['min'],
                                            area_info['max'])
                dwelling = Dwelling(
                    dwelling_id=dwelling_id,
                    floor_area=round(floor_area, 1),
                    floor_area_range=floor_area_range,
                    house_type=house_type,
                    house_type_sv=house_type_sv,
                )
                dwellings.append(dwelling)
                dwelling_id += 1

        return dwellings

    # ------------------------------------------------------------------
    # Building footprints
    # ------------------------------------------------------------------

    def _load_building_footprints(self) -> Optional[pd.DataFrame]:
        """Load building footprints from bundled GeoPackage files."""
        try:
            import geopandas as gpd  # noqa: F401
        except ImportError:
            logger.debug(
                "geopandas not available, skipping building footprints")
            return None

        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        footprints_dir = os.path.join(data_dir, 'data', 'footprints')

        area_short_name = (
            self.area_name.split(' ', 1)[-1]
            if ' ' in self.area_name else self.area_name)
        safe_name = area_short_name.replace(' ', '_').replace('/', '_')

        expected_file = f'{safe_name}_heights.gpkg'
        footprint_path = os.path.join(footprints_dir, expected_file)

        if not os.path.exists(footprint_path):
            nfd_name = unicodedata.normalize('NFD', safe_name)
            nfd_path = os.path.join(
                footprints_dir, f'{nfd_name}_heights.gpkg')
            if os.path.exists(nfd_path):
                footprint_path = nfd_path
            else:
                nfc_name = unicodedata.normalize('NFC', safe_name)
                nfc_path = os.path.join(
                    footprints_dir, f'{nfc_name}_heights.gpkg')
                if os.path.exists(nfc_path):
                    footprint_path = nfc_path
                else:
                    if os.path.exists(footprints_dir):
                        for f in os.listdir(footprints_dir):
                            f_normalized = unicodedata.normalize('NFC', f)
                            if (f_normalized.lower()
                                    == expected_file.lower()):
                                footprint_path = os.path.join(
                                    footprints_dir, f)
                                break

        if not os.path.exists(footprint_path):
            logger.warning(
                f"Building heights file not found: "
                f"{area_short_name}_heights.gpkg. "
                "Synthetic population will lack spatial building "
                "allocation.")
            return None

        try:
            import geopandas as gpd
            gdf = gpd.read_file(footprint_path)
            logger.info(
                f"Loaded {len(gdf)} building footprint polygons "
                f"for {area_short_name}")

            gdf['footprint_area'] = gdf.geometry.area
            gdf['num_floors'] = (
                (gdf['height'] / 3.0).round().astype(int).clip(lower=1))
            gdf['total_floor_area'] = (
                gdf['footprint_area'] * gdf['num_floors'])

            agg_dict = {
                'footprint_area': 'sum',
                'num_floors': 'max',
                'total_floor_area': 'sum',
                'height': 'max',
                'geometry': 'first',
            }
            if 'andamal1' in gdf.columns:
                agg_dict['andamal1'] = 'first'
            if 'objekttyp' in gdf.columns:
                agg_dict['objekttyp'] = 'first'

            merged = gdf.groupby('objektidentitet').agg(
                agg_dict).reset_index()
            merged['centroid'] = merged['geometry'].apply(
                lambda g: g.centroid)

            logger.info(f"Merged to {len(merged)} unique buildings")
            if 'objekttyp' in merged.columns:
                type_counts = merged['objekttyp'].value_counts()
                logger.info(f"Building types: {type_counts.to_dict()}")

            return merged
        except Exception as e:
            logger.warning(f"Failed to load footprints: {e}")
            return None

    # ------------------------------------------------------------------
    # Dwelling → building linking
    # ------------------------------------------------------------------

    def _link_dwellings_to_buildings(
        self,
        dwellings: List[Dwelling],
        buildings: pd.DataFrame,
    ) -> None:
        """Distribute dwellings across building footprints by floor area."""
        MFH_NET_TO_GROSS = 0.65
        SFH_NET_TO_GROSS = 0.85
        MIN_BUILDING_AREA = 50

        buildings = buildings.copy()

        if 'objekttyp' in buildings.columns:
            residential_mask = buildings['objekttyp'] == 'Bostad'
            n_residential = residential_mask.sum()
            n_total = len(buildings)
            logger.info(
                f"Filtering by objekttyp: "
                f"{n_residential}/{n_total} residential buildings")
            residential = buildings[residential_mask].copy()

            if 'andamal1' in residential.columns:
                is_sfh = residential['andamal1'].str.contains(
                    'Småhus', na=False)
                residential['net_to_gross'] = MFH_NET_TO_GROSS
                residential.loc[is_sfh, 'net_to_gross'] = SFH_NET_TO_GROSS
                logger.info(
                    f"Building types: {(~is_sfh).sum()} MFH, "
                    f"{is_sfh.sum()} SFH")
            else:
                residential['net_to_gross'] = MFH_NET_TO_GROSS
        else:
            logger.warning(
                "⚠️  No 'objekttyp' column in building data — cannot "
                "distinguish residential buildings. Filtering by size "
                "only (less accurate). Regenerate height files to "
                "include building types.")
            residential = buildings.copy()
            residential['net_to_gross'] = MFH_NET_TO_GROSS

        residential['residential_area'] = (
            residential['total_floor_area'] * residential['net_to_gross'])
        residential = residential[
            residential['residential_area'] >= MIN_BUILDING_AREA].copy()

        if len(residential) == 0:
            logger.warning(
                f"⚠️  No residential buildings found for "
                f"{self.area_name}. Dwellings cannot be allocated to "
                "buildings. Check if building data has 'objekttyp' "
                "column.")
            return

        total_building_area = residential['residential_area'].sum()
        total_dwelling_area = sum(d.floor_area for d in dwellings)

        logger.info(
            f"Building net capacity: {total_building_area:,.0f}m², "
            f"Dwelling area needed: {total_dwelling_area:,.0f}m²")

        building_data = []
        for _, row in residential.iterrows():
            centroid = row['centroid']
            num_floors = max(1, int(row['num_floors']))
            building_data.append({
                'id': row['objektidentitet'],
                'residential_area': row['residential_area'],
                'num_floors': num_floors,
                'centroid_x': centroid.x,
                'centroid_y': centroid.y,
                'used_area': 0,
                'dwellings': [],
                'floor_counts': [0] * num_floors,
            })

        building_data.sort(
            key=lambda b: b['residential_area'], reverse=True)
        dwellings_sorted = sorted(
            dwellings, key=lambda d: d.floor_area, reverse=True)

        fill_ratio = min(
            1.0, total_dwelling_area / total_building_area)
        logger.info(f"Target fill ratio: {fill_ratio * 100:.1f}%")

        for dwelling in dwellings_sorted:
            best_building = None
            best_score = float('inf')

            for b in building_data:
                remaining = b['residential_area'] - b['used_area']
                if remaining < dwelling.floor_area:
                    continue
                score = b['used_area'] / b['residential_area']
                if score < best_score:
                    best_score = score
                    best_building = b

            if best_building is None:
                candidates = sorted(
                    building_data,
                    key=lambda b: (b['residential_area']
                                   - b['used_area']),
                    reverse=True)
                if (candidates
                        and (candidates[0]['residential_area']
                             - candidates[0]['used_area']) > 0):
                    best_building = candidates[0]

            if best_building:
                dwelling.building_id = best_building['id']
                dwelling.centroid_x = best_building['centroid_x']
                dwelling.centroid_y = best_building['centroid_y']

                num_floors = best_building['num_floors']
                min_floor_idx = min(
                    range(num_floors),
                    key=lambda f: best_building['floor_counts'][f])
                dwelling.floor_number = min_floor_idx
                best_building['floor_counts'][min_floor_idx] += 1

                best_building['used_area'] += dwelling.floor_area
                best_building['dwellings'].append(dwelling.dwelling_id)

        buildings_with_dwellings = sum(
            1 for b in building_data if len(b['dwellings']) > 0)
        utilizations = [
            b['used_area'] / b['residential_area'] * 100
            for b in building_data if len(b['dwellings']) > 0]
        avg_util = (sum(utilizations) / len(utilizations)
                    if utilizations else 0)
        max_util = max(utilizations) if utilizations else 0
        min_util = min(utilizations) if utilizations else 0

        logger.info(
            f"Distributed dwellings across "
            f"{buildings_with_dwellings}/{len(building_data)} buildings")
        logger.info(
            f"Building utilization: min={min_util:.0f}%, "
            f"avg={avg_util:.0f}%, max={max_util:.0f}%")

    # ------------------------------------------------------------------
    # Household → dwelling matching
    # ------------------------------------------------------------------

    def _match_households_to_dwellings(
        self,
        households: List[Household],
        dwellings: List[Dwelling],
    ) -> None:
        """Match households to appropriately sized dwellings."""
        dwellings_by_type_and_size: Dict[str, Dict[int, List[Dwelling]]] = {
            'Småhus': {},
            'Flerbostadshus': {},
            'Övriga hus': {},
        }

        for d in dwellings:
            house_type = d.house_type_sv
            if house_type not in dwellings_by_type_and_size:
                house_type = 'Flerbostadshus'
            rec_size = d.recommended_occupants
            if rec_size not in dwellings_by_type_and_size[house_type]:
                dwellings_by_type_and_size[house_type][rec_size] = []
            dwellings_by_type_and_size[house_type][rec_size].append(d)

        for house_type in dwellings_by_type_and_size:
            for rec_size in dwellings_by_type_and_size[house_type]:
                random.shuffle(
                    dwellings_by_type_and_size[house_type][rec_size])

        households_to_match = list(households)
        random.shuffle(households_to_match)

        for hh in households_to_match:
            preferred_type = hh.assigned_hustyp
            dwelling = self._find_best_dwelling(
                hh.size, dwellings_by_type_and_size, preferred_type)
            if dwelling:
                hh.assign_dwelling(dwelling)

    @staticmethod
    def _find_best_dwelling(
        household_size: int,
        dwellings_by_type_and_size: Dict[str, Dict[int, List[Dwelling]]],
        preferred_type: Optional[str] = None,
    ) -> Optional[Dwelling]:
        """Find the best vacant dwelling for a household."""
        if (preferred_type
                and preferred_type in dwellings_by_type_and_size):
            search_order = (
                [preferred_type]
                + [t for t in dwellings_by_type_and_size
                   if t != preferred_type])
        else:
            search_order = ['Flerbostadshus', 'Småhus', 'Övriga hus']

        # Priority 1: exact match
        for house_type in search_order:
            size_groups = dwellings_by_type_and_size.get(house_type, {})
            if household_size in size_groups:
                for dwelling in size_groups[household_size]:
                    if dwelling.is_vacant():
                        return dwelling

        # Priority 2: close match (±1, ±2)
        for delta in [1, -1, 2, -2]:
            target_size = household_size + delta
            if target_size < 1:
                continue
            for house_type in search_order:
                size_groups = dwellings_by_type_and_size.get(
                    house_type, {})
                if target_size in size_groups:
                    for dwelling in size_groups[target_size]:
                        if (dwelling.is_vacant()
                                and dwelling.can_fit(household_size)):
                            return dwelling

        # Priority 3: any compatible dwelling
        for house_type in search_order:
            for size_group in dwellings_by_type_and_size.get(
                    house_type, {}).values():
                for dwelling in size_group:
                    if (dwelling.is_vacant()
                            and dwelling.can_fit(household_size)):
                        return dwelling

        return None
