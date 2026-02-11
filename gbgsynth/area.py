"""
Area-specific population synthesis orchestrator.

This module provides the GbgArea class that coordinates data fetching
and synthesis for a specific geographic area.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional

from gbgsynth.api_client import PxWebClient
from gbgsynth.config import Config
from gbgsynth.exceptions import DataNotGeneratedError, InvalidDataError
from gbgsynth.models import Agent, Household, Dwelling
from gbgsynth.synthesizer import PopulationSynthesizer
from gbgsynth.prognosis import PrognosisScaler, scale_population_marginals

logger = logging.getLogger(__name__)


class GbgArea:
    """
    Represents a specific primary area in Gothenburg.

    Handles data fetching, synthesis orchestration, and export for
    a single geographic area.
    """

    def __init__(
        self,
        area_code: str,
        area_name: str,
        year: int,
        client: Optional[PxWebClient] = None,
        config: Optional[Config] = None,
        area_api_value: Optional[str] = None
    ):
        """
        Initialize area synthesizer.

        Args:
            area_code: Primary area code (e.g., "107")
            area_name: Area name (e.g., "107 Haga")
            year: Year to synthesize
            client: PxWeb client (will create if None)
            config: Configuration (will create if None)
            area_api_value: Full API value for queries (e.g., "107 Haga")
        """
        self.area_code = area_code
        self.area_name = area_name
        self.year = year
        self.client = client or PxWebClient()
        self.config = config or Config()
        # Use API value for queries, fallback to area_name if not provided
        self.area_api_value = area_api_value or area_name

        # Synthesis results
        self.individuals: List[Agent] = []
        self.households: List[Household] = []
        self.dwellings: List[Dwelling] = []
        self._is_generated = False
        
        # Store original marginals for validation
        self._marginals: dict = {}
        
        # Synthesis statistics
        self.stats: dict = {}

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        status = "generated" if self._is_generated else "not generated"
        if self._is_generated:
            return f"GbgArea('{self.area_name}', year={self.year}, pop={len(self.individuals)}, hh={len(self.households)})"
        return f"GbgArea('{self.area_name}', year={self.year}, {status})"

    def generate(self, buildings: Optional[pd.DataFrame] = None,
                 allocate_dwellings: bool = True,
                 engine: str = 'topdown') -> 'GbgArea':
        """
        Generate synthetic population for this area.

        Steps:
        1. Fetch required census tables from API (cached)
        2. Run population synthesizer
        3. Allocate dwellings (if enabled)
        4. Store results

        Args:
            buildings: Optional GeoDataFrame with building footprints for
                      spatial linking.
            allocate_dwellings: If True (default), fetch dwelling data and
                              match households to dwelling units.
            engine: Synthesis algorithm — ``'topdown'`` (default),
                ``'ipf'``, or ``'constrained_ipf'``.

        Returns:
            self (for method chaining)
            
        Example:
            >>> area = city.get_area("Haga")
            >>> area.generate().save("output/")
        """
        logger.info(f"Generating population for {self.area_name} ({self.year})")

        # Fetch data (cached by API client)
        population_data = self._fetch_population_data()
        household_data = self._fetch_household_data()
        household_position_data = self._fetch_household_position_data()
        income_data = self._fetch_income_data()
        education_level_data = self._fetch_education_level_data()
        income_source_data = self._fetch_income_source_data()
        hh_type_children_data = self._fetch_hh_type_children_data()
        car_data = self._fetch_car_data()
        
        # Validate we have required data
        if population_data.empty:
            raise InvalidDataError(
                f"No population data available for {self.area_name} ({self.year})",
                field="population_data"
            )
        if household_data.empty:
            raise InvalidDataError(
                f"No household data available for {self.area_name} ({self.year})",
                field="household_data"
            )
        
        # Store marginals for validation
        self._marginals = {
            'population': population_data.copy(),
            'household': household_data.copy(),
            'household_position': household_position_data.copy() if household_position_data is not None else None,
            'income': income_data.copy() if income_data is not None else None,
            'education_level': education_level_data.copy() if education_level_data is not None else None,
            'income_source': income_source_data.copy() if income_source_data is not None else None,
            'hh_type_children': hh_type_children_data.copy() if hh_type_children_data is not None else None,
        }

        # Synthesize
        synthesizer = PopulationSynthesizer(self.config, engine=engine)
        self.individuals, self.households = synthesizer.synthesize(
            population_data=population_data,
            household_data=household_data,
            income_data=income_data,
            car_data=car_data,
            buildings=buildings,
            household_position_data=household_position_data,
            education_level_data=education_level_data,
            income_source_data=income_source_data
        )
        
        # Store synthesis stats
        self.stats = synthesizer.stats

        self._is_generated = True
        logger.info(f"Synthesis complete: {len(self.individuals)} individuals, {len(self.households)} households")
        
        # Allocate dwellings if requested
        if allocate_dwellings:
            self._allocate_dwellings()
        
        return self  # Allow method chaining

    def _allocate_dwellings(self) -> None:
        """
        Fetch dwelling data and allocate households to dwelling units.
        
        Uses SCB data on dwelling counts by floor area and housing type
        to create dwelling units, then matches households to appropriately
        sized dwellings. If building footprints are available, dwellings
        are distributed across actual buildings.
        """
        logger.info("Allocating households to dwellings...")
        
        # Fetch dwelling size distribution
        dwelling_data = self._fetch_dwelling_data()
        if dwelling_data is None or dwelling_data.empty:
            logger.warning("No dwelling data available, skipping allocation")
            return
        
        # Create dwelling units from SCB data
        self.dwellings = self._create_dwellings(dwelling_data)
        logger.info(f"Created {len(self.dwellings)} dwelling units")
        
        # Try to load building footprints and link dwellings to buildings
        buildings = self._load_building_footprints()
        if buildings is not None and len(buildings) > 0:
            self._link_dwellings_to_buildings(buildings)
        else:
            logger.warning(
                f"⚠️  No building footprints available for {self.area_name}. "
                "Dwellings will NOT be geo-located to buildings. "
                "Run 'generate_neighbourhood_heights()' from data_utils to download building data."
            )
        
        # Match households to dwellings
        self._match_households_to_dwellings()
        
        # Report allocation results
        allocated = sum(1 for hh in self.households if hh.dwelling_id is not None)
        linked = sum(1 for d in self.dwellings if d.building_id is not None)
        logger.info(f"Allocated {allocated}/{len(self.households)} households to dwellings")
        if linked > 0:
            logger.info(f"Linked {linked}/{len(self.dwellings)} dwellings to buildings")
        elif len(self.dwellings) > 0:
            logger.warning(
                f"⚠️  0/{len(self.dwellings)} dwellings linked to buildings for {self.area_name}. "
                "Synthetic population lacks realistic spatial distribution!"
            )

    def _load_building_footprints(self) -> Optional['pd.DataFrame']:
        """
        Load building footprints for this area from bundled GeoPackage files.
        
        Returns:
            GeoDataFrame with building footprints or None if not available
        """
        import os
        import unicodedata
        
        try:
            import geopandas as gpd
        except ImportError:
            logger.debug("geopandas not available, skipping building footprints")
            return None
        
        # Build path to footprint file
        data_dir = os.path.dirname(os.path.abspath(__file__))
        footprints_dir = os.path.join(data_dir, 'data', 'footprints')
        
        # Try to find matching footprint file by area name
        # Area names like "107 Haga" -> "Haga_heights.gpkg"
        area_short_name = self.area_name.split(' ', 1)[-1] if ' ' in self.area_name else self.area_name
        safe_name = area_short_name.replace(' ', '_').replace('/', '_')
        
        # Try both NFC and NFD normalized versions (macOS uses NFD for filenames)
        expected_file = f'{safe_name}_heights.gpkg'
        footprint_path = os.path.join(footprints_dir, expected_file)
        
        if not os.path.exists(footprint_path):
            # Try NFD normalization (macOS filesystem)
            nfd_name = unicodedata.normalize('NFD', safe_name)
            nfd_file = f'{nfd_name}_heights.gpkg'
            nfd_path = os.path.join(footprints_dir, nfd_file)
            if os.path.exists(nfd_path):
                footprint_path = nfd_path
            else:
                # Try NFC normalization
                nfc_name = unicodedata.normalize('NFC', safe_name)
                nfc_file = f'{nfc_name}_heights.gpkg'
                nfc_path = os.path.join(footprints_dir, nfc_file)
                if os.path.exists(nfc_path):
                    footprint_path = nfc_path
                else:
                    # Search directory for case-insensitive match
                    if os.path.exists(footprints_dir):
                        for f in os.listdir(footprints_dir):
                            f_normalized = unicodedata.normalize('NFC', f)
                            if f_normalized.lower() == expected_file.lower():
                                footprint_path = os.path.join(footprints_dir, f)
                                break
        
        if not os.path.exists(footprint_path):
            logger.warning(
                f"Building heights file not found: {area_short_name}_heights.gpkg. "
                "Synthetic population will lack spatial building allocation."
            )
            return None
        
        try:
            gdf = gpd.read_file(footprint_path)
            logger.info(f"Loaded {len(gdf)} building footprint polygons for {area_short_name}")
            
            # Calculate building metrics per polygon
            gdf['footprint_area'] = gdf.geometry.area
            gdf['num_floors'] = (gdf['height'] / 3.0).round().astype(int).clip(lower=1)
            gdf['total_floor_area'] = gdf['footprint_area'] * gdf['num_floors']
            
            # Merge duplicate building IDs (buildings split into multiple polygons)
            # Group by objektidentitet and aggregate
            agg_dict = {
                'footprint_area': 'sum',
                'num_floors': 'max',  # Use max floors for the building
                'total_floor_area': 'sum',
                'height': 'max',
                'geometry': 'first'  # Keep first geometry for centroid
            }
            # Include andamal1 and objekttyp if available (pre-joined in height files)
            if 'andamal1' in gdf.columns:
                agg_dict['andamal1'] = 'first'
            if 'objekttyp' in gdf.columns:
                agg_dict['objekttyp'] = 'first'
                
            merged = gdf.groupby('objektidentitet').agg(agg_dict).reset_index()
            
            # Calculate centroid from first geometry
            merged['centroid'] = merged['geometry'].apply(lambda g: g.centroid)
            
            logger.info(f"Merged to {len(merged)} unique buildings")
            
            # Log building type distribution if available
            if 'objekttyp' in merged.columns:
                type_counts = merged['objekttyp'].value_counts()
                logger.info(f"Building types: {type_counts.to_dict()}")
            
            return merged
        except Exception as e:
            logger.warning(f"Failed to load footprints: {e}")
            return None

    def _link_dwellings_to_buildings(self, buildings: 'pd.DataFrame') -> None:
        """
        Distribute dwellings across building footprints based on floor area.
        
        Uses building heights to estimate total residential floor area,
        then distributes SCB dwelling counts proportionally across buildings.
        Ensures dwellings don't exceed building capacity.
        
        Args:
            buildings: DataFrame with building footprints and heights (merged by ID)
        """
        # Net-to-gross ratios for residential floor area
        # Multi-family housing (MFH) has significant common areas:
        # - Corridors, stairwells, elevators: ~15-20%
        # - Storage, laundry, technical rooms: ~5-10%
        # - Typical net dwelling area is 65-75% of gross floor area
        MFH_NET_TO_GROSS = 0.65  # Multi-family housing (apartments)
        SFH_NET_TO_GROSS = 0.85  # Single-family (most space is living area)
        
        MIN_BUILDING_AREA = 50  # Minimum net residential m² (filters tiny sheds)
        
        buildings = buildings.copy()
        
        # Filter to residential buildings using objekttyp field if available
        if 'objekttyp' in buildings.columns:
            # objekttyp == 'Bostad' covers all residential buildings
            residential_mask = buildings['objekttyp'] == 'Bostad'
            n_residential = residential_mask.sum()
            n_total = len(buildings)
            logger.info(f"Filtering by objekttyp: {n_residential}/{n_total} residential buildings")
            residential = buildings[residential_mask].copy()
            
            # Apply net-to-gross based on andamal1 (building subtype)
            if 'andamal1' in residential.columns:
                # Single-family homes have higher net-to-gross ratio
                is_sfh = residential['andamal1'].str.contains('Småhus', na=False)
                residential['net_to_gross'] = MFH_NET_TO_GROSS
                residential.loc[is_sfh, 'net_to_gross'] = SFH_NET_TO_GROSS
                logger.info(f"Building types: {(~is_sfh).sum()} MFH, {is_sfh.sum()} SFH")
            else:
                residential['net_to_gross'] = MFH_NET_TO_GROSS
        else:
            # Fallback: filter by size only (legacy behavior)
            logger.warning(
                "⚠️  No 'objekttyp' column in building data - cannot distinguish residential buildings. "
                "Filtering by size only (less accurate). Regenerate height files to include building types."
            )
            residential = buildings.copy()
            residential['net_to_gross'] = MFH_NET_TO_GROSS
        
        # Calculate net residential capacity per building
        residential['residential_area'] = residential['total_floor_area'] * residential['net_to_gross']
        
        # Filter out tiny structures
        residential = residential[residential['residential_area'] >= MIN_BUILDING_AREA].copy()
        
        if len(residential) == 0:
            logger.warning(
                f"⚠️  No residential buildings found for {self.area_name}. "
                "Dwellings cannot be allocated to buildings. Check if building data has 'objekttyp' column."
            )
            return
        
        total_building_area = residential['residential_area'].sum()
        
        # Total dwelling floor area from SCB data
        total_dwelling_area = sum(d.floor_area for d in self.dwellings)
        
        logger.info(f"Building net capacity: {total_building_area:,.0f}m², Dwelling area needed: {total_dwelling_area:,.0f}m²")
        
        # Create building list sorted by capacity (largest first)
        building_data = []
        for idx, row in residential.iterrows():
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
                'floor_counts': [0] * num_floors
            })
        
        # Sort buildings by capacity (largest first)
        building_data.sort(key=lambda b: b['residential_area'], reverse=True)
        
        # Sort dwellings by floor area (largest first for bin-packing)
        dwellings_sorted = sorted(self.dwellings, key=lambda d: d.floor_area, reverse=True)
        
        # Calculate target fill ratio based on total areas
        fill_ratio = min(1.0, total_dwelling_area / total_building_area)
        logger.info(f"Target fill ratio: {fill_ratio*100:.1f}%")
        
        # Allocate dwellings to buildings, spreading evenly by fill ratio
        for dwelling in dwellings_sorted:
            best_building = None
            best_score = float('inf')
            
            for b in building_data:
                remaining = b['residential_area'] - b['used_area']
                
                # Skip if dwelling doesn't fit at all
                if remaining < dwelling.floor_area:
                    continue
                
                # Calculate current fill ratio for this building
                current_fill = b['used_area'] / b['residential_area']
                
                # Score = how far above target fill ratio this building is
                # Lower score is better (prefer buildings that are below target)
                score = current_fill
                
                if score < best_score:
                    best_score = score
                    best_building = b
            
            # If no building has space, find one with most remaining capacity
            if best_building is None:
                candidates = sorted(building_data, key=lambda b: b['residential_area'] - b['used_area'], reverse=True)
                if candidates and (candidates[0]['residential_area'] - candidates[0]['used_area']) > 0:
                    best_building = candidates[0]
            
            if best_building:
                # Assign dwelling to building
                dwelling.building_id = best_building['id']
                dwelling.centroid_x = best_building['centroid_x']
                dwelling.centroid_y = best_building['centroid_y']
                
                # Assign to floor with fewest dwellings
                num_floors = best_building['num_floors']
                min_floor_idx = min(range(num_floors), key=lambda f: best_building['floor_counts'][f])
                dwelling.floor_number = min_floor_idx
                best_building['floor_counts'][min_floor_idx] += 1
                
                best_building['used_area'] += dwelling.floor_area
                best_building['dwellings'].append(dwelling.dwelling_id)
        
        # Log distribution stats
        buildings_with_dwellings = sum(1 for b in building_data if len(b['dwellings']) > 0)
        
        # Calculate utilization stats
        utilizations = [b['used_area'] / b['residential_area'] * 100 for b in building_data if len(b['dwellings']) > 0]
        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0
        max_util = max(utilizations) if utilizations else 0
        min_util = min(utilizations) if utilizations else 0
        
        logger.info(f"Distributed dwellings across {buildings_with_dwellings}/{len(building_data)} buildings")
        logger.info(f"Building utilization: min={min_util:.0f}%, avg={avg_util:.0f}%, max={max_util:.0f}%")

    def _fetch_dwelling_data(self) -> Optional[pd.DataFrame]:
        """Fetch dwelling size distribution from SCB."""
        table_path = self.config.get_table_id('DWELLING_SIZE')
        
        try:
            df = self.client.query_all_variables(table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} dwelling records")
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch dwelling data: {e}")
            return None

    def _create_dwellings(self, dwelling_data: pd.DataFrame) -> List[Dwelling]:
        """
        Create Dwelling objects from SCB dwelling size distribution.
        
        Args:
            dwelling_data: DataFrame with columns for Hustyp, Bostadsarea, Antal
            
        Returns:
            List of Dwelling objects
        """
        import random
        
        dwellings = []
        dwelling_id = 1
        
        # Get floor area mappings from config
        floor_area_map = self.config._config.get('floor_area_mappings', {})
        
        # House type Swedish -> English mapping
        house_type_map = {
            'Småhus': 'detached_house',
            'Flerbostadshus': 'apartment',
            'Övriga hus': 'other'
        }
        
        # Find the column names in the data
        hustyp_col = None
        area_col = None
        antal_col = 'Antal'
        
        for col in dwelling_data.columns:
            if 'Hustyp' in col or col == 'Hustyp':
                hustyp_col = col
            if 'Bostadsarea' in col or 'area' in col.lower():
                area_col = col
        
        if hustyp_col is None or area_col is None:
            logger.warning(f"Could not find required columns. Available: {list(dwelling_data.columns)}")
            return dwellings
        
        # Process each row
        for _, row in dwelling_data.iterrows():
            house_type_sv = row[hustyp_col]
            floor_area_range = row[area_col]
            count = int(row.get(antal_col, 0))
            
            if count <= 0:
                continue
            
            # Get floor area range details
            area_info = floor_area_map.get(floor_area_range, {
                'min': 50, 'max': 80, 'midpoint': 65
            })
            
            house_type = house_type_map.get(house_type_sv, 'other')
            
            # Create dwelling units
            for _ in range(count):
                # Randomize floor area within range
                floor_area = random.uniform(area_info['min'], area_info['max'])
                
                dwelling = Dwelling(
                    dwelling_id=dwelling_id,
                    floor_area=round(floor_area, 1),
                    floor_area_range=floor_area_range,
                    house_type=house_type,
                    house_type_sv=house_type_sv
                )
                dwellings.append(dwelling)
                dwelling_id += 1
        
        return dwellings

    def _match_households_to_dwellings(self) -> None:
        """
        Match households to appropriately sized dwellings.
        
        Uses a greedy algorithm that:
        1. Groups dwellings by recommended occupancy
        2. Matches households to dwellings with matching recommended size
        3. Falls back to any compatible dwelling if exact match unavailable
        """
        import random
        
        # Group dwellings by recommended occupancy and house type
        # Structure: {house_type: {recommended_size: [dwellings]}}
        dwellings_by_type_and_size: Dict[str, Dict[int, List[Dwelling]]] = {
            'Småhus': {},
            'Flerbostadshus': {},
            'Övriga hus': {}
        }
        
        for d in self.dwellings:
            house_type = d.house_type_sv
            if house_type not in dwellings_by_type_and_size:
                house_type = 'Flerbostadshus'  # Default
            
            rec_size = d.recommended_occupants
            if rec_size not in dwellings_by_type_and_size[house_type]:
                dwellings_by_type_and_size[house_type][rec_size] = []
            dwellings_by_type_and_size[house_type][rec_size].append(d)
        
        # Shuffle within each group for randomness
        for house_type in dwellings_by_type_and_size:
            for rec_size in dwellings_by_type_and_size[house_type]:
                random.shuffle(dwellings_by_type_and_size[house_type][rec_size])
        
        # Process households - shuffle for fairness
        households_to_match = list(self.households)
        random.shuffle(households_to_match)
        
        for hh in households_to_match:
            # Get preferred house type (if assigned)
            preferred_type = hh.assigned_hustyp
            
            # Try to find a suitable dwelling
            dwelling = self._find_best_dwelling(
                hh.size, 
                dwellings_by_type_and_size, 
                preferred_type
            )
            
            if dwelling:
                hh.assign_dwelling(dwelling)

    def _find_best_dwelling(
        self, 
        household_size: int, 
        dwellings_by_type_and_size: Dict[str, Dict[int, List[Dwelling]]],
        preferred_type: Optional[str] = None
    ) -> Optional[Dwelling]:
        """
        Find the best vacant dwelling for a household.
        
        Prioritizes dwellings where recommended_occupants matches household size.
        
        Args:
            household_size: Number of people in household
            dwellings_by_type_and_size: Nested dict of dwellings by type and recommended size
            preferred_type: Preferred Swedish house type
            
        Returns:
            Suitable Dwelling or None
        """
        # Define search order based on preference
        if preferred_type and preferred_type in dwellings_by_type_and_size:
            search_order = [preferred_type] + [t for t in dwellings_by_type_and_size if t != preferred_type]
        else:
            search_order = ['Flerbostadshus', 'Småhus', 'Övriga hus']
        
        # Priority 1: Exact match on recommended size
        for house_type in search_order:
            size_groups = dwellings_by_type_and_size.get(house_type, {})
            if household_size in size_groups:
                for dwelling in size_groups[household_size]:
                    if dwelling.is_vacant():
                        return dwelling
        
        # Priority 2: Close match (±1 person)
        for delta in [1, -1, 2, -2]:
            target_size = household_size + delta
            if target_size < 1:
                continue
            for house_type in search_order:
                size_groups = dwellings_by_type_and_size.get(house_type, {})
                if target_size in size_groups:
                    for dwelling in size_groups[target_size]:
                        if dwelling.is_vacant() and dwelling.can_fit(household_size):
                            return dwelling
        
        # Priority 3: Any compatible dwelling
        for house_type in search_order:
            for size_group in dwellings_by_type_and_size.get(house_type, {}).values():
                for dwelling in size_group:
                    if dwelling.is_vacant() and dwelling.can_fit(household_size):
                        return dwelling
        
        return None

    def _fetch_population_data(self) -> pd.DataFrame:
        """Fetch population demographics (age/sex/household role)."""
        table_path = self.config.get_table_id('BEFOLKNING_HH')
        
        try:
            df = self.client.query_all_variables(table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} population records")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch population data: {e}")
            # Return minimal fallback
            return pd.DataFrame({
                'Ålder': ['25-44 år'],
                'Kön': ['Män'],
                'Hushållstyp': ['Ensamstående'],
                'Antal': [100]
            })

    def _fetch_household_data(self) -> pd.DataFrame:
        """Fetch household size and type statistics."""
        table_path = self.config.get_table_id('HOUSEHOLD_SIZE')
        
        try:
            df = self.client.query_all_variables(table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} household records")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch household data: {e}")
            # Return minimal fallback
            return pd.DataFrame({
                'Hushållsstorlek': ['1 person', '2 personer'],
                'Hustyp': ['Flerbostadshus', 'Flerbostadshus'],
                'Antal': [50, 30]
            })

    def _fetch_income_data(self) -> Optional[pd.DataFrame]:
        """Fetch income distribution data.
        
        Note: The income table may use different area naming than population tables.
        We try the standard name first, then discover the correct name from metadata if needed.
        This makes the code resilient to API naming inconsistencies.
        """
        table_path = self.config.get_table_id('INCOME')
        
        # Try with standard area name first
        try:
            df = self.client.query_all_variables(table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} income records")
            return df
        except Exception as e:
            # If 400 error, the area name might differ in this table - try to discover it
            if '400' in str(e):
                discovered_name = self._discover_income_area_name(table_path)
                if discovered_name and discovered_name != self.area_api_value:
                    try:
                        df = self.client.query_all_variables(table_path, discovered_name, self.year)
                        logger.info(f"Fetched {len(df)} income records (using discovered name '{discovered_name}')")
                        return df
                    except:
                        pass
            
            # Try previous years if current year not available
            for fallback_year in [self.year - 1, self.year - 2, 2023, 2022]:
                try:
                    df = self.client.query_all_variables(table_path, self.area_api_value, fallback_year)
                    logger.info(f"Fetched {len(df)} income records (using {fallback_year} data)")
                    return df
                except:
                    continue
            logger.warning(f"Income data not available: {e}")
            return None
    
    def _fetch_education_level_data(self) -> Optional[pd.DataFrame]:
        """Fetch education level distribution data by age and sex.
        
        Uses the 23_InkomsterUtbildning_PRI.px table which provides
        population counts (Folkmängd) AND income statistics (Medianinkomst,
        Medelinkomst) by education level, age group, and sex for adults 18+.
        
        Returns the full table filtered to individual sexes, non-total
        education levels, and individual age groups. The synthesizer uses:
        - Folkmängd rows for education level assignment probabilities
        - Medianinkomst rows for realistic income amount assignment
        
        Education levels:
        - Förgymnasial utbildning (pre-secondary)
        - Gymnasial utbildning (secondary)
        - Eftergymnasial utbildning (post-secondary)
        - Uppgift saknas (unknown)
        
        Returns:
            DataFrame with education level distributions and income stats,
            or None if unavailable.
        """
        table_path = self.config.get_table_id('EDUCATION_LEVEL')
        
        if not table_path:
            logger.warning("EDUCATION_LEVEL table not configured")
            return None
        
        try:
            df = self.client.query_all_variables(table_path, self.area_api_value, self.year)
            
            # Filter to individual sexes, non-total education levels,
            # and individual age groups (not the "18- år" total)
            # Keep ALL metrics (Folkmängd, Medianinkomst, etc.)
            mask = (
                (df['Kön'] != 'Båda kön')
                & (df['Utbildningsnivå'] != 'Totalt (alla utbildningsnivåer)')
                & (df['Ålder'] != '18- år')
            )
            result = df[mask].copy()
            logger.info(f"Fetched {len(result)} education level records")
            return result
        except Exception as e:
            # Try previous years
            for fallback_year in [self.year - 1, self.year - 2, 2023, 2022]:
                try:
                    df = self.client.query_all_variables(table_path, self.area_api_value, fallback_year)
                    mask = (
                        (df['Kön'] != 'Båda kön')
                        & (df['Utbildningsnivå'] != 'Totalt (alla utbildningsnivåer)')
                        & (df['Ålder'] != '18- år')
                    )
                    result = df[mask].copy()
                    logger.info(f"Fetched {len(result)} education level records (using {fallback_year} data)")
                    return result
                except:
                    continue
            logger.warning(f"Education level data not available: {e}")
            return None

    def _fetch_income_source_data(self) -> Optional[pd.DataFrame]:
        """Fetch primary income source distribution by sex.
        
        Uses the 20_HuvudInk_PRI.px table which provides population counts
        by primary income source and sex for adults aged 20+.
        
        NOTE: This table uses numeric area indices (0, 1, 2, ...) instead of
        area names ("101 Kungsladugård"), so we resolve the index from metadata.
        
        Income source categories (9):
        - Ersättning för arbete (work)
        - Ersättning vid arbetslöshet (unemployment)
        - Ersättning för studier (studies)
        - Pension
        - Ersättning vid långvarigt nedsatt arbetsförmåga (disability)
        - Ersättning vid sjukdom (sickness)
        - Ersättning vid föräldraledighet... (parental_leave)
        - Ekonomiskt stöd (financial_support)
        - Saknar ersättningar (no_income)
        
        Returns:
            DataFrame with income source distribution by sex, or None.
        """
        import requests
        
        table_path = self.config.get_table_id('INCOME_SOURCE')
        if not table_path:
            logger.warning("INCOME_SOURCE table not configured")
            return None
        
        try:
            # Resolve numeric area index from metadata
            url = f"{self.client.BASE_URL}{table_path}"
            metadata = self.client.fetch_metadata(table_path)
            
            area_index = None
            for var in metadata.get('variables', []):
                if var['code'] == 'Område':
                    for idx, text in zip(var['values'], var['valueTexts']):
                        if text.startswith(self.area_code + ' '):
                            area_index = idx
                            break
                    break
            
            if area_index is None:
                logger.warning(f"Could not find area index for {self.area_code} in income source table")
                return None
            
            # Query with numeric index
            year_str = str(self.year)
            query = {
                'query': [
                    {'code': 'Område', 'selection': {'filter': 'item', 'values': [area_index]}},
                    {'code': 'Kön', 'selection': {'filter': 'all', 'values': ['*']}},
                    {'code': 'Huvudsaklig inkomstkälla', 'selection': {'filter': 'all', 'values': ['*']}},
                    {'code': 'År', 'selection': {'filter': 'item', 'values': [year_str]}},
                ],
                'response': {'format': 'json'},
            }
            resp = requests.post(url, json=query, timeout=30)
            resp.raise_for_status()
            df = self.client._parse_json_response(resp.json())
            logger.info(f"Fetched {len(df)} income source records")
            return df
        except Exception as e:
            # Try previous years
            for fallback_year in [self.year - 1, self.year - 2, 2023, 2022]:
                try:
                    year_str = str(fallback_year)
                    query['query'][-1]['selection']['values'] = [year_str]
                    resp = requests.post(url, json=query, timeout=30)
                    resp.raise_for_status()
                    df = self.client._parse_json_response(resp.json())
                    logger.info(f"Fetched {len(df)} income source records (using {fallback_year} data)")
                    return df
                except:
                    continue
            logger.warning(f"Income source data not available: {e}")
            return None

    def _discover_income_area_name(self, table_path: str) -> Optional[str]:
        """Discover the correct area name for the income table by querying metadata.
        
        Different API tables may use slightly different area name spellings.
        This method searches the table's available area values to find one
        matching our area code.
        """
        try:
            metadata = self.client.fetch_metadata(table_path)
            for var in metadata.get('variables', []):
                if 'område' in var.get('text', '').lower():
                    values = var.get('values', [])
                    # Find value starting with our area code
                    for v in values:
                        if v.startswith(self.area_code + ' '):
                            return v
        except Exception as e:
            logger.debug(f"Could not discover income area name: {e}")
        return None

    def _fetch_household_position_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch detailed household position data (includes child role).
        
        This table provides age×sex×position breakdown where position includes:
        - Person i gift par (married/partnered) -> cohabiting
        - Personer i samboförhållande (cohabiting) -> cohabiting
        - Ensamstående förälder (single parent) -> single
        - Barn (child) -> child
        - Ensamboende (living alone) -> single
        - Ej ensamboende personer, övriga (other) -> other
        """
        table_path = self.config.get_table_id('HOUSEHOLD_POSITION')
        
        if not table_path:
            logger.warning("HOUSEHOLD_POSITION table not configured, using default roles")
            return None
        
        try:
            df = self.client.query_all_variables(table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} household position records")
            return df
        except Exception as e:
            # Try previous years
            for fallback_year in [self.year - 1, self.year - 2, 2021, 2020]:
                try:
                    df = self.client.query_all_variables(table_path, self.area_api_value, fallback_year)
                    logger.info(f"Fetched {len(df)} household position records (using {fallback_year} data)")
                    return df
                except:
                    continue
            logger.warning(f"Household position data not available: {e}")
            return None

    def _fetch_hh_type_children_data(self) -> Optional[pd.DataFrame]:
        """Fetch household type × number of children (0-17) distribution.
        
        Uses the 10_HHTypBarnU18_PRI.px table which cross-tabulates household
        type (Ensamstående / Sammanboende / Övriga hushåll) with the number of
        children aged 0–17 living in the household (0, 1, 2, 3, 4+).
        
        This provides a joint distribution at the household level that can be
        compared against the synthesised population to validate that family
        structure is realistic.
        
        Returns:
            DataFrame with columns Hushållstyp, Antal barn 0-17 år, Antal,
            or None if the table is unavailable.
        """
        table_path = self.config.get_table_id('HH_TYPE_CHILDREN')
        if not table_path:
            logger.debug("HH_TYPE_CHILDREN table not configured")
            return None
        
        try:
            df = self.client.query_all_variables(table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} HH type×children records")
            return df
        except Exception as e:
            # Try previous year
            for fallback_year in [self.year - 1, self.year - 2]:
                try:
                    df = self.client.query_all_variables(table_path, self.area_api_value, fallback_year)
                    logger.info(f"Fetched {len(df)} HH type×children records (using {fallback_year})")
                    return df
                except Exception:
                    continue
            logger.warning(f"HH type×children data not available: {e}")
            return None

    def _fetch_car_data(self) -> Optional[pd.DataFrame]:
        """Fetch car ownership statistics."""
        table_path = self.config.get_table_id('CARS')
        
        try:
            df = self.client.query_table(table_path, self.area_api_value, self.year)
            logger.info(f"Fetched car ownership data")
            return df
        except Exception as e:
            logger.warning(f"Car data not available: {e}")
            return None

    def save(self, output_dir: str = ".", prefix: Optional[str] = None) -> dict:
        """
        Save individuals, households, and dwellings to CSV files.
        
        Creates three files:
        - {prefix}_individuals.csv
        - {prefix}_households.csv
        - {prefix}_dwellings.csv (if dwellings allocated)
        
        Args:
            output_dir: Directory to save files (default: current directory)
            prefix: Filename prefix (default: "{area_code}_{area_name}")
            
        Returns:
            Dictionary with paths to saved files:
            {'individuals': '...', 'households': '...', 'dwellings': '...' or None}
            
        Raises:
            RuntimeError: If generate() hasn't been called yet
            
        Example:
            >>> area.generate()
            >>> files = area.save()  # Saves to current dir
            >>> files = area.save("output/", prefix="haga_2024")
        """
        import os
        
        if not self._is_generated:
            raise DataNotGeneratedError("saving")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Build prefix
        if prefix is None:
            # Use short name without code prefix (e.g., "Haga" not "107 Haga")
            name_only = self.area_name.split(' ', 1)[-1] if ' ' in self.area_name else self.area_name
            safe_name = name_only.replace(' ', '_')
            prefix = f"{self.area_code}_{safe_name}"
        
        ind_path = os.path.join(output_dir, f"{prefix}_individuals.csv")
        hh_path = os.path.join(output_dir, f"{prefix}_households.csv")
        
        self.save_to_csv(ind_path)
        self.save_households_to_csv(hh_path)
        
        result = {
            'individuals': ind_path,
            'households': hh_path,
            'dwellings': None
        }
        
        # Save dwellings if available
        if self.dwellings:
            dw_path = os.path.join(output_dir, f"{prefix}_dwellings.csv")
            self.save_dwellings_to_csv(dw_path)
            result['dwellings'] = dw_path
        
        logger.info(f"Saved population to {output_dir}/")
        
        return result

    def save_dwellings_to_csv(self, filepath: str) -> None:
        """
        Save dwelling data to CSV.

        Args:
            filepath: Output file path

        Raises:
            DataNotGeneratedError: If generate() hasn't been called yet
        """
        if not self._is_generated:
            raise DataNotGeneratedError("saving dwellings")

        if not self.dwellings:
            logger.warning("No dwellings to save")
            return

        df = pd.DataFrame([dw.to_dict() for dw in self.dwellings])
        
        # Add area information
        df['area_code'] = self.area_code
        df['area_name'] = self.area_name
        df['year'] = self.year

        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} dwellings to {filepath}")

    def save_to_csv(self, filepath: str) -> None:
        """
        Save the synthetic population (individuals) to CSV.

        Args:
            filepath: Output file path

        Raises:
            DataNotGeneratedError: If generate() hasn't been called yet
        """
        if not self._is_generated:
            raise DataNotGeneratedError("saving individuals")

        # Convert to DataFrame
        df = pd.DataFrame([ind.to_dict() for ind in self.individuals])
        
        # Add area information
        df['area_code'] = self.area_code
        df['area_name'] = self.area_name
        df['year'] = self.year

        # Save
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} individuals to {filepath}")

    def save_households_to_csv(self, filepath: str) -> None:
        """
        Save household data to CSV.

        Args:
            filepath: Output file path

        Raises:
            DataNotGeneratedError: If generate() hasn't been called yet
        """
        if not self._is_generated:
            raise DataNotGeneratedError("saving households")

        df = pd.DataFrame([hh.to_dict() for hh in self.households])
        
        # Add area information
        df['area_code'] = self.area_code
        df['area_name'] = self.area_name
        df['year'] = self.year

        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} households to {filepath}")

    def to_dataframes(self) -> dict:
        """
        Get population data as pandas DataFrames.
        
        Returns:
            Dictionary with 'individuals', 'households', and 'dwellings' DataFrames
            
        Example:
            >>> dfs = area.to_dataframes()
            >>> dfs['individuals'].head()
            >>> dfs['households'].describe()
        """
        if not self._is_generated:
            raise DataNotGeneratedError("accessing data")
        
        ind_df = pd.DataFrame([ind.to_dict() for ind in self.individuals])
        ind_df['area_code'] = self.area_code
        ind_df['area_name'] = self.area_name
        ind_df['year'] = self.year
        
        hh_df = pd.DataFrame([hh.to_dict() for hh in self.households])
        hh_df['area_code'] = self.area_code
        hh_df['area_name'] = self.area_name
        hh_df['year'] = self.year
        
        result = {
            'individuals': ind_df,
            'households': hh_df,
            'dwellings': None
        }
        
        if self.dwellings:
            dw_df = pd.DataFrame([dw.to_dict() for dw in self.dwellings])
            dw_df['area_code'] = self.area_code
            dw_df['area_name'] = self.area_name
            dw_df['year'] = self.year
            result['dwellings'] = dw_df
        
        return result

    @property
    def individuals_df(self) -> pd.DataFrame:
        """
        Get individuals as a pandas DataFrame.
        
        Convenient property for quick access to individual-level data.
        
        Example:
            >>> area.generate()
            >>> area.individuals_df.head()
            >>> area.individuals_df['age'].describe()
        """
        return self.to_dataframes()['individuals']
    
    @property
    def households_df(self) -> pd.DataFrame:
        """
        Get households as a pandas DataFrame.
        
        Convenient property for quick access to household-level data.
        
        Example:
            >>> area.generate()
            >>> area.households_df.head()
            >>> area.households_df.groupby('size')['cars'].mean()
        """
        return self.to_dataframes()['households']

    def export(self, format: str, output_path: str, **kwargs) -> str:
        """
        Export population to specified format for downstream simulation tools.
        
        Supported formats:
        - "sweloadsim": SweLoadSim household energy simulation (JSON)
        
        Args:
            format: Export format name (e.g., "sweloadsim")
            output_path: Path for output file
            **kwargs: Format-specific options passed to exporter
                For "sweloadsim":
                - config: SweLoadSimConfig instance (optional)
            
        Returns:
            Path to created output file
            
        Raises:
            ValueError: If format is not recognized
            DataNotGeneratedError: If generate() hasn't been called
            
        Example:
            >>> area.generate()
            >>> area.export("sweloadsim", "haga.json")
            
            >>> # With custom config
            >>> from gbgsynth.exporters import SweLoadSimConfig
            >>> config = SweLoadSimConfig.future_2035()
            >>> area.export("sweloadsim", "haga_2035.json", config=config)
            
            >>> # With seed for reproducibility
            >>> config = SweLoadSimConfig(seed=42)
            >>> area.export("sweloadsim", "haga_reproducible.json", config=config)
        """
        from gbgsynth.exporters import get_exporter
        from pathlib import Path
        
        exporter = get_exporter(format, **kwargs)
        result_path = exporter.export(self, Path(output_path))
        logger.info(f"Exported {len(self.households)} households to {result_path} ({format} format)")
        return str(result_path)

    def get_summary_statistics(self) -> dict:
        """
        Get summary statistics for the generated population.

        Returns:
            Dictionary with key statistics

        Raises:
            RuntimeError: If generate() hasn't been called yet
        """
        if not self._is_generated:
            raise RuntimeError("Must call generate() before getting statistics")

        total_pop = len(self.individuals)
        total_hh = len(self.households)

        # Count housing types
        hustyp_counts = {}
        linked_to_buildings = 0
        for hh in self.households:
            ht = hh.assigned_hustyp or 'unassigned'
            hustyp_counts[ht] = hustyp_counts.get(ht, 0) + 1
            if hh.building_id is not None:
                linked_to_buildings += 1

        return {
            'area_code': self.area_code,
            'area_name': self.area_name,
            'year': self.year,
            'total_population': total_pop,
            'total_households': total_hh,
            'avg_household_size': total_pop / total_hh if total_hh > 0 else 0,
            'num_children': sum(1 for a in self.individuals if a.is_child()),
            'num_adults': sum(1 for a in self.individuals if a.is_adult()),
            'num_couples': sum(1 for h in self.households if h.is_couple()),
            'num_single_parent': sum(1 for h in self.households if h.is_single_parent()),
            'num_single_person': sum(1 for h in self.households if h.is_single()),
            'avg_income': sum(a.income or 0 for a in self.individuals) / total_pop if total_pop > 0 else 0,
            'total_cars': sum(h.cars for h in self.households),
            # Housing type statistics
            'hustyp_smahus': hustyp_counts.get('Småhus', 0),
            'hustyp_flerbostadshus': hustyp_counts.get('Flerbostadshus', 0),
            'hustyp_specialbostad': hustyp_counts.get('Specialbostad', 0),
            'households_linked_to_buildings': linked_to_buildings
        }

    def compare_to_marginals(self, print_report: bool = True, use_logging: bool = False) -> dict:
        """
        Compare synthesized population against original census marginals.
        
        This validates how well the synthetic population reproduces the
        target marginal distributions for key attributes.
        
        Args:
            print_report: If True, outputs a formatted comparison report
            use_logging: If True (and print_report=True), uses logging.info
                        instead of print. Useful for library integration.
            
        Returns:
            Dictionary containing comparison metrics:
            - For each category: actual counts, synth counts, difference, error %
            - Overall fit statistics (RMSE, MAE, max error)
            
        Raises:
            RuntimeError: If generate() hasn't been called yet
        """
        import numpy as np
        
        if not self._is_generated:
            raise RuntimeError("Must call generate() before comparing to marginals")
        
        if not self._marginals:
            raise RuntimeError("No marginals stored - regenerate with current version")
        
        comparisons = {}
        
        # 1. Compare Sex Distribution
        comparisons['sex'] = self._compare_sex_distribution()
        
        # 2. Compare Age Distribution  
        comparisons['age'] = self._compare_age_distribution()
        
        # 3. Compare Household Role Distribution
        comparisons['role'] = self._compare_role_distribution()
        
        # 4. Compare Household Size Distribution
        comparisons['household_size'] = self._compare_household_size_distribution()
        
        # 5. Compare Housing Type Distribution
        comparisons['housing_type'] = self._compare_housing_type_distribution()
        
        # 6. Compare Education Level Distribution (replaces broken income standard)
        comparisons['education'] = self._compare_education_distribution()
        
        # 7. Compare Income Source Distribution
        comparisons['income_source'] = self._compare_income_source_distribution()
        
        # 8. Compare Median Income (informational - excluded from MAPE grade)
        comparisons['median_income'] = self._compare_median_income()
        
        # 9. Joint validation: HH type × children (informational)
        comparisons['hh_type_children'] = self._compare_hh_type_children()
        
        # 10. Joint validation: HH role × age × sex (informational)
        comparisons['joint_role_age_sex'] = self._compare_joint_role_age_sex()
        
        # Calculate overall fit statistics
        # Note: informational comparisons are excluded because they either
        # compare SEK amounts, or compare joint distributions whose univariate
        # marginals are already included in the MAPE.
        excluded_from_mape = {'overall', 'median_income',
                              'hh_type_children', 'joint_role_age_sex'}
        all_actual = []
        all_synth = []
        all_pct_errors = []  # Percentage errors for each category
        
        # ── Voas & Williamson (2001) per-dimension metrics ───────────
        # Reference: de Mooij et al. (2024) GenSynthPop, §4.3
        # Computed per contingency table (dimension), then summarised.
        from scipy import stats as sp_stats
        
        dim_metrics = {}  # dimension_key → {sae, tae, chi2, z2, ...}
        
        for cat, data in comparisons.items():
            if cat in excluded_from_mape:
                continue
            if data and 'comparison' in data:
                dim_actual = []
                dim_synth = []
                for row in data['comparison']:
                    if row.get('exclude_from_mape'):
                        continue
                    dim_actual.append(row['actual'])
                    dim_synth.append(row['synth'])
                    all_actual.append(row['actual'])
                    all_synth.append(row['synth'])
                    if row['actual'] > 0:
                        pct_err = abs(row['synth'] - row['actual']) / row['actual'] * 100
                        all_pct_errors.append(pct_err)
                
                if dim_actual:
                    ea = np.array(dim_actual, dtype=float)
                    oa = np.array(dim_synth, dtype=float)
                    da = oa - ea
                    
                    # TAE & SAE
                    tae_d = float(np.sum(np.abs(da)))
                    n_d = float(np.sum(ea))
                    sae_d = tae_d / n_d if n_d > 0 else 0.0
                    
                    # Pearson X² (Eq. 3) — 0-expected replaced with 1
                    e_safe = np.where(ea == 0, 1.0, ea)
                    chi2_d = float(np.sum(da ** 2 / e_safe))
                    
                    # Binomial Z² (Eq. 4)
                    sum_e = n_d if n_d > 0 else 1.0
                    sum_o = float(np.sum(oa)) if np.sum(oa) > 0 else 1.0
                    t = oa / sum_o
                    p = np.where(ea != 0, ea / sum_e, 1.0 / sum_e)
                    correction = 1.0 / (2.0 * sum_e)
                    corrected = np.maximum(np.abs(t - p) - correction, 0.0)
                    denom = np.sqrt(p * (1.0 - p) / sum_e)
                    denom_safe = np.where(denom == 0, 1.0, denom)
                    z2_d = float(np.sum((corrected / denom_safe) ** 2))
                    
                    dof_d = len(dim_actual)
                    chi2_p_d = float(1.0 - sp_stats.chi2.cdf(chi2_d, dof_d)) if dof_d > 0 else 1.0
                    z2_p_d = float(1.0 - sp_stats.chi2.cdf(z2_d, dof_d)) if dof_d > 0 else 1.0
                    
                    dim_metrics[cat] = {
                        'tae': tae_d,
                        'sae': sae_d,
                        'chi2': chi2_d,
                        'chi2_p': chi2_p_d,
                        'z2': z2_d,
                        'z2_p': z2_p_d,
                        'dof': dof_d,
                    }
                    # Attach to the comparison dict so it's available per-dimension
                    data['fit'] = dim_metrics[cat]
        
        if all_actual:
            actual_arr = np.array(all_actual, dtype=float)
            synth_arr = np.array(all_synth, dtype=float)
            diff_arr = synth_arr - actual_arr
            
            # MAPE — treats all categories equally regardless of size
            mape = float(np.mean(all_pct_errors)) if all_pct_errors else 0.0
            
            # Weighted MAPE (weighted by census count)
            weighted_pct_errors = []
            for actual, synth in zip(all_actual, all_synth):
                if actual > 0:
                    weighted_pct_errors.append(abs(synth - actual) / actual * 100 * actual)
            wmape = sum(weighted_pct_errors) / sum(all_actual) if sum(all_actual) > 0 else 0.0
            
            # Aggregate V&W metrics: median SAE across dimensions
            sae_values = [m['sae'] for m in dim_metrics.values()]
            chi2_p_values = [m['chi2_p'] for m in dim_metrics.values()]
            z2_p_values = [m['z2_p'] for m in dim_metrics.values()]
            
            comparisons['overall'] = {
                'total_actual': int(sum(all_actual)),
                'total_synth': int(sum(all_synth)),
                'rmse': float(np.sqrt(np.mean(diff_arr ** 2))),
                'mae': float(np.mean(np.abs(diff_arr))),
                'max_error': int(np.max(np.abs(diff_arr))),
                'correlation': float(np.corrcoef(actual_arr, synth_arr)[0, 1]) if len(actual_arr) > 1 else 1.0,
                'mape': mape,
                'wmape': wmape,
                'n_categories': len(all_actual),
                'max_pct_error': max(all_pct_errors) if all_pct_errors else 0.0,
                # Voas & Williamson (2001) — per-dimension then aggregated
                'sae_median': float(np.median(sae_values)) if sae_values else 0.0,
                'sae_max': float(np.max(sae_values)) if sae_values else 0.0,
                'sae_mean': float(np.mean(sae_values)) if sae_values else 0.0,
                'chi2_p_min': float(np.min(chi2_p_values)) if chi2_p_values else 1.0,
                'z2_p_min': float(np.min(z2_p_values)) if z2_p_values else 1.0,
                'dim_metrics': dim_metrics,  # Full per-dimension detail
            }
        
        if print_report:
            self._print_comparison_report(comparisons, use_logging=use_logging)
        
        return comparisons

    def log_statistics(self, include_marginal_comparison: bool = True) -> None:
        """
        Log summary statistics and optionally marginal comparison via logging.
        
        This is a convenience method for getting all statistics output through
        the logging system instead of print statements. Configure logging level
        to INFO or lower to see the output.
        
        Args:
            include_marginal_comparison: If True, also logs the full marginal
                                        comparison report
                                        
        Raises:
            RuntimeError: If generate() hasn't been called yet
        """
        if not self._is_generated:
            raise RuntimeError("Must call generate() before logging statistics")
        
        stats = self.get_summary_statistics()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"POPULATION SUMMARY: {self.area_name} ({self.year})")
        logger.info("=" * 60)
        logger.info(f"  Total Population:       {stats['total_population']:,}")
        logger.info(f"  Total Households:       {stats['total_households']:,}")
        logger.info(f"  Avg Household Size:     {stats['avg_household_size']:.2f}")
        logger.info(f"  Number of Children:     {stats['num_children']:,}")
        logger.info(f"  Number of Adults:       {stats['num_adults']:,}")
        logger.info(f"  Couple Households:      {stats['num_couples']:,}")
        logger.info(f"  Single Parent HH:       {stats['num_single_parent']:,}")
        logger.info(f"  Single Person HH:       {stats['num_single_person']:,}")
        logger.info(f"  Average Income:         {stats['avg_income']:,.0f} SEK")
        logger.info(f"  Total Cars:             {stats['total_cars']:,}")
        logger.info("")
        
        # Log synthesis statistics if available
        if self.stats:
            logger.info("Synthesis Statistics:")
            logger.info(f"  Method: {self.stats.get('method', 'unknown')}")
            if 'rmse' in self.stats:
                logger.info(f"  RMSE: {self.stats['rmse']:.4f}")
            if 'converged' in self.stats:
                logger.info(f"  Converged: {self.stats['converged']}")
            if 'iterations' in self.stats:
                logger.info(f"  Iterations: {self.stats['iterations']}")
            if 'households_created' in self.stats:
                logger.info(f"  Households: {self.stats['households_created']}")
            if 'individuals_placed' in self.stats:
                logger.info(f"  Individuals Placed: {self.stats['individuals_placed']}")
            logger.info("")
        
        if include_marginal_comparison:
            self.compare_to_marginals(print_report=True, use_logging=True)

    def _compare_sex_distribution(self) -> dict:
        """Compare sex distribution between marginals and synthesis."""
        pop_data = self._marginals.get('population')
        if pop_data is None or pop_data.empty:
            return {}
        
        # Get actual from marginals
        sex_col = 'Kön' if 'Kön' in pop_data.columns else 'sex'
        count_col = 'Antal' if 'Antal' in pop_data.columns else pop_data.columns[-1]
        
        if sex_col not in pop_data.columns:
            return {}
        
        actual = pop_data.groupby(sex_col)[count_col].sum().to_dict()
        
        # Get synth counts - match to actual API labels
        # API might use "Man"/"Kvinna" or "Män"/"Kvinnor"
        synth = {}
        for ind in self.individuals:
            # Try to match actual labels
            if 'Man' in actual or 'Kvinna' in actual:
                sex = 'Kvinna' if ind.sex == 'female' else 'Man'
            else:
                sex = 'Kvinnor' if ind.sex == 'female' else 'Män'
            synth[sex] = synth.get(sex, 0) + 1
        
        # Build comparison
        comparison = []
        for category in set(list(actual.keys()) + list(synth.keys())):
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1)
            })
        
        return {'name': 'Sex Distribution', 'comparison': comparison}
    
    def _compare_age_distribution(self) -> dict:
        """Compare age distribution between marginals and synthesis."""
        pop_data = self._marginals.get('population')
        if pop_data is None or pop_data.empty:
            return {}
        
        age_col = 'Ålder' if 'Ålder' in pop_data.columns else 'age_group'
        count_col = 'Antal' if 'Antal' in pop_data.columns else pop_data.columns[-1]
        
        if age_col not in pop_data.columns:
            return {}
        
        actual = pop_data.groupby(age_col)[count_col].sum().to_dict()
        
        # Dynamically parse age groups from actual data
        # API uses formats like "0-5 år", "16-18 år", "85- år"
        def parse_age_range(label):
            """Parse age range from label like '16-18 år' -> (16, 18)"""
            import re
            label = str(label).strip()
            # Handle "16-18 år" first (more specific pattern)
            match = re.match(r'(\d+)-(\d+)\s*år', label)
            if match:
                return (int(match.group(1)), int(match.group(2)))
            # Handle "85- år" or "85+ år" (open-ended, no digit after dash)
            match = re.match(r'(\d+)[-+]\s*år', label)
            if match:
                return (int(match.group(1)), 150)
            # Handle "85-w år" format
            match = re.match(r'(\d+)-\w\s*år', label)
            if match:
                return (int(match.group(1)), 150)
            return None
        
        # Build synth counts matching actual categories
        synth = {cat: 0 for cat in actual.keys()}
        for ind in self.individuals:
            for category in actual.keys():
                age_range = parse_age_range(category)
                if age_range and age_range[0] <= ind.age <= age_range[1]:
                    synth[category] = synth.get(category, 0) + 1
                    break
        
        comparison = []
        for category in actual.keys():
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1)
            })
        
        # Sort by numeric lower bound so output is in natural age order
        comparison.sort(key=lambda r: parse_age_range(r['category'])[0]
                        if parse_age_range(r['category']) else 999)
        
        return {'name': 'Age Distribution', 'comparison': comparison}
    
    def _compare_household_size_distribution(self) -> dict:
        """Compare household size distribution."""
        hh_data = self._marginals.get('household')
        if hh_data is None or hh_data.empty:
            return {}
        
        size_col = 'Hushållsstorlek' if 'Hushållsstorlek' in hh_data.columns else 'hh_size'
        count_col = 'Antal' if 'Antal' in hh_data.columns else hh_data.columns[-1]
        
        if size_col not in hh_data.columns:
            return {}
        
        actual = hh_data.groupby(size_col)[count_col].sum().to_dict()
        
        # Count synth household sizes
        synth = {}
        size_labels = {
            1: '1 person',
            2: '2 personer', 
            3: '3 personer',
            4: '4 personer',
            5: '5 personer',
            6: '6 eller fler personer'  # Match actual data label
        }
        for hh in self.households:
            size = min(hh.size, 6)
            label = size_labels.get(size, f'{size} personer')
            synth[label] = synth.get(label, 0) + 1
        
        comparison = []
        for category in actual.keys():
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1)
            })
        
        return {'name': 'Household Size Distribution', 'comparison': comparison}
    
    def _compare_housing_type_distribution(self) -> dict:
        """Compare housing type (Hustyp) distribution."""
        hh_data = self._marginals.get('household')
        if hh_data is None or hh_data.empty:
            return {}
        
        type_col = 'Hustyp' if 'Hustyp' in hh_data.columns else 'house_type'
        count_col = 'Antal' if 'Antal' in hh_data.columns else hh_data.columns[-1]
        
        if type_col not in hh_data.columns:
            return {}
        
        actual = hh_data.groupby(type_col)[count_col].sum().to_dict()
        
        # Count synth housing types
        synth = {}
        for hh in self.households:
            ht = hh.assigned_hustyp or 'Okänd'
            synth[ht] = synth.get(ht, 0) + 1
        
        comparison = []
        for category in set(list(actual.keys()) + list(synth.keys())):
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1)
            })
        
        return {'name': 'Housing Type Distribution', 'comparison': comparison}
    
    def _compare_role_distribution(self) -> dict:
        """Compare individual-level household position distribution.

        Uses the 60_FolkmHHStallning position data as ground truth and
        classifies each synthesised individual into 5 census-aligned
        categories:

            Sammanboende  (census Gift/reg.partner + Sambo merged)
            Ensam förälder
            Barn
            Ensamboende
            Övriga

        Gift and Sambo are merged because the synthesiser has no concept
        of marital status — it only assigns ``cohabiting``.  The 7-category
        breakdown (with Gift/Sambo split) is available in the informational
        ``joint_role_age_sex`` comparison.
        """
        pos_data = self._marginals.get('household_position')
        if pos_data is None or (hasattr(pos_data, 'empty') and pos_data.empty):
            return {}

        # ── Identify columns ─────────────────────────────────────────
        pos_col = None
        for col in pos_data.columns:
            if 'ställning' in col.lower() or 'position' in col.lower():
                pos_col = col
                break
        if pos_col is None:
            return {}

        count_col = 'Antal' if 'Antal' in pos_data.columns else pos_data.columns[-1]

        # ── Map census position → 5 collapsed categories ─────────────
        def collapsed_pos(pos_str: str) -> str:
            ps = str(pos_str).lower()
            if 'gift' in ps or 'partner' in ps or 'sambo' in ps:
                return 'Sammanboende'
            elif 'ensamstående förälder' in ps:
                return 'Ensam förälder'
            elif ps.startswith('barn'):
                return 'Barn'
            elif 'ensamboende' in ps:
                return 'Ensamboende'
            elif 'övriga' in ps or 'ej ensam' in ps:
                return 'Övriga'
            elif 'uppgift' in ps:
                return 'Uppgift saknas'
            return str(pos_str)[:16]

        # ── Census actual counts (aggregate over age × sex) ──────────
        actual: dict[str, int] = {}
        for _, row in pos_data.iterrows():
            pos = collapsed_pos(row[pos_col])
            val = int(row[count_col]) if pd.notna(row[count_col]) else 0
            actual[pos] = actual.get(pos, 0) + val

        if not actual:
            return {}

        # ── Synthesised counts ───────────────────────────────────────
        hh_by_id = {h.household_id: h for h in self.households}
        synth: dict[str, int] = {}

        for ind in self.individuals:
            role = ind.hh_role
            if role == 'child':
                pos = 'Barn'
            elif role == 'cohabiting':
                pos = 'Sammanboende'
            elif role == 'single_parent':
                pos = 'Ensam förälder'
            elif role == 'single':
                hh = hh_by_id.get(ind.household_id)
                if hh and len(hh.members) == 1:
                    pos = 'Ensamboende'
                elif hh and any(m.hh_role == 'child' for m in hh.members):
                    pos = 'Ensam förälder'
                else:
                    pos = 'Ensamboende'
            elif role == 'other':
                pos = 'Övriga'
            else:
                pos = 'Uppgift saknas'
            synth[pos] = synth.get(pos, 0) + 1

        # ── Build comparison rows ────────────────────────────────────
        comparison = []
        all_positions = sorted(set(list(actual.keys()) + list(synth.keys())))
        for pos in all_positions:
            if pos == 'Uppgift saknas':
                continue
            act_val = actual.get(pos, 0)
            syn_val = synth.get(pos, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': pos,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        return {'name': 'Household Position Distribution', 'comparison': comparison}
    
    def _compare_education_distribution(self) -> dict:
        """Compare education level distribution for adults 18+.
        
        Compares the synthesized education levels against census data from
        the 23_InkomsterUtbildning table. This replaces the broken income
        standard comparison which had a structurally unmatchable category.
        
        Education levels compared:
        - Förgymnasial utbildning (pre_secondary)
        - Gymnasial utbildning (secondary)
        - Eftergymnasial utbildning (post_secondary)
        - Uppgift saknas (unknown)
        
        Returns:
            Dict with 'name' and 'comparison' keys, or empty dict if no data.
        """
        edu_data = self._marginals.get('education_level')
        if edu_data is None or (hasattr(edu_data, 'empty') and edu_data.empty):
            return {}
        
        # Map internal education values back to Swedish names for display
        edu_display = {
            'pre_secondary': 'Förgymnasial utbildning',
            'secondary': 'Gymnasial utbildning',
            'post_secondary': 'Eftergymnasial utbildning',
            'unknown': 'Uppgift saknas',
        }
        
        # Census actual counts: sum across age groups and sexes per education level
        # Filter to Folkmängd rows only (the table also contains Medianinkomst etc.)
        actual = {}
        for _, row in edu_data.iterrows():
            # Only count population rows, not income statistics rows
            metric = row.get('Tabellvärde', 'Folkmängd')
            if metric != 'Folkmängd':
                continue
            edu_sv = row['Utbildningsnivå']
            count = int(row['Antal']) if pd.notna(row['Antal']) else 0
            actual[edu_sv] = actual.get(edu_sv, 0) + count
        
        # Synth counts: count adults by education level
        synth = {}
        for ind in self.individuals:
            if ind.age < 18:
                continue  # Only compare adults
            edu = getattr(ind, 'education', None)
            if edu and edu != 'child':
                display_name = edu_display.get(edu, edu)
                synth[display_name] = synth.get(display_name, 0) + 1
            else:
                synth['Uppgift saknas'] = synth.get('Uppgift saknas', 0) + 1
        
        # Build comparison
        comparison = []
        all_categories = sorted(set(list(actual.keys()) + list(synth.keys())))
        for category in all_categories:
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1)
            })
        
        return {'name': 'Education Level Distribution', 'comparison': comparison}

    def _compare_income_source_distribution(self) -> dict:
        """Compare primary income source distribution for adults 20+.
        
        Compares the synthesized income source categories against census data
        from the 20_HuvudInk table. This provides a comparison dimension for
        how well the synthetic population matches the real distribution of
        employment, pensions, studies, etc.
        
        Income source categories compared (9):
        - work, unemployment, studies, pension, disability,
          sickness, parental_leave, financial_support, no_income
        
        Returns:
            Dict with 'name' and 'comparison' keys, or empty dict if no data.
        """
        source_data = self._marginals.get('income_source')
        if source_data is None or (hasattr(source_data, 'empty') and source_data.empty):
            return {}
        
        # Map internal values to Swedish names for display
        source_display = {
            'work': 'Ersättning för arbete',
            'unemployment': 'Ersättning vid arbetslöshet',
            'studies': 'Ersättning för studier',
            'pension': 'Pension',
            'disability': 'Ersättning vid långvarigt nedsatt arbetsförmåga',
            'sickness': 'Ersättning vid sjukdom',
            'parental_leave': 'Ersättning vid föräldraledighet...',
            'financial_support': 'Ekonomiskt stöd',
            'no_income': 'Saknar ersättningar',
        }
        
        # Reverse map for census → internal key → display
        source_map = {
            'Ersättning för arbete': 'work',
            'Ersättning vid arbetslöshet': 'unemployment',
            'Ersättning för studier': 'studies',
            'Pension': 'pension',
            'Ersättning vid långvarigt nedsatt arbetsförmåga': 'disability',
            'Ersättning vid sjukdom': 'sickness',
            'Ersättning vid föräldraledighet eller närståendeomvårdnad': 'parental_leave',
            'Ekonomiskt stöd': 'financial_support',
            'Saknar ersättningar': 'no_income',
        }
        
        # Find column names
        source_col = None
        for col in source_data.columns:
            if 'inkomstkälla' in col.lower() or 'huvudsaklig' in col.lower():
                source_col = col
                break
        if source_col is None:
            return {}
        
        count_col = 'Antal' if 'Antal' in source_data.columns else source_data.columns[-1]
        
        # Census actual counts: sum across sexes per income source
        actual = {}
        for _, row in source_data.iterrows():
            src_sv = row[source_col]
            src_en = source_map.get(src_sv, src_sv)
            display = source_display.get(src_en, src_sv)
            count = int(row[count_col]) if pd.notna(row[count_col]) else 0
            actual[display] = actual.get(display, 0) + count
        
        # Synth counts: count adults 20+ by income source
        synth = {}
        for ind in self.individuals:
            if ind.age < 20:
                continue
            src = getattr(ind, 'income_source', None)
            if src:
                display = source_display.get(src, src)
                synth[display] = synth.get(display, 0) + 1
        
        # Build comparison
        comparison = []
        all_categories = sorted(set(list(actual.keys()) + list(synth.keys())))
        for category in all_categories:
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1)
            })
        
        return {'name': 'Income Source Distribution', 'comparison': comparison}

    def _compare_median_income(self) -> dict:
        """Compare median income (SEK) by education level × age group × sex.
        
        Uses Medianinkomst rows from the education table as census ground truth
        and computes median income from the synthesized population for the same
        groups.  Each row in the comparison represents one
        (education, age_group, sex) combination.
        
        This comparison is **informational** — it is excluded from the overall
        MAPE grade because it compares SEK amounts (thousands) rather than
        population counts, and mixing units would distort the aggregate metric.
        
        Returns:
            Dict with 'name' and 'comparison' keys, or empty dict if no data.
        """
        import statistics
        edu_data = self._marginals.get('education_level')
        if edu_data is None or (hasattr(edu_data, 'empty') and edu_data.empty):
            return {}
        
        if 'Tabellvärde' not in edu_data.columns:
            return {}
        
        median_rows = edu_data[edu_data['Tabellvärde'] == 'Medianinkomst']
        if median_rows.empty:
            return {}
        
        # Build census median income lookup
        edu_map = {
            'Förgymnasial utbildning': 'pre_secondary',
            'Gymnasial utbildning': 'secondary',
            'Eftergymnasial utbildning': 'post_secondary',
            'Uppgift saknas': 'unknown',
        }
        sex_map_sv = {'Man': 'male', 'Kvinna': 'female'}
        
        age_groups = [
            (18, 24, '18-24 år'),
            (25, 34, '25-34 år'),
            (35, 44, '35-44 år'),
            (45, 54, '45-54 år'),
            (55, 64, '55-64 år'),
            (65, 74, '65-74 år'),
            (75, 120, '75- år'),
        ]
        
        census_medians = {}  # (age_label, sex_en, edu_en) -> median_sek
        for _, row in median_rows.iterrows():
            sex_en = sex_map_sv.get(row.get('Kön', ''))
            edu_en = edu_map.get(row.get('Utbildningsnivå', ''))
            age_label = row.get('Ålder', '')
            val = row.get('Antal', 0)
            if sex_en and edu_en and age_label and pd.notna(val) and val > 0:
                census_medians[(age_label, sex_en, edu_en)] = float(val)
        
        if not census_medians:
            return {}
        
        # Group synthesized individuals into the same buckets
        synth_buckets: dict = {}  # same key -> list of incomes
        for ind in self.individuals:
            if ind.age < 18 or ind.income is None:
                continue
            
            # Find age group
            age_label = None
            for ag_min, ag_max, label in age_groups:
                if ag_min <= ind.age <= ag_max:
                    age_label = label
                    break
            if age_label is None:
                continue
            
            edu = getattr(ind, 'education', None) or 'unknown'
            key = (age_label, ind.sex, edu)
            if key not in synth_buckets:
                synth_buckets[key] = []
            synth_buckets[key].append(ind.income)
        
        # Build comparison rows — one per (education × age × sex) group
        edu_display = {
            'pre_secondary': 'Förgymnasial',
            'secondary': 'Gymnasial',
            'post_secondary': 'Eftergymnasial',
            'unknown': 'Uppgift saknas',
        }
        sex_display = {'male': 'M', 'female': 'K'}
        
        comparison = []
        for key in sorted(census_medians.keys()):
            age_label, sex_en, edu_en = key
            census_val = int(round(census_medians[key]))
            synth_incomes = synth_buckets.get(key, [])
            synth_val = int(round(statistics.median(synth_incomes))) if synth_incomes else 0
            diff = synth_val - census_val
            error_pct = (diff / census_val * 100) if census_val > 0 else 0
            
            cat = f"{edu_display.get(edu_en, edu_en)} {age_label} {sex_display.get(sex_en, sex_en)}"
            comparison.append({
                'category': cat,
                'actual': census_val,
                'synth': synth_val,
                'diff': diff,
                'error_pct': round(error_pct, 1)
            })
        
        return {'name': 'Median Income (SEK, informational)', 'comparison': comparison}

    def _compare_hh_type_children(self) -> dict:
        """Compare household type × number of children (0-17) joint distribution.
        
        Uses the 10_HHTypBarnU18_PRI.px table which cross-tabulates household
        type (Ensamstående, Sammanboende, Övriga) with child count (0–4+).
        
        This is an **informational** comparison — it is excluded from MAPE
        because its univariate marginals (HH type, child counts) overlap with
        existing MAPE dimensions.  The value is in revealing joint-distribution
        mismatches such as "too many single-parent households with 3 children".
        
        Returns:
            Dict with 'name' and 'comparison' keys, or empty dict.
        """
        hh_tc_data = self._marginals.get('hh_type_children')
        if hh_tc_data is None or (hasattr(hh_tc_data, 'empty') and hh_tc_data.empty):
            return {}
        
        # Identify columns
        type_col = None
        child_col = None
        for col in hh_tc_data.columns:
            cl = col.lower()
            if 'hushållstyp' in cl:
                type_col = col
            elif 'barn' in cl:
                child_col = col
        if type_col is None or child_col is None:
            return {}
        
        count_col = 'Antal' if 'Antal' in hh_tc_data.columns else hh_tc_data.columns[-1]
        
        # Census: cross-tab of type × children
        actual = {}
        for _, row in hh_tc_data.iterrows():
            ht = str(row[type_col]).strip()
            nc = str(row[child_col]).strip()
            val = int(row[count_col]) if pd.notna(row[count_col]) else 0
            key = f"{ht} | {nc}"
            actual[key] = actual.get(key, 0) + val
        
        # Synth: count children aged 0-17 per household, classify HH type
        hh_role_map = {
            'single': 'Ensamstående',
            'cohabiting': 'Sammanboende',
            'other': 'Övriga hushåll',
        }
        child_bins = ['0 barn', '1 barn', '2 barn', '3 barn', '4 barn eller fler']
        
        synth = {}
        for hh in self.households:
            # Determine HH type from members' roles
            roles = [m.hh_role for m in hh.members]
            if any(r == 'cohabiting' for r in roles):
                hh_type_sv = 'Sammanboende'
            elif any(r == 'other' for r in roles):
                hh_type_sv = 'Övriga hushåll'
            else:
                hh_type_sv = 'Ensamstående'
            
            # Count children 0-17
            n_children = sum(1 for m in hh.members if m.age <= 17)
            if n_children >= 4:
                nc_label = '4 barn eller fler'
            else:
                nc_label = child_bins[n_children]
            
            key = f"{hh_type_sv} | {nc_label}"
            synth[key] = synth.get(key, 0) + 1
        
        # Build comparison
        comparison = []
        all_keys = sorted(set(list(actual.keys()) + list(synth.keys())))
        for key in all_keys:
            act_val = actual.get(key, 0)
            syn_val = synth.get(key, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': key,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1)
            })
        
        return {'name': 'HH Type × Children 0-17 (informational)', 'comparison': comparison}

    def _compare_joint_role_age_sex(self) -> dict:
        """Compare the joint age × sex × household position distribution.
        
        Uses the already-stored household_position marginal from the
        60_FolkmHHStallning_PRI.px table which provides population counts by
        age group (11) × sex (2) × position (7).  The synthesized population is
        cross-tabulated in the same buckets and compared.
        
        This is **informational** — excluded from MAPE because the univariate
        age, sex, and role marginals are already MAPE dimensions.  The value is
        in revealing joint mismatches such as "too many 25-34 yr old female
        children" (adult still classified as child in synthesis).
        
        Returns:
            Dict with 'name' and 'comparison' keys, or empty dict.
        """
        pos_data = self._marginals.get('household_position')
        if pos_data is None or (hasattr(pos_data, 'empty') and pos_data.empty):
            return {}
        
        # Identify columns
        age_col = None
        sex_col = None
        pos_col = None
        for col in pos_data.columns:
            cl = col.lower()
            if 'ålder' in cl:
                age_col = col
            elif 'kön' in cl:
                sex_col = col
            elif 'ställning' in cl or 'position' in cl:
                pos_col = col
        if not all([age_col, sex_col, pos_col]):
            return {}
        
        count_col = 'Antal' if 'Antal' in pos_data.columns else pos_data.columns[-1]
        
        # Simplify position labels for display
        def short_pos(pos_str: str) -> str:
            ps = str(pos_str).lower()
            if 'gift' in ps or 'partner' in ps:
                return 'Gift/reg.partner'
            elif 'sambo' in ps:
                return 'Sambo'
            elif 'ensamstående förälder' in ps:
                return 'Ensam förälder'
            elif ps.startswith('barn'):
                return 'Barn'
            elif 'ensamboende' in ps:
                return 'Ensamboende'
            elif 'övriga' in ps or 'ej ensam' in ps:
                return 'Övriga'
            elif 'uppgift' in ps:
                return 'Uppgift saknas'
            return str(pos_str)[:16]
        
        # Map internal role → census position (for synth counting)
        # The census has 7 detailed positions; we map our simplified roles
        role_to_census_short = {
            'cohabiting': None,  # Split into Gift/Sambo — handle below
            'single': None,     # Could be Ensamboende or Ensam förälder
            'child': 'Barn',
            'other': 'Övriga',
        }
        
        # Parse age ranges for synth bucketing
        import re
        def parse_age_range(label):
            label = str(label).strip()
            m = re.match(r'(\d+)-(\d+)\s*år', label)
            if m:
                return (int(m.group(1)), int(m.group(2)))
            m = re.match(r'(\d+)[-+]\s*år', label)
            if m:
                return (int(m.group(1)), 200)
            return None
        
        # Census actual: aggregate into (age_group, sex, short_position) -> count
        # But there are 11 × 2 × 7 = 154 cells — too granular for a report.
        # Aggregate by (short_position) only, summing over age and sex, to show
        # the 7-category role distribution more precisely than the univariate
        # MAPE (which uses 3 collapsed categories: single/cohabiting/other).
        actual = {}
        age_groups_seen = set()
        for _, row in pos_data.iterrows():
            pos = short_pos(row[pos_col])
            val = int(row[count_col]) if pd.notna(row[count_col]) else 0
            actual[pos] = actual.get(pos, 0) + val
            age_groups_seen.add(str(row[age_col]))
        
        if not actual:
            return {}
        
        # Build household_id -> Household lookup for O(1) access
        hh_by_id = {h.household_id: h for h in self.households}
        
        # Synth: classify each individual into the 7-category scheme
        # Use census Gift/Sambo ratio to split cohabiting (same as role comparison)
        act_gift = actual.get('Gift/reg.partner', 0)
        act_sambo = actual.get('Sambo', 0)
        total_cohab_census = act_gift + act_sambo
        gift_frac = act_gift / total_cohab_census if total_cohab_census > 0 else 0.5

        cohab_ids = [ind.agent_id for ind in self.individuals if ind.hh_role == 'cohabiting']
        n_gift = round(len(cohab_ids) * gift_frac)
        gift_set = set(sorted(cohab_ids)[:n_gift])

        synth = {}
        for ind in self.individuals:
            role = ind.hh_role
            if role == 'child':
                pos = 'Barn'
            elif role == 'cohabiting':
                pos = 'Gift/reg.partner' if ind.agent_id in gift_set else 'Sambo'
            elif role == 'single_parent':
                pos = 'Ensam förälder'
            elif role == 'single':
                hh = hh_by_id.get(ind.household_id)
                if hh and len(hh.members) == 1:
                    pos = 'Ensamboende'
                elif hh and any(m.hh_role == 'child' for m in hh.members):
                    pos = 'Ensam förälder'
                else:
                    pos = 'Ensamboende'
            elif role == 'other':
                pos = 'Övriga'
            else:
                pos = 'Uppgift saknas'
            synth[pos] = synth.get(pos, 0) + 1
        
        # Build comparison
        comparison = []
        all_positions = sorted(set(list(actual.keys()) + list(synth.keys())))
        for pos in all_positions:
            if pos == 'Uppgift saknas':
                continue  # Skip unknown/missing
            act_val = actual.get(pos, 0)
            syn_val = synth.get(pos, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': pos,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1)
            })
        
        return {
            'name': 'Detailed HH Position (7-cat, informational)',
            'comparison': comparison
        }

    def _compare_income_distribution(self) -> dict:
        """Compare income standard distribution (low vs not low income)."""
        income_data = self._marginals.get('income')
        if income_data is None or income_data.empty:
            return {}
        
        # Find income category column
        income_col = None
        for col in ['Inkomststandard', 'Inkomst', 'income']:
            if col in income_data.columns:
                income_col = col
                break
        
        if income_col is None:
            return {}
        
        # The income data uses "Inkomststandard" with categories:
        # - "Ingår i helårshushåll som inte har låg inkomststandard" (not low)
        # - "Ingår i helårshushåll som har låg inkomststandard" (low)
        # - "Ingår ej i helårshushåll" (not in year-round household)
        
        count_col = None
        for col in income_data.columns:
            if income_data[col].dtype in ['int64', 'float64'] and col not in ['År']:
                count_col = col
                break
        
        if count_col is None:
            count_col = income_data.columns[-1]
        
        actual = income_data.groupby(income_col)[count_col].sum().to_dict()
        
        # For synthesized population, we use income_decile
        # Map deciles to low/not-low (deciles 1-2 = low income approximation)
        synth = {}
        
        # Match synth to actual categories
        low_income_key = None
        not_low_key = None
        other_key = None
        
        for cat in actual.keys():
            cat_lower = str(cat).lower()
            if 'inte har låg' in cat_lower or 'not low' in cat_lower:
                not_low_key = cat
            elif 'har låg' in cat_lower or 'low' in cat_lower:
                low_income_key = cat
            elif 'ej i' in cat_lower or 'not in' in cat_lower:
                other_key = cat
        
        # Count synth - use income_standard attribute if available
        for ind in self.individuals:
            if hasattr(ind, 'income_standard') and ind.income_standard:
                if ind.income_standard == 'low':
                    key = low_income_key or 'Low income'
                else:
                    key = not_low_key or 'Not low income'
                synth[key] = synth.get(key, 0) + 1
            elif hasattr(ind, 'income_decile') and ind.income_decile:
                if ind.income_decile <= 2:
                    key = low_income_key or 'Low income'
                else:
                    key = not_low_key or 'Not low income'
                synth[key] = synth.get(key, 0) + 1
            else:
                # Fallback - assign to not low income
                key = not_low_key or 'Not low income'
                synth[key] = synth.get(key, 0) + 1
        
        comparison = []
        for category in set(list(actual.keys()) + list(synth.keys())):
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1)
            })
        
        return {'name': 'Income Standard Distribution', 'comparison': comparison}
    
    def _print_comparison_report(self, comparisons: dict, use_logging: bool = False) -> None:
        """Print a formatted comparison report.
        
        Args:
            comparisons: Dictionary of comparison data from compare_to_marginals
            use_logging: If True, output via logging.info instead of print
        """
        output = print if not use_logging else lambda msg: logger.info(msg)
        
        output("\n" + "=" * 70)
        output(f"MARGINAL COMPARISON REPORT: {self.area_name} ({self.year})")
        output("=" * 70)
        
        for key, data in comparisons.items():
            if key == 'overall':
                continue
            if not data or 'comparison' not in data:
                continue
                
            output(f"\n{data['name']}")
            output("-" * 50)
            output(f"{'Category':<20} {'Actual':>10} {'Synth':>10} {'Diff':>8} {'Error':>8}")
            output("-" * 50)
            
            for row in data['comparison']:
                cat = row['category'][:18] if len(row['category']) > 18 else row['category']
                excluded = ' *' if row.get('exclude_from_mape') else ''
                output(f"{cat:<20} {row['actual']:>10} {row['synth']:>10} {row['diff']:>+8} {row['error_pct']:>7.1f}%{excluded}")
            
            # Subtotals / averages
            total_actual = sum(r['actual'] for r in data['comparison'])
            total_synth = sum(r['synth'] for r in data['comparison'])
            n_rows = len(data['comparison'])
            output("-" * 50)
            if key == 'median_income' and n_rows > 0:
                avg_a = total_actual // n_rows
                avg_s = total_synth // n_rows
                avg_d = avg_s - avg_a
                output(f"{'AVERAGE':<20} {avg_a:>10} {avg_s:>10} {avg_d:>+8}")
            else:
                total_diff = total_synth - total_actual
                output(f"{'TOTAL':<20} {total_actual:>10} {total_synth:>10} {total_diff:>+8}")
            
            # Mark informational sections
            if key in ('median_income', 'hh_type_children', 'joint_role_age_sex'):
                output("  (informational — excluded from MAPE grade)")
            elif data.get('fit'):
                f = data['fit']
                output(f"  SAE={f['sae']:.4f}  X²={f['chi2']:.2f}(p={f['chi2_p']:.4f})  Z²={f['z2']:.2f}(p={f['z2_p']:.4f})")
        
        # Overall statistics
        if 'overall' in comparisons:
            ov = comparisons['overall']
            output("\n" + "=" * 70)
            output("OVERALL FIT STATISTICS")
            output("=" * 70)
            output(f"  Total Population (Actual):  {ov['total_actual']:,}")
            output(f"  Total Population (Synth):   {ov['total_synth']:,}")
            output(f"  Categories Compared:        {ov.get('n_categories', 'N/A')}")
            output(f"  Root Mean Square Error:     {ov['rmse']:.2f}")
            output(f"  Mean Absolute Error:        {ov['mae']:.2f}")
            output(f"  Max Category Error:         {ov['max_error']}")
            output(f"  Pearson Correlation:        {ov['correlation']:.4f}")
            output("")
            output("  Percentage Error Metrics:")
            output(f"    MAPE (unweighted):        {ov.get('mape', 0):.1f}%  ← treats all categories equally")
            output(f"    Weighted MAPE:            {ov.get('wmape', 0):.1f}%  ← weighted by category size")
            output(f"    Max Category Error:       {ov.get('max_pct_error', 0):.1f}%")
            output("")
            output("  Voas & Williamson (2001) — per-dimension then aggregated:")
            output(f"    SAE median:               {ov.get('sae_median', 0):.4f}  ← lower is better")
            output(f"    SAE mean:                 {ov.get('sae_mean', 0):.4f}")
            output(f"    SAE max:                  {ov.get('sae_max', 0):.4f}")
            output(f"    X² p-value (worst dim):   {ov.get('chi2_p_min', 0):.4f}  ← higher is better")
            output(f"    Z² p-value (worst dim):   {ov.get('z2_p_min', 0):.4f}")
            if ov.get('dim_metrics'):
                output("")
                output(f"    {'Dimension':<20} {'SAE':>8} {'X²':>10} {'X² p':>8} {'Z²':>10} {'Z² p':>8}")
                output("    " + "-" * 64)
                for dim_key, dm in ov['dim_metrics'].items():
                    output(f"    {dim_key:<20} {dm['sae']:>8.4f} {dm['chi2']:>10.2f} {dm['chi2_p']:>8.4f} {dm['z2']:>10.2f} {dm['z2_p']:>8.4f}")
        
        output("=" * 70 + "\n")
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get the marginal comparison as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: dimension, category, actual, synth, diff, error_pct
            
        Raises:
            RuntimeError: If generate() hasn't been called yet
        """
        comparisons = self.compare_to_marginals(print_report=False)
        
        rows = []
        for dim_key, data in comparisons.items():
            if dim_key == 'overall' or not data or 'comparison' not in data:
                continue
            for row in data['comparison']:
                rows.append({
                    'dimension': data['name'],
                    'category': row['category'],
                    'actual': row['actual'],
                    'synth': row['synth'],
                    'diff': row['diff'],
                    'error_pct': row['error_pct']
                })
        
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Prognosis scaling
    # ------------------------------------------------------------------

    def scale_to_year(
        self,
        target_year: int,
        base_year: int = 2025,
        allocate_dwellings: bool = True,
    ) -> 'GbgArea':
        """
        Generate a future-year population scaled by official prognosis data.

        This fetches the population prognosis from the Gothenburg statistics
        API at the mellanområde (intermediate area) level, computes per-age-group
        scale factors between ``base_year`` and ``target_year``, then applies
        those factors to this area's census marginals and re-synthesises.

        The prognosis is available for years 2025–2032 (spring 2025 forecast).

        Args:
            target_year: Future year to project to (2025–2032)
            base_year: Reference year in the prognosis (default 2025)
            allocate_dwellings: Whether to allocate dwellings (default True)

        Returns:
            A **new** ``GbgArea`` instance with the scaled population.
            The original area is not modified.

        Example:
            >>> city = GbgSynth(year=2024)
            >>> haga = city.synthesize("Haga")
            >>> haga_2030 = haga.scale_to_year(2030)
            >>> print(len(haga_2030.individuals))
        """
        logger.info(
            f"Scaling {self.area_name} from prognosis "
            f"{base_year}→{target_year}"
        )

        # 1. Fetch original census marginals
        population_data = self._fetch_population_data()
        household_data = self._fetch_household_data()

        # 2. Get scale factors from prognosis
        scaler = PrognosisScaler(
            base_year=base_year,
            target_year=target_year,
        )
        scaled_pop, scaled_hh = scaler.scale_marginals(
            self.area_code, population_data, household_data
        )

        # Log the scaling summary
        summary = scaler.summary(self.area_code)
        logger.info(
            f"Prognosis scaling for {self.area_name}: "
            f"{summary['mel_name']} ({summary['base_population']}→"
            f"{summary['target_population']}, {summary['overall_growth']})"
        )

        # 3. Create a new area instance for the future year
        future_area = GbgArea(
            area_code=self.area_code,
            area_name=self.area_name,
            year=target_year,
            client=self.client,
            config=self.config,
            area_api_value=self.area_api_value,
        )

        # 4. Fetch supplementary data using original year (census data)
        household_position_data = self._fetch_household_position_data()
        income_data = self._fetch_income_data()
        car_data = self._fetch_car_data()

        # 4b. Scale position data with the same single-year prognosis
        #     factors. The position data has the same Ålder/Antal columns
        #     as the population data but with finer role breakdowns.
        #     Without scaling it, the synthesizer would generate the
        #     base-year headcount regardless of the scaled pop marginals.
        base_df, target_df = scaler.get_prognosis(self.area_code)
        if household_position_data is not None:
            household_position_data = scale_population_marginals(
                household_position_data, base_df, target_df
            )

        # 5. Store marginals for validation
        future_area._marginals = {
            'population': scaled_pop.copy(),
            'household': scaled_hh.copy(),
            'household_position': (
                household_position_data.copy()
                if household_position_data is not None else None
            ),
            'income': income_data.copy() if income_data is not None else None,
        }

        # 6. Synthesise with scaled marginals
        synthesizer = PopulationSynthesizer(future_area.config)
        future_area.individuals, future_area.households = synthesizer.synthesize(
            population_data=scaled_pop,
            household_data=scaled_hh,
            income_data=income_data,
            car_data=car_data,
            household_position_data=household_position_data,
        )

        future_area.stats = synthesizer.stats
        future_area.stats['prognosis'] = summary
        future_area._is_generated = True

        logger.info(
            f"Future synthesis complete for {self.area_name} ({target_year}): "
            f"{len(future_area.individuals)} individuals, "
            f"{len(future_area.households)} households"
        )

        # 7. Allocate dwellings if requested
        if allocate_dwellings:
            future_area._allocate_dwellings()

        return future_area

    def get_prognosis_summary(
        self,
        base_year: int = 2025,
        target_year: int = 2030,
    ) -> dict:
        """
        Preview the prognosis scaling factors without generating.

        Args:
            base_year: Reference year (default 2025)
            target_year: Target year (default 2030)

        Returns:
            Dict with mel area info, population totals, scale factors
        """
        scaler = PrognosisScaler(
            base_year=base_year,
            target_year=target_year,
        )
        return scaler.summary(self.area_code)
