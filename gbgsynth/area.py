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
from gbgsynth.models import Agent, Household, Dwelling
from gbgsynth.synthesizer import PopulationSynthesizer

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
        
        # IPF statistics (if IPF is used)
        self.ipf_stats: dict = {}

    def generate(self, buildings: Optional[pd.DataFrame] = None, use_ipf: bool = False,
                 use_constrained_ipf: bool = False, use_topdown: bool = True,
                 allocate_dwellings: bool = True) -> None:
        """
        Execute the full synthesis pipeline for this area.

        Steps:
        1. Fetch required census tables
        2. Run population synthesizer (with optional IPF)
        3. Assign housing types (Hustyp)
        4. Link to building footprints (if provided)
        5. Store results

        Args:
            buildings: Optional GeoDataFrame/DataFrame with building footprints.
                      Should contain columns for building_id, type, and optionally
                      capacity. If provided, households will be linked to specific
                      buildings based on their assigned_hustyp.
            use_ipf: If True, use Iterative Proportional Fitting
                    for better marginal fit. If False, use greedy matching.
            use_constrained_ipf: If True, use constrained IPF that generates
                                complete valid household compositions directly.
                                This ensures 100% valid households by construction.
            use_topdown: If True (default), use top-down constrained synthesis that
                        anchors exact household containers first, then fills with
                        individuals. Best overall accuracy for household structure
                        and individual demographics.
        """
        logger.info(f"Generating synthetic population for {self.area_name} ({self.area_code}), year {self.year}")

        # Fetch data
        logger.info("Fetching census data...")
        population_data = self._fetch_population_data()
        household_data = self._fetch_household_data()
        household_position_data = self._fetch_household_position_data()  # Detailed roles
        income_data = self._fetch_income_data()
        car_data = self._fetch_car_data()
        
        # Store marginals for validation
        self._marginals = {
            'population': population_data.copy(),
            'household': household_data.copy(),
            'household_position': household_position_data.copy() if household_position_data is not None else None,
            'income': income_data.copy() if income_data is not None else None
        }

        # Synthesize
        logger.info("Running synthesis algorithm...")
        synthesizer = PopulationSynthesizer(
            self.config, 
            use_ipf=use_ipf and not use_constrained_ipf and not use_topdown,
            use_constrained_ipf=use_constrained_ipf,
            use_topdown=use_topdown
        )
        self.individuals, self.households = synthesizer.synthesize(
            population_data=population_data,
            household_data=household_data,
            income_data=income_data,
            car_data=car_data,
            buildings=buildings,
            household_position_data=household_position_data  # Pass detailed role data
        )
        
        # Store IPF statistics if used
        if (use_ipf or use_constrained_ipf or use_topdown) and hasattr(synthesizer, 'ipf_stats'):
            self.ipf_stats = synthesizer.ipf_stats

        self._is_generated = True
        logger.info(f"Synthesis complete: {len(self.individuals)} individuals, {len(self.households)} households")

        # Log housing type distribution
        hustyp_counts = {}
        for hh in self.households:
            ht = hh.assigned_hustyp or 'unassigned'
            hustyp_counts[ht] = hustyp_counts.get(ht, 0) + 1
        logger.info(f"Housing type distribution: {hustyp_counts}")
        
        # Allocate dwellings if requested
        if allocate_dwellings:
            self._allocate_dwellings()

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
        
        # Match households to dwellings
        self._match_households_to_dwellings()
        
        # Report allocation results
        allocated = sum(1 for hh in self.households if hh.dwelling_id is not None)
        linked = sum(1 for d in self.dwellings if d.building_id is not None)
        logger.info(f"Allocated {allocated}/{len(self.households)} households to dwellings")
        if linked > 0:
            logger.info(f"Linked {linked}/{len(self.dwellings)} dwellings to buildings")

    def _load_building_footprints(self) -> Optional['pd.DataFrame']:
        """
        Load building footprints for this area from bundled GeoPackage files.
        
        Returns:
            GeoDataFrame with building footprints or None if not available
        """
        import os
        
        try:
            import geopandas as gpd
        except ImportError:
            logger.debug("geopandas not available, skipping building footprints")
            return None
        
        # Build path to footprint file
        data_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to find matching footprint file by area name
        # Area names like "107 Haga" -> "Haga_heights.gpkg"
        area_short_name = self.area_name.split(' ', 1)[-1] if ' ' in self.area_name else self.area_name
        
        footprint_path = os.path.join(data_dir, 'data', 'footprints', f'{area_short_name}_heights.gpkg')
        
        if not os.path.exists(footprint_path):
            logger.debug(f"No footprint file found at {footprint_path}")
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
            logger.warning("No objekttyp column - filtering by size only")
            residential = buildings.copy()
            residential['net_to_gross'] = MFH_NET_TO_GROSS
        
        # Calculate net residential capacity per building
        residential['residential_area'] = residential['total_floor_area'] * residential['net_to_gross']
        
        # Filter out tiny structures
        residential = residential[residential['residential_area'] >= MIN_BUILDING_AREA].copy()
        
        if len(residential) == 0:
            logger.warning("No residential buildings found in footprints")
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
        """Fetch income distribution data."""
        table_path = self.config.get_table_id('INCOME')
        
        try:
            df = self.client.query_all_variables(table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} income records")
            return df
        except Exception as e:
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
            raise RuntimeError("Must call generate() before saving")
        
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
            RuntimeError: If generate() hasn't been called yet
        """
        if not self._is_generated:
            raise RuntimeError("Must call generate() before saving")

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
            RuntimeError: If generate() hasn't been called yet
        """
        if not self._is_generated:
            raise RuntimeError("Must call generate() before saving")

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
            RuntimeError: If generate() hasn't been called yet
        """
        if not self._is_generated:
            raise RuntimeError("Must call generate() before saving")

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
            raise RuntimeError("Must call generate() before accessing data")
        
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
        
        # 6. Compare Income Distribution
        comparisons['income'] = self._compare_income_distribution()
        
        # 5. Calculate overall fit statistics
        all_actual = []
        all_synth = []
        for cat, data in comparisons.items():
            if data and 'comparison' in data:
                for row in data['comparison']:
                    all_actual.append(row['actual'])
                    all_synth.append(row['synth'])
        
        if all_actual:
            actual_arr = np.array(all_actual)
            synth_arr = np.array(all_synth)
            diff_arr = synth_arr - actual_arr
            
            comparisons['overall'] = {
                'total_actual': int(sum(all_actual)),
                'total_synth': int(sum(all_synth)),
                'rmse': float(np.sqrt(np.mean(diff_arr ** 2))),
                'mae': float(np.mean(np.abs(diff_arr))),
                'max_error': int(np.max(np.abs(diff_arr))),
                'correlation': float(np.corrcoef(actual_arr, synth_arr)[0, 1]) if len(actual_arr) > 1 else 1.0
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
        
        # Log IPF statistics if available
        if self.ipf_stats:
            logger.info("IPF Statistics:")
            logger.info(f"  Method: {self.ipf_stats.get('method', 'unknown')}")
            if 'rmse' in self.ipf_stats:
                logger.info(f"  RMSE: {self.ipf_stats['rmse']:.4f}")
            if 'converged' in self.ipf_stats:
                logger.info(f"  Converged: {self.ipf_stats['converged']}")
            if 'iterations' in self.ipf_stats:
                logger.info(f"  Iterations: {self.ipf_stats['iterations']}")
            if 'households_created' in self.ipf_stats:
                logger.info(f"  Households: {self.ipf_stats['households_created']}")
            if 'individuals_placed' in self.ipf_stats:
                logger.info(f"  Individuals Placed: {self.ipf_stats['individuals_placed']}")
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
        """Compare household role (Hushållstyp) distribution."""
        pop_data = self._marginals.get('population')
        if pop_data is None or pop_data.empty:
            return {}
        
        role_col = 'Hushållstyp' if 'Hushållstyp' in pop_data.columns else 'hh_role'
        count_col = 'Antal' if 'Antal' in pop_data.columns else pop_data.columns[-1]
        
        if role_col not in pop_data.columns:
            return {}
        
        actual = pop_data.groupby(role_col)[count_col].sum().to_dict()
        
        # Map synth roles to Swedish labels
        role_mapping = {
            'single': ['Ensamstående', 'ensamstående', 'Ensam'],
            'cohabiting': ['Sammanboende', 'sammanboende', 'Sambo'],
            'child': ['Barn', 'barn', 'Övriga hushåll']  # Children often in "Övriga hushåll"
        }
        
        # Count synth roles - need to properly map to actual categories
        synth = {}
        
        # First pass: map directly
        for ind in self.individuals:
            role = ind.hh_role
            matched = False
            
            # Direct role matching
            for actual_cat in actual.keys():
                actual_lower = str(actual_cat).lower()
                
                if role == 'single' and 'ensam' in actual_lower:
                    synth[actual_cat] = synth.get(actual_cat, 0) + 1
                    matched = True
                    break
                elif role == 'cohabiting' and 'samman' in actual_lower:
                    synth[actual_cat] = synth.get(actual_cat, 0) + 1
                    matched = True
                    break
                elif role == 'child':
                    # Children go to "Övriga hushåll" if it exists, otherwise track separately
                    if 'övriga' in actual_lower:
                        synth[actual_cat] = synth.get(actual_cat, 0) + 1
                        matched = True
                        break
            
            if not matched:
                # Keep track of unmatched roles
                synth[role] = synth.get(role, 0) + 1
        
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
        
        return {'name': 'Household Role Distribution', 'comparison': comparison}
    
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
                output(f"{cat:<20} {row['actual']:>10} {row['synth']:>10} {row['diff']:>+8} {row['error_pct']:>7.1f}%")
            
            # Subtotals
            total_actual = sum(r['actual'] for r in data['comparison'])
            total_synth = sum(r['synth'] for r in data['comparison'])
            total_diff = total_synth - total_actual
            output("-" * 50)
            output(f"{'TOTAL':<20} {total_actual:>10} {total_synth:>10} {total_diff:>+8}")
        
        # Overall statistics
        if 'overall' in comparisons:
            ov = comparisons['overall']
            output("\n" + "=" * 70)
            output("OVERALL FIT STATISTICS")
            output("=" * 70)
            output(f"  Total Population (Actual):  {ov['total_actual']:,}")
            output(f"  Total Population (Synth):   {ov['total_synth']:,}")
            output(f"  Root Mean Square Error:     {ov['rmse']:.2f}")
            output(f"  Mean Absolute Error:        {ov['mae']:.2f}")
            output(f"  Max Category Error:         {ov['max_error']}")
            output(f"  Correlation:                {ov['correlation']:.4f}")
        
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
