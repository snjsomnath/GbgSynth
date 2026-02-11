"""
Main GbgSynth interface - user-facing API.

This module provides the high-level GbgSynth class for easy interaction
with the library.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from gbgsynth.api_client import PxWebClient
from gbgsynth.config import Config
from gbgsynth.area import GbgArea
from gbgsynth.exceptions import AreaNotFoundError
from gbgsynth.data_utils import (
    ensure_shapefile_available,
    ensure_areas_json_available,
    get_shapefile_path,
)

logger = logging.getLogger(__name__)

# Path to bundled data
_DATA_DIR = Path(__file__).parent / "data"
_AREAS_JSON = _DATA_DIR / "areas.json"
_SHAPEFILE = _DATA_DIR / "pri_shp" / "pri.shp"


def _load_area_registry() -> Dict[str, dict]:
    """Load the bundled area registry, generating it if necessary."""
    # Ensure areas.json exists (will generate from shapefile if needed)
    if not _AREAS_JSON.exists():
        if not ensure_areas_json_available():
            raise FileNotFoundError(
                "areas.json not found and could not be generated. "
                "Ensure geopandas is installed or download the shapefile manually."
            )
    
    with open(_AREAS_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


# Module-level cache for area registry
_AREA_REGISTRY: Optional[Dict[str, dict]] = None


def get_area_registry() -> Dict[str, dict]:
    """
    Get the area registry (cached).
    
    Returns:
        Dictionary mapping area codes to info dicts with 'name' and 'full' keys.
    """
    global _AREA_REGISTRY
    if _AREA_REGISTRY is None:
        _AREA_REGISTRY = _load_area_registry()
    return _AREA_REGISTRY


class GbgSynth:
    """
    Main interface for synthetic population generation in Gothenburg.

    This class provides a simple API for discovering areas and
    generating synthetic populations.
    
    Area Identification:
        All methods support flexible area matching. You can use:
        - Area code: "107"
        - Area name: "Haga" (case-insensitive)
        - Full name: "107 Haga"
        
        Examples: "Haga", "haga", "107", "107 Haga" all refer to the same area.

    Quick Start:
        >>> city = GbgSynth(year=2024)
        >>> haga = city.synthesize("Haga")  # One-liner synthesis
        >>> haga.save()  # Saves individuals.csv and households.csv
        
    Two-Step Workflow:
        >>> city = GbgSynth(year=2024)
        >>> haga = city.get_area("Haga")  # Get area without generating
        >>> haga.generate()  # Generate when ready
        
    Discovery:
        >>> GbgSynth.list_areas()  # All area names
        >>> GbgSynth.get_area_code("Haga")  # Look up code: '107'
    """

    # Class-level area registry (no API call needed)
    areas = property(lambda self: get_area_registry())

    def __init__(self, year: int = 2024, log_level: str = 'INFO'):
        """
        Initialize GbgSynth.

        Args:
            year: Year to generate population for (default: 2024)
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.year = year
        self._client: Optional[PxWebClient] = None
        self.config = Config()

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        logger.info(f"Initialized GbgSynth for year {year}")

    def __repr__(self) -> str:
        """Return string representation."""
        num_areas = len(get_area_registry())
        return f"GbgSynth(year={self.year}, {num_areas} areas available)"

    @property
    def client(self) -> PxWebClient:
        """Lazy-load API client only when needed."""
        if self._client is None:
            self._client = PxWebClient()
        return self._client

    @staticmethod
    def list_areas() -> List[str]:
        """
        List all neighbourhood names (no API call).
        
        Returns:
            List of area names like ['Haga', 'Annedal', ...]
        """
        registry = get_area_registry()
        return [info['name'] for info in registry.values()]

    @staticmethod
    def get_area_code(name: str) -> Optional[str]:
        """
        Look up area code by name (no API call).
        
        Supports flexible matching - all of these work:
            - Area code: "107"
            - Area name: "Haga" (case-insensitive)
            - Full name: "107 Haga"
            - Partial match: "haga" or "HAGA"
        
        Args:
            name: Area identifier in any of the formats above
            
        Returns:
            Area code like '107', or None if not found
            
        Example:
            >>> GbgSynth.get_area_code("Haga")
            '107'
            >>> GbgSynth.get_area_code("107")
            '107'
            >>> GbgSynth.get_area_code("haga")  # case-insensitive
            '107'
        """
        registry = get_area_registry()
        name_lower = name.lower().strip()
        
        # Direct code lookup
        if name_lower in registry:
            return name_lower
            
        # Try parsing "107 Haga" format
        parts = name.split(maxsplit=1)
        if parts[0] in registry:
            return parts[0]
        
        # Search by name
        for code, info in registry.items():
            if name_lower == info['name'].lower():
                return code
            if name_lower in info['full'].lower():
                return code
        
        return None

    @staticmethod
    def get_boundary(area_code: str):
        """
        Get the geographic boundary for an area.
        
        Args:
            area_code: Area code like '107'
            
        Returns:
            Shapely geometry object, or None if geopandas not installed
            
        Raises:
            ValueError: If area code not found
        """
        try:
            import geopandas as gpd
        except ImportError:
            logger.warning("geopandas not installed - cannot load boundaries")
            return None
        
        # Ensure shapefile is available (download if needed)
        if not ensure_shapefile_available():
            logger.warning("Shapefile not available and could not be downloaded")
            return None
        
        shapefile_path = get_shapefile_path()
        if shapefile_path is None:
            # This should not happen if ensure_shapefile_available returned True
            logger.error("Shapefile path is None after successful availability check")
            return None
        
        gdf = gpd.read_file(shapefile_path)
        # Match by PRIMÄROMRÅ column
        match = gdf[gdf['PRIMÄROMRÅ'].astype(str) == str(area_code)]
        
        if len(match) == 0:
            raise ValueError(f"Area code '{area_code}' not found in shapefile")
        
        return match.iloc[0].geometry

    @staticmethod
    def get_all_boundaries():
        """
        Get all neighbourhood boundaries as a GeoDataFrame.
        
        Returns:
            GeoDataFrame with area codes and geometries, or None if geopandas not installed
        """
        try:
            import geopandas as gpd
        except ImportError:
            logger.warning("geopandas not installed - cannot load boundaries")
            return None
        
        # Ensure shapefile is available (download if needed)
        if not ensure_shapefile_available():
            logger.warning("Shapefile not available and could not be downloaded")
            return None
        
        shapefile_path = get_shapefile_path()
        if shapefile_path is None:
            # This should not happen if ensure_shapefile_available returned True
            logger.error("Shapefile path is None after successful availability check")
            return None
        
        gdf = gpd.read_file(shapefile_path)
        # Rename columns for easier use
        gdf = gdf.rename(columns={
            'PRIMÄROMRÅ': 'area_code',
            'PRIMÄRNAMN': 'area_name'
        })
        return gdf[['area_code', 'area_name', 'geometry']]

    def synthesize(self, area: str, **kwargs) -> GbgArea:
        """
        One-liner synthesis: generate population for an area.
        
        This is a convenience method that combines get_area() and generate().
        Equivalent to: city.get_area(area).generate(**kwargs)
        
        Supports flexible area matching - all of these work:
            - Area code: "107"
            - Area name: "Haga" (case-insensitive)
            - Full name: "107 Haga"
        
        Args:
            area: Area identifier in any of the formats above
            **kwargs: Passed to generate().  Notable options:
                - ``engine`` (str): ``'topdown'`` (default), ``'ipf'``,
                  or ``'constrained_ipf'``.
                - ``buildings`` (DataFrame): Building footprints.
                - ``allocate_dwellings`` (bool): Default ``True``.
            
        Returns:
            GbgArea object with generated population
            
        Raises:
            AreaNotFoundError: If area cannot be found
            
        Example:
            >>> city = GbgSynth(2024)
            >>> haga = city.synthesize("Haga")
            >>> haga_ipf = city.synthesize("Haga", engine="ipf")
            
            # Flexible area matching:
            >>> city.synthesize("107")
            >>> city.synthesize("Haga")
            >>> city.synthesize("haga")
            >>> city.synthesize("107 Haga")
        """
        gbg_area = self.get_area(area)
        gbg_area.generate(**kwargs)
        return gbg_area

    def get_all_areas(self) -> Dict[str, str]:
        """
        Get all available primary areas in Gothenburg.
        
        Uses local registry (no API call needed).

        Returns:
            Dictionary mapping area codes to full names
            Example: {"107": "107 Haga", "108": "108 Annedal", ...}
        """
        registry = get_area_registry()
        return {code: info['full'] for code, info in registry.items()}

    def _get_area_api_value(self, area_code: str) -> str:
        """
        Get the full API value for an area code.
        
        Uses local registry first, falls back to API.
        
        Args:
            area_code: Short code like "107"
            
        Returns:
            Full API value like "107 Haga"
        """
        registry = get_area_registry()
        if area_code in registry:
            return registry[area_code]['full']
        # Fallback to API for unknown codes
        areas_raw = self.client.get_area_codes()
        if area_code in areas_raw:
            return areas_raw[area_code]['api_value']
        return area_code

    def get_area(self, area_identifier: str) -> GbgArea:
        """
        Get a GbgArea object for a specific area.
        
        Use this when you want to inspect data before generating,
        or when you want more control over the synthesis process.
        
        Supports flexible area matching - all of these work:
            - Area code: "107"
            - Area name: "Haga" (case-insensitive)
            - Full name: "107 Haga"

        Args:
            area_identifier: Area identifier in any of the formats above

        Returns:
            GbgArea object ready for synthesis

        Raises:
            AreaNotFoundError: If area cannot be found

        Example:
            >>> city = GbgSynth(year=2024)
            >>> haga = city.get_area("Haga")  # or "107" or "haga"
            >>> haga.generate()
        """
        registry = get_area_registry()
        
        # Look up area code
        code = self.get_area_code(area_identifier)
        
        if code is None:
            raise AreaNotFoundError(area_identifier)
        
        info = registry[code]
        api_value = self._get_area_api_value(code)
        
        return GbgArea(
            area_code=code,
            area_name=info['full'],
            area_api_value=api_value,
            year=self.year,
            client=self.client,
            config=self.config
        )

    def synthesize_future(
        self,
        area: str,
        target_year: int,
        base_year: int = 2025,
        **kwargs,
    ) -> GbgArea:
        """
        Synthesise a population scaled to a future prognosis year.

        This is a convenience method that combines ``synthesize()`` with
        prognosis scaling. It first generates a base-year population,
        then re-synthesises with marginals scaled by the official
        Gothenburg population prognosis.

        The prognosis data is published at the mellanområde (intermediate
        area) level. The mapping from primärområde to mellanområde is
        handled automatically.

        Available prognosis years: 2025–2032.

        Args:
            area: Area identifier (name, code, or full name)
            target_year: Future year to project to (2025–2032)
            base_year: Reference year in the prognosis (default 2025)
            **kwargs: Passed to ``scale_to_year()``

        Returns:
            GbgArea with the future-year scaled population

        Raises:
            AreaNotFoundError: If area cannot be found
            ValueError: If target_year outside 2025–2032

        Example:
            >>> city = GbgSynth(year=2024)
            >>> haga_2030 = city.synthesize_future("Haga", target_year=2030)
            >>> print(len(haga_2030.individuals))

            # Compare to base year:
            >>> haga_now = city.synthesize("Haga")
            >>> print(f"Growth: {len(haga_2030.individuals) - len(haga_now.individuals)}")
        """
        gbg_area = self.get_area(area)
        return gbg_area.scale_to_year(
            target_year=target_year,
            base_year=base_year,
            **kwargs,
        )

    def generate_all_areas(self, output_dir: str = "./output") -> None:
        """
        Generate synthetic populations for all areas (batch processing).

        Args:
            output_dir: Directory to save output files

        Example:
            >>> city = GbgSynth(year=2024)
            >>> city.generate_all_areas(output_dir="./gothenburg_2024")
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        all_areas = self.get_all_areas()
        total = len(all_areas)

        logger.info(f"Starting batch generation for {total} areas")

        for i, (code, name) in enumerate(all_areas.items(), 1):
            logger.info(f"[{i}/{total}] Processing {code}: {name}")
            
            try:
                area = self.get_area(code)
                area.generate()
                
                # Save outputs
                individuals_file = os.path.join(output_dir, f"{code}_{name.replace(' ', '_')}_individuals.csv")
                households_file = os.path.join(output_dir, f"{code}_{name.replace(' ', '_')}_households.csv")
                
                area.save_to_csv(individuals_file)
                area.save_households_to_csv(households_file)
                
                # Log summary
                stats = area.get_summary_statistics()
                logger.info(f"  ✓ {stats['total_population']} individuals, {stats['total_households']} households")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to generate {code}: {e}")
                continue

        logger.info(f"Batch generation complete. Files saved to {output_dir}")
