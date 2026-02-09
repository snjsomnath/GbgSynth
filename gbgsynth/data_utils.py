"""
Utility functions for managing bundled data files.

This module handles downloading and setting up required data files
that are not bundled with the package.
"""

import io
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Data directory paths
_DATA_DIR = Path(__file__).parent / "data"
_PRI_SHP_DIR = _DATA_DIR / "pri_shp"
_AREAS_JSON = _DATA_DIR / "areas.json"
_FOOTPRINTS_GPKG = _DATA_DIR / "footprints.gpkg"
_FOOTPRINTS_DIR = _DATA_DIR / "footprints"

# Download URL for primary area shapefile from Gothenburg city
PRI_SHAPEFILE_URL = (
    "https://goteborg.se/wps/wcm/connect/4b21c246-9f7c-4b9f-9360-262051792c62/"
    "Prim%C3%A4romr%C3%A5de_shp.zip?MOD=AJPERES"
)

# Required shapefile components
SHAPEFILE_EXTENSIONS = [".shp", ".shx", ".dbf", ".prj"]


def is_shapefile_available() -> bool:
    """
    Check if the primary area shapefile is available.
    
    Returns:
        True if all required shapefile components exist, False otherwise.
    """
    if not _PRI_SHP_DIR.exists():
        return False
    
    for ext in SHAPEFILE_EXTENSIONS:
        if not (_PRI_SHP_DIR / f"pri{ext}").exists():
            return False
    
    return True


def download_pri_shapefile(
    url: Optional[str] = None,
    target_dir: Optional[Path] = None,
    timeout: int = 30,
    force: bool = False
) -> bool:
    """
    Download and set up the primary area shapefile from Gothenburg city website.
    
    Downloads the ZIP file, extracts it, renames files to standard names,
    and organizes them in the correct directory structure.
    
    Args:
        url: URL to download from (default: Gothenburg city website)
        target_dir: Target directory for the shapefile (default: bundled data dir)
        timeout: Request timeout in seconds
        force: If True, re-download even if files already exist
        
    Returns:
        True if download and setup succeeded, False otherwise.
        
    Raises:
        requests.RequestException: If download fails
        zipfile.BadZipFile: If the downloaded file is not a valid ZIP
    """
    url = url or PRI_SHAPEFILE_URL
    target_dir = Path(target_dir) if target_dir else _PRI_SHP_DIR
    
    # Check if already available
    if not force and is_shapefile_available():
        logger.info("Shapefile already available, skipping download")
        return True
    
    logger.info(f"Downloading primary area shapefile from {url}")
    
    # Download the ZIP file
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract and process the ZIP file
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # List contents to find the shapefile components
        file_list = zf.namelist()
        logger.debug(f"ZIP contents: {file_list}")
        
        # Find and extract shapefile components
        extracted_files = _extract_and_rename_shapefile(zf, file_list, target_dir)
        
    if not extracted_files:
        logger.error("No shapefile components found in ZIP")
        return False
    
    logger.info(f"Successfully extracted shapefile to {target_dir}")
    return True


def _extract_and_rename_shapefile(
    zf: zipfile.ZipFile,
    file_list: list,
    target_dir: Path
) -> list:
    """
    Extract shapefile components from ZIP and rename to standard names.
    
    Args:
        zf: Open ZipFile object
        file_list: List of files in the ZIP
        target_dir: Target directory for extraction
        
    Returns:
        List of extracted file paths
    """
    extracted = []
    
    for ext in SHAPEFILE_EXTENSIONS:
        # Find file with this extension (case-insensitive)
        matching_files = [
            f for f in file_list 
            if f.lower().endswith(ext.lower()) and not f.startswith('__MACOSX')
        ]
        
        if not matching_files:
            logger.warning(f"No {ext} file found in ZIP")
            continue
        
        # Use the first matching file
        source_file = matching_files[0]
        target_file = target_dir / f"pri{ext}"
        
        # Extract the file
        with zf.open(source_file) as src:
            content = src.read()
            
        with open(target_file, 'wb') as dst:
            dst.write(content)
        
        extracted.append(target_file)
        logger.debug(f"Extracted {source_file} -> {target_file}")
    
    return extracted


def ensure_shapefile_available(auto_download: bool = True) -> bool:
    """
    Ensure the primary area shapefile is available, downloading if necessary.
    
    This is the main entry point for code that needs the shapefile.
    
    Args:
        auto_download: If True, automatically download if not available
        
    Returns:
        True if shapefile is available (or was successfully downloaded),
        False otherwise.
    """
    if is_shapefile_available():
        return True
    
    if not auto_download:
        logger.warning("Shapefile not available and auto_download is disabled")
        return False
    
    logger.info("Shapefile not found, attempting to download...")
    
    try:
        return download_pri_shapefile()
    except requests.RequestException as e:
        logger.error(f"Failed to download shapefile: {e}")
        return False
    except zipfile.BadZipFile as e:
        logger.error(f"Downloaded file is not a valid ZIP: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading shapefile: {e}")
        return False


def get_shapefile_path() -> Optional[Path]:
    """
    Get the path to the primary area shapefile.
    
    Returns:
        Path to the .shp file if available, None otherwise.
    """
    shp_path = _PRI_SHP_DIR / "pri.shp"
    if shp_path.exists():
        return shp_path
    return None


def cleanup_shapefile(target_dir: Optional[Path] = None) -> bool:
    """
    Remove the shapefile directory and all its contents.
    
    Useful for testing or forcing a fresh download.
    
    Args:
        target_dir: Directory to clean up (default: bundled data dir)
        
    Returns:
        True if cleanup succeeded, False otherwise.
    """
    target_dir = Path(target_dir) if target_dir else _PRI_SHP_DIR
    
    if not target_dir.exists():
        return True
    
    try:
        shutil.rmtree(target_dir)
        logger.info(f"Cleaned up {target_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to cleanup {target_dir}: {e}")
        return False


# ============================================================================
# Areas JSON functions
# ============================================================================

def is_areas_json_available(areas_json_path: Optional[Path] = None) -> bool:
    """
    Check if the areas.json file exists.
    
    Args:
        areas_json_path: Path to check (default: bundled data path)
        
    Returns:
        True if areas.json exists, False otherwise.
    """
    path = Path(areas_json_path) if areas_json_path else _AREAS_JSON
    return path.exists()


def generate_areas_json(
    shapefile_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    area_code_column: str = "PRIMÄROMRÅ",
    area_name_column: str = "PRIMÄRNAMN",
    force: bool = False
) -> bool:
    """
    Generate areas.json from the primary area shapefile.
    
    Reads the shapefile and creates a JSON file mapping area codes
    to their names and full identifiers.
    
    Args:
        shapefile_path: Path to the shapefile (default: bundled shapefile)
        output_path: Path for output JSON (default: bundled areas.json)
        area_code_column: Column name for area codes in shapefile
        area_name_column: Column name for area names in shapefile
        force: If True, regenerate even if areas.json already exists
        
    Returns:
        True if generation succeeded, False otherwise.
        
    Raises:
        ImportError: If geopandas is not installed
    """
    output_path = Path(output_path) if output_path else _AREAS_JSON
    
    # Check if already exists
    if not force and output_path.exists():
        logger.info("areas.json already exists, skipping generation")
        return True
    
    # Determine shapefile path
    if shapefile_path:
        shp_path = Path(shapefile_path)
    else:
        shp_path = get_shapefile_path()
        if shp_path is None:
            logger.error("Shapefile not available, cannot generate areas.json")
            return False
    
    # Try to import geopandas
    try:
        import geopandas as gpd
    except ImportError:
        logger.error("geopandas is required to generate areas.json from shapefile")
        raise ImportError(
            "geopandas is required to generate areas.json. "
            "Install it with: pip install geopandas"
        )
    
    logger.info(f"Generating areas.json from {shp_path}")
    
    try:
        # Read the shapefile
        gdf = gpd.read_file(shp_path)
        
        # Build the areas dictionary
        areas = {}
        for _, row in gdf.iterrows():
            code = str(row[area_code_column])
            name = str(row[area_name_column])
            areas[code] = {
                "name": name,
                "full": f"{code} {name}"
            }
        
        # Sort by area code for consistent output
        sorted_areas = dict(sorted(areas.items(), key=lambda x: int(x[0])))
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_areas, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated areas.json with {len(areas)} areas")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate areas.json: {e}")
        return False


def ensure_areas_json_available(auto_generate: bool = True) -> bool:
    """
    Ensure areas.json is available, generating from shapefile if necessary.
    
    This function first ensures the shapefile is available, then checks
    for areas.json and generates it if missing.
    
    Args:
        auto_generate: If True, automatically generate from shapefile if missing
        
    Returns:
        True if areas.json is available, False otherwise.
    """
    if is_areas_json_available():
        return True
    
    if not auto_generate:
        logger.warning("areas.json not available and auto_generate is disabled")
        return False
    
    # First ensure shapefile is available
    if not ensure_shapefile_available():
        logger.error("Cannot generate areas.json: shapefile not available")
        return False
    
    logger.info("areas.json not found, generating from shapefile...")
    
    try:
        return generate_areas_json()
    except ImportError as e:
        logger.error(f"Cannot generate areas.json: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error generating areas.json: {e}")
        return False


def get_areas_json_path() -> Optional[Path]:
    """
    Get the path to areas.json.
    
    Returns:
        Path to areas.json if it exists, None otherwise.
    """
    if _AREAS_JSON.exists():
        return _AREAS_JSON
    return None


def ensure_data_available(auto_download: bool = True, auto_generate: bool = True) -> bool:
    """
    Ensure all bundled data files are available.
    
    This is the main entry point for ensuring all required data is present.
    It will:
    1. Download the shapefile if not present
    2. Generate areas.json from the shapefile if not present
    3. Download footprints if not present (requires dtcc package)
    4. Generate neighbourhood heights if not present (requires dtcc package)
    
    Args:
        auto_download: If True, download shapefile and footprints if not available
        auto_generate: If True, generate areas.json and heights if not available
        
    Returns:
        True if all data is available, False otherwise.
    """
    # First ensure shapefile
    if not ensure_shapefile_available(auto_download=auto_download):
        return False
    
    # Then ensure areas.json
    if not ensure_areas_json_available(auto_generate=auto_generate):
        return False
    
    # Footprints are optional - try to download but don't fail if unavailable
    if auto_download and not is_footprints_available():
        try:
            ensure_footprints_available(auto_download=True)
        except Exception as e:
            logger.warning(f"Could not download footprints (optional): {e}")
    
    # Neighbourhood heights are optional - try to generate but don't fail
    if auto_generate and is_footprints_available():
        try:
            ensure_neighbourhood_heights_available(auto_generate=True)
        except Exception as e:
            logger.warning(f"Could not generate neighbourhood heights (optional): {e}")
    
    return True


# ============================================================================
# Footprints functions
# ============================================================================

def is_footprints_available(footprints_path: Optional[Path] = None) -> bool:
    """
    Check if the footprints GeoPackage file exists.
    
    Args:
        footprints_path: Path to check (default: bundled data path)
        
    Returns:
        True if footprints.gpkg exists, False otherwise.
    """
    path = Path(footprints_path) if footprints_path else _FOOTPRINTS_GPKG
    return path.exists()


def get_shapefile_bounds() -> Optional[tuple]:
    """
    Get the bounding box of the primary area shapefile in EPSG:3006.
    
    The shapefile is reprojected to EPSG:3006 (SWEREF99 TM) before
    extracting the bounds, which is the coordinate system expected
    by the DTCC package.
    
    Returns:
        Tuple of (minx, miny, maxx, maxy) in EPSG:3006 or None if 
        shapefile not available.
        
    Raises:
        ImportError: If geopandas is not installed.
    """
    shapefile_path = get_shapefile_path()
    if shapefile_path is None:
        logger.error("Shapefile not available, cannot get bounds")
        return None
    
    try:
        import geopandas as gpd
    except ImportError:
        logger.error("geopandas is required to read shapefile bounds")
        raise ImportError(
            "geopandas is required to read shapefile bounds. "
            "Install it with: pip install geopandas"
        )
    
    gdf = gpd.read_file(shapefile_path)
    
    # Reproject to EPSG:3006 (SWEREF99 TM) if not already
    if gdf.crs is not None and gdf.crs.to_epsg() != 3006:
        logger.debug(f"Reprojecting from {gdf.crs} to EPSG:3006")
        gdf = gdf.to_crs(epsg=3006)
    elif gdf.crs is None:
        logger.warning("Shapefile has no CRS defined, assuming EPSG:3006")
    
    bounds = gdf.total_bounds  # Returns (minx, miny, maxx, maxy)
    logger.debug(f"Shapefile bounds (EPSG:3006): {bounds}")
    return tuple(bounds)


def download_footprints(
    output_path: Optional[Path] = None,
    bounds: Optional[tuple] = None,
    force: bool = False
) -> bool:
    """
    Download building footprints using the DTCC package.
    
    Downloads footprints for the area covered by the primary shapefile
    and saves them to a GeoPackage file.
    
    Args:
        output_path: Path for output GeoPackage (default: bundled footprints.gpkg)
        bounds: Bounding box as (minx, miny, maxx, maxy). If None, uses shapefile bounds.
        force: If True, re-download even if file already exists
        
    Returns:
        True if download succeeded, False otherwise.
        
    Raises:
        ImportError: If dtcc or geopandas is not installed.
    """
    output_path = Path(output_path) if output_path else _FOOTPRINTS_GPKG
    
    # Check if already exists
    if not force and output_path.exists():
        logger.info("Footprints already available, skipping download")
        return True
    
    # Get bounds if not provided
    if bounds is None:
        bounds = get_shapefile_bounds()
        if bounds is None:
            logger.error("Cannot download footprints: shapefile bounds not available")
            return False
    
    # Try to import dtcc
    try:
        import dtcc
    except ImportError:
        logger.error("dtcc package is required to download footprints")
        raise ImportError(
            "dtcc package is required to download footprints. "
            "Install it with: pip install dtcc"
        )
    
    logger.info(f"Downloading footprints for bounds: {bounds}")
    
    try:
        # Create DTCC Bounds object
        dtcc_bounds = dtcc.Bounds(
            xmin=bounds[0],
            ymin=bounds[1],
            xmax=bounds[2],
            ymax=bounds[3]
        )
        
        # Download footprints using dtcc
        # This downloads to a temporary location
        footprints = dtcc.download_footprints(bounds=dtcc_bounds)
        
        # Find and copy the downloaded GeoPackage file
        # dtcc stores downloads in a cache directory
        if hasattr(footprints, 'path') and footprints.path:
            source_path = Path(footprints.path)
            if source_path.exists():
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, output_path)
                logger.info(f"Copied footprints to {output_path}")
                return True
        
        # If footprints is a GeoDataFrame, save it directly
        try:
            import geopandas as gpd
            if hasattr(footprints, 'to_file') or isinstance(footprints, gpd.GeoDataFrame):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                footprints.to_file(output_path, driver='GPKG')
                logger.info(f"Saved footprints to {output_path}")
                return True
        except Exception as e:
            logger.debug(f"Could not save as GeoDataFrame: {e}")
        
        # Try to find the downloaded file in dtcc cache
        dtcc_cache = _find_dtcc_cache_file(footprints)
        if dtcc_cache and dtcc_cache.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dtcc_cache, output_path)
            logger.info(f"Copied footprints from cache to {output_path}")
            return True
        
        logger.error("Could not locate downloaded footprints file")
        return False
        
    except Exception as e:
        logger.error(f"Failed to download footprints: {e}")
        return False


def _find_dtcc_cache_file(footprints_obj) -> Optional[Path]:
    """
    Try to find the GeoPackage file in dtcc's cache directory.
    
    Args:
        footprints_obj: The object returned by dtcc.download_footprints()
        
    Returns:
        Path to the cached GeoPackage file, or None if not found.
    """
    import tempfile
    
    # Common locations where dtcc might store downloaded files
    search_dirs = [
        Path(tempfile.gettempdir()),
        Path.home() / '.dtcc' / 'cache',
        Path.home() / '.cache' / 'dtcc',
    ]
    
    # Also check if the object has any path-like attributes
    for attr in ['path', 'filepath', 'file_path', 'filename', '_path']:
        if hasattr(footprints_obj, attr):
            path = getattr(footprints_obj, attr)
            if path and Path(path).exists():
                return Path(path)
    
    # Search for recent .gpkg files in common directories
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        for gpkg_file in search_dir.glob('**/*.gpkg'):
            # Check if it's a recent file (within last hour)
            import time
            if time.time() - gpkg_file.stat().st_mtime < 3600:
                logger.debug(f"Found potential footprints cache: {gpkg_file}")
                return gpkg_file
    
    return None


def ensure_footprints_available(auto_download: bool = True) -> bool:
    """
    Ensure the footprints GeoPackage is available, downloading if necessary.
    
    This function first ensures the shapefile is available (to get bounds),
    then downloads footprints using dtcc if missing.
    
    Args:
        auto_download: If True, automatically download if not available
        
    Returns:
        True if footprints are available, False otherwise.
    """
    if is_footprints_available():
        return True
    
    if not auto_download:
        logger.warning("Footprints not available and auto_download is disabled")
        return False
    
    # First ensure shapefile is available (needed for bounds)
    if not ensure_shapefile_available():
        logger.error("Cannot download footprints: shapefile not available")
        return False
    
    logger.info("Footprints not found, attempting to download using dtcc...")
    
    try:
        return download_footprints()
    except ImportError as e:
        logger.warning(f"Cannot download footprints: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading footprints: {e}")
        return False


def get_footprints_path() -> Optional[Path]:
    """
    Get the path to the footprints GeoPackage.
    
    Returns:
        Path to footprints.gpkg if it exists, None otherwise.
    """
    if _FOOTPRINTS_GPKG.exists():
        return _FOOTPRINTS_GPKG
    return None


# ============================================================================
# Neighbourhood Heights functions
# ============================================================================

# Neighbourhoods to skip during height processing (e.g., no buildings or data issues)
SKIP_NEIGHBOURHOODS = []


def get_neighbourhood_heights_dir() -> Path:
    """
    Get the path to the neighbourhood heights directory.
    
    Returns:
        Path to the footprints directory.
    """
    return _FOOTPRINTS_DIR


def is_neighbourhood_heights_available(
    neighbourhood_name: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> bool:
    """
    Check if neighbourhood height files are available.
    
    Args:
        neighbourhood_name: If provided, check only this neighbourhood.
                           If None, check all neighbourhoods from areas.json.
        output_dir: Directory to check (default: bundled footprints dir)
        
    Returns:
        True if all required height files exist, False otherwise.
    """
    output_dir = Path(output_dir) if output_dir else _FOOTPRINTS_DIR
    
    if not output_dir.exists():
        return False
    
    if neighbourhood_name:
        # Check single neighbourhood
        safe_name = neighbourhood_name.replace(" ", "_").replace("/", "_")
        height_file = output_dir / f"{safe_name}_heights.gpkg"
        return height_file.exists()
    
    # Check all neighbourhoods from areas.json
    if not is_areas_json_available():
        logger.warning("areas.json not available, cannot check all neighbourhoods")
        return False
    
    with open(_AREAS_JSON, 'r', encoding='utf-8') as f:
        areas = json.load(f)
    
    for code, info in areas.items():
        name = info.get('name', '')
        if name in SKIP_NEIGHBOURHOODS:
            continue
        safe_name = name.replace(" ", "_").replace("/", "_")
        height_file = output_dir / f"{safe_name}_heights.gpkg"
        if not height_file.exists():
            return False
    
    return True


def get_missing_neighbourhood_heights(
    output_dir: Optional[Path] = None
) -> list:
    """
    Get list of neighbourhoods that are missing height files.
    
    Args:
        output_dir: Directory to check (default: bundled footprints dir)
        
    Returns:
        List of neighbourhood names that need processing.
    """
    output_dir = Path(output_dir) if output_dir else _FOOTPRINTS_DIR
    
    if not is_areas_json_available():
        logger.warning("areas.json not available")
        return []
    
    with open(_AREAS_JSON, 'r', encoding='utf-8') as f:
        areas = json.load(f)
    
    missing = []
    for code, info in areas.items():
        name = info.get('name', '')
        if name in SKIP_NEIGHBOURHOODS:
            continue
        safe_name = name.replace(" ", "_").replace("/", "_")
        height_file = output_dir / f"{safe_name}_heights.gpkg"
        if not height_file.exists():
            missing.append({'code': code, 'name': name})
    
    return missing


def generate_neighbourhood_heights(
    neighbourhood_name: Optional[str] = None,
    neighbourhood_code: Optional[str] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
    cell_size: int = 5
) -> bool:
    """
    Generate building heights for a neighbourhood or all neighbourhoods.
    
    Uses DTCC to download pointcloud data, compute terrain, and extract
    building heights for each neighbourhood.
    
    Args:
        neighbourhood_name: Name of specific neighbourhood to process.
                           If None, processes all missing neighbourhoods.
        neighbourhood_code: Code of the neighbourhood (for metadata).
        output_dir: Directory to save height files (default: bundled footprints dir)
        force: If True, regenerate even if file exists.
        cell_size: Cell size for terrain raster (default: 5, higher = less RAM)
        
    Returns:
        True if processing succeeded, False otherwise.
        
    Raises:
        ImportError: If required packages (geopandas, dtcc) are not installed.
    """
    import gc
    
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required to generate neighbourhood heights. "
            "Install it with: pip install geopandas"
        )
    
    try:
        import dtcc
    except ImportError:
        raise ImportError(
            "dtcc is required to generate neighbourhood heights. "
            "Install it with: pip install dtcc"
        )
    
    from shapely.geometry import Polygon
    
    output_dir = Path(output_dir) if output_dir else _FOOTPRINTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load shapefile for bounds
    shapefile_path = get_shapefile_path()
    if shapefile_path is None:
        logger.error("Shapefile not available, cannot generate heights")
        return False
    
    gbg_shp = gpd.read_file(shapefile_path).to_crs(epsg=3006)
    
    # Determine which neighbourhoods to process
    if neighbourhood_name:
        neighbourhoods_to_process = [{'name': neighbourhood_name, 'code': neighbourhood_code}]
    else:
        # Get all missing neighbourhoods
        neighbourhoods_to_process = get_missing_neighbourhood_heights(output_dir)
        if not neighbourhoods_to_process:
            logger.info("All neighbourhood heights already generated")
            return True
    
    success_count = 0
    total_count = len(neighbourhoods_to_process)
    
    for nbr_info in neighbourhoods_to_process:
        nbr_name = nbr_info['name']
        nbr_code = nbr_info.get('code', 'N/A')
        
        if nbr_name in SKIP_NEIGHBOURHOODS:
            logger.info(f"Skipping {nbr_name} (excluded from processing)")
            continue
        
        safe_name = nbr_name.replace(" ", "_").replace("/", "_")
        output_path = output_dir / f"{safe_name}_heights.gpkg"
        
        if output_path.exists() and not force:
            logger.info(f"Skipping {nbr_name} (already processed)")
            success_count += 1
            continue
        
        logger.info(f"Processing neighbourhood: {nbr_name}")
        
        # Initialize for cleanup
        pointcloud = buildings = raster = nbr_gdf = None
        
        try:
            # Get neighbourhood geometry from shapefile
            nbr_rows = gbg_shp[gbg_shp['PRIMÄRNAMN'] == nbr_name]
            if len(nbr_rows) == 0:
                logger.warning(f"Neighbourhood {nbr_name} not found in shapefile")
                continue
            
            row = nbr_rows.iloc[0]
            if nbr_code == 'N/A':
                nbr_code = row.get('PRIMÄRKOD', 'N/A')
            
            # Get bounds for this neighbourhood
            bounds = dtcc.Bounds(*nbr_rows.total_bounds)
            
            # Download data
            pointcloud = dtcc.download_pointcloud(bounds=bounds)
            buildings = dtcc.download_footprints(bounds=bounds)
            
            if not buildings or len(buildings) == 0:
                logger.warning(f"No footprints found for {nbr_name}")
                continue
            
            # Processing pipeline
            pointcloud = pointcloud.remove_global_outliers(3.0)
            raster = dtcc.builder.build_terrain_raster(
                pointcloud, cell_size=cell_size, ground_only=True
            )
            
            # Extract roof points and compute heights
            buildings = dtcc.extract_roof_points(buildings, pointcloud)
            buildings = dtcc.compute_building_heights(buildings, raster, overwrite=True)
            
            # Extract data to GeoDataFrame
            data = []
            for b in buildings:
                h = b.attributes.get('height', 0)
                oid = b.attributes.get('objektidentitet', 'N/A')
                
                geom = None
                # Extract geometry from the building object
                if hasattr(b, 'geometry') and isinstance(b.geometry, dict):
                    for g_val in b.geometry.values():
                        if hasattr(g_val, 'vertices') and len(g_val.vertices) > 0:
                            coords = [(v[0], v[1]) for v in g_val.vertices]
                            if len(coords) >= 3:
                                geom = Polygon(coords)
                                break
                
                # Fallback for older DTCC structures
                if geom is None and hasattr(b, 'footprint'):
                    fp = b.footprint
                    if hasattr(fp, 'vertices'):
                        geom = Polygon([(v[0], v[1]) for v in fp.vertices])
                
                if geom is not None:
                    # Extract building type attributes for residential filtering
                    andamal1 = b.attributes.get('andamal1', None)
                    objekttyp = b.attributes.get('objekttyp', None)
                    
                    data.append({
                        'objektidentitet': oid,
                        'height': h,
                        'andamal1': andamal1,
                        'objekttyp': objekttyp,
                        'neighborhood_name': nbr_name,
                        'neighborhood_code': nbr_code,
                        'geometry': geom
                    })
            
            # Save results
            if data:
                nbr_gdf = gpd.GeoDataFrame(data, crs="EPSG:3006")
                nbr_gdf.to_file(output_path, driver="GPKG", layer='buildings')
                logger.info(f"Saved {len(nbr_gdf)} buildings for {nbr_name}")
                success_count += 1
            else:
                logger.warning(f"Failed to extract valid geometries for {nbr_name}")
        
        except Exception as e:
            logger.error(f"Error processing {nbr_name}: {e}")
        
        finally:
            # Release memory
            del pointcloud, buildings, raster, nbr_gdf
            gc.collect()
    
    logger.info(f"Processed {success_count}/{total_count} neighbourhoods successfully")
    return success_count > 0


def ensure_neighbourhood_heights_available(auto_generate: bool = True) -> bool:
    """
    Ensure neighbourhood height files are available.
    
    This function first ensures footprints are available, then generates
    height files for all neighbourhoods that are missing.
    
    Args:
        auto_generate: If True, automatically generate if not available
        
    Returns:
        True if all heights are available (or generation succeeded), False otherwise.
    """
    if is_neighbourhood_heights_available():
        return True
    
    if not auto_generate:
        logger.warning("Neighbourhood heights not available and auto_generate is disabled")
        return False
    
    # First ensure footprints are available
    if not is_footprints_available():
        logger.error("Cannot generate heights: footprints not available")
        return False
    
    logger.info("Some neighbourhood heights missing, generating...")
    
    try:
        return generate_neighbourhood_heights()
    except ImportError as e:
        logger.warning(f"Cannot generate neighbourhood heights: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error generating heights: {e}")
        return False


def get_neighbourhood_heights_path(neighbourhood_name: str) -> Optional[Path]:
    """
    Get the path to a neighbourhood's height file.
    
    Args:
        neighbourhood_name: Name of the neighbourhood.
        
    Returns:
        Path to the height file if it exists, None otherwise.
    """
    safe_name = neighbourhood_name.replace(" ", "_").replace("/", "_")
    height_file = _FOOTPRINTS_DIR / f"{safe_name}_heights.gpkg"
    if height_file.exists():
        return height_file
    return None
