"""GbgSynth - Synthetic Population Generator for Gothenburg.

Simple API:
    >>> from gbgsynth import GbgSynth
    >>> city = GbgSynth(year=2024)
    >>> haga = city.synthesize("Haga")
    >>> haga.save("output/")

For advanced usage, see the documentation.
"""

__version__ = "0.3.0"

# Core API - what most users need
from gbgsynth.gbgsynth import GbgSynth
from gbgsynth.area import GbgArea
from gbgsynth.models import Agent, Household, Dwelling

# Exceptions for error handling
from gbgsynth.exceptions import (
    GbgSynthError,
    AreaNotFoundError,
    APIError,
    DataNotGeneratedError,
    InvalidDataError,
    PrivacySuppressionError,
)

# Optional modules (import explicitly if needed)
from gbgsynth import plotting
from gbgsynth import validation
from gbgsynth import exporters
from gbgsynth import prognosis

# Prognosis scaling
from gbgsynth.prognosis import (
    PrognosisScaler,
    PrognosisClient,
    get_pri_to_mel,
    get_mel_for_pri,
    get_sibling_pri_codes,
    PROGNOSIS_YEARS,
)

# Exporter convenience imports
from gbgsynth.exporters import (
    get_exporter,
    list_exporters,
    SweLoadSimConfig,
)

# Sanity checks for population validation
from gbgsynth.sanity_checks import (
    run_all_checks,
    validate_population,
    SanityCheckResult,
    SanityViolation,
)

# Data utilities for managing bundled data
from gbgsynth.data_utils import (
    download_pri_shapefile,
    download_mel_shapefile,
    download_footprints,
    ensure_shapefile_available,
    ensure_mel_shapefile_available,
    ensure_areas_json_available,
    ensure_footprints_available,
    ensure_neighbourhood_heights_available,
    ensure_data_available,
    generate_areas_json,
    generate_neighbourhood_heights,
    is_shapefile_available,
    is_mel_shapefile_available,
    is_areas_json_available,
    is_footprints_available,
    is_neighbourhood_heights_available,
    get_neighbourhood_heights_path,
    get_missing_neighbourhood_heights,
    get_mel_shapefile_path,
)

__version__ = "0.3.0"
__all__ = [
    # Core
    "GbgSynth",
    "GbgArea",
    "Agent",
    "Household",
    "Dwelling",
    # Exceptions
    "GbgSynthError",
    "AreaNotFoundError",
    "APIError",
    "DataNotGeneratedError",
    "InvalidDataError",
    "PrivacySuppressionError",
    # Modules
    "plotting",
    "validation",
    "exporters",
    # Exporters
    "get_exporter",
    "list_exporters",
    "SweLoadSimConfig",
    # Data utilities
    "download_pri_shapefile",
    "download_mel_shapefile",
    "download_footprints",
    "ensure_shapefile_available",
    "ensure_mel_shapefile_available",
    "ensure_areas_json_available",
    "ensure_footprints_available",
    "ensure_neighbourhood_heights_available",
    "ensure_data_available",
    "generate_areas_json",
    "generate_neighbourhood_heights",
    "is_shapefile_available",
    "is_mel_shapefile_available",
    "is_areas_json_available",
    "is_footprints_available",
    "is_neighbourhood_heights_available",
    "get_neighbourhood_heights_path",
    "get_missing_neighbourhood_heights",
    "get_mel_shapefile_path",
    # Prognosis scaling
    "prognosis",
    "PrognosisScaler",
    "PrognosisClient",
    "get_pri_to_mel",
    "get_mel_for_pri",
    "get_sibling_pri_codes",
    "PROGNOSIS_YEARS",
]
