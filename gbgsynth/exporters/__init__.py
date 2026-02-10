"""
GbgSynth Exporters - Format converters for downstream simulation tools.

Available exporters:
- sweloadsim: SweLoadSim household energy simulation
- matsim: MATSim transport simulation (planned)
- csv: Simple tabular export

Usage:
    >>> from gbgsynth.exporters import get_exporter, list_exporters
    >>> exporter = get_exporter("sweloadsim", ev_probability=0.4)
    >>> exporter.export(area, "output.json")
    
    # Or via Area.export():
    >>> area.export("sweloadsim", "output.json")
"""

from .base import BaseExporter
from .sweloadsim import (
    SweLoadSimExporter,
    SweLoadSimConfig,
    HeatingConfig,
    EVConfig,
    SolarConfig,
    BatteryConfig,
)

EXPORTERS = {
    "sweloadsim": SweLoadSimExporter,
}


def get_exporter(name: str, **kwargs) -> BaseExporter:
    """
    Get an exporter instance by name.
    
    Args:
        name: Exporter name ("sweloadsim", "matsim", "csv")
        **kwargs: Exporter-specific configuration options
        
    Returns:
        Configured exporter instance
        
    Raises:
        ValueError: If exporter name is not recognized
        
    Example:
        >>> exporter = get_exporter("sweloadsim", config=SweLoadSimConfig.swedish_2024())
        >>> exporter.export(area, "output.json")
    """
    if name not in EXPORTERS:
        available = ", ".join(EXPORTERS.keys())
        raise ValueError(f"Unknown exporter '{name}'. Available: {available}")
    return EXPORTERS[name](**kwargs)


def list_exporters() -> list:
    """
    List available exporter names.
    
    Returns:
        List of exporter name strings
    """
    return list(EXPORTERS.keys())


__all__ = [
    # Functions
    "get_exporter",
    "list_exporters",
    # Base class
    "BaseExporter",
    # SweLoadSim
    "SweLoadSimExporter",
    "SweLoadSimConfig",
    "HeatingConfig",
    "EVConfig",
    "SolarConfig",
    "BatteryConfig",
]
