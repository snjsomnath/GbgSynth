"""
Household container creation from census size/type marginals.

This module handles Phase 1 of the top-down synthesis: creating exact
household containers whose size distribution matches the census data.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from gbgsynth.config import Config
from gbgsynth.models import Household

logger = logging.getLogger(__name__)


def parse_household_size(size_str, config: Config) -> int:
    """Parse household size from a string label using config mappings.

    Args:
        size_str: Size label (e.g. ``'3 personer'``) or an ``int``.
        config: Config object whose ``household_size_mappings`` dict is
                used for the lookup.

    Returns:
        Integer household size (defaults to 1 for unknown labels).
    """
    if isinstance(size_str, int):
        return size_str
    mapping = config.household_size_mappings
    return mapping.get(size_str, 1)


def create_household_containers(
    household_data: pd.DataFrame,
    config: Config,
    start_id: int = 1,
) -> tuple:
    """Create exact household containers from census size distribution.

    Each row in *household_data* is expected to carry a size label and a
    count.  Containers are created largest-first so that downstream
    matching can fill complex households before simple ones.

    Args:
        household_data: DataFrame with household size / type / count
            columns (Swedish or English names accepted).
        config: Config object for parsing size labels.
        start_id: Starting household ID (auto-incremented).

    Returns:
        ``(containers, next_id)`` — a list of :class:`Household` objects
        (all empty, ``house_type=None``) and the next unused ID.
    """
    hh_size_col = (
        'Hushållsstorlek'
        if 'Hushållsstorlek' in household_data.columns
        else 'hh_size'
    )
    count_col = (
        'Antal'
        if 'Antal' in household_data.columns
        else household_data.columns[-1]
    )

    size_counts: Dict[str, int] = (
        household_data.groupby(hh_size_col)[count_col].sum().to_dict()
    )

    containers: List[Household] = []
    next_id = start_id

    for size_label, count in sorted(
        size_counts.items(),
        key=lambda x: parse_household_size(x[0], config),
        reverse=True,
    ):
        size = parse_household_size(size_label, config)
        if size == 0 or count == 0:
            continue

        for _ in range(int(count)):
            hh = Household(
                household_id=next_id,
                size=size,
                house_type=None,
                cars=0,
                assigned_hustyp=None,
            )
            containers.append(hh)
            next_id += 1

    logger.info("Created %d exact household containers", len(containers))

    size_dist: Dict[int, int] = {}
    for hh in containers:
        size_dist[hh.size] = size_dist.get(hh.size, 0) + 1
    logger.info("Household size distribution: %s", dict(sorted(size_dist.items())))

    return containers, next_id
