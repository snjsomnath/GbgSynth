"""
Housing-type assignment and building linkage.

This module:
1. Assigns Swedish ``Hustyp`` labels (Småhus / Flerbostadshus /
   Specialbostad) to households based on size-conditioned census
   distributions.
2. Links households to physical building footprints.
"""

import logging
import random
from typing import Dict, List, Optional

import pandas as pd

from gbgsynth.config import Config
from gbgsynth.models import Household

logger = logging.getLogger(__name__)


# ── Type translation helpers ─────────────────────────────────────────────

def parse_house_type(type_str) -> str:
    """Parse house-type string to internal English label."""
    if pd.isna(type_str):
        return 'apartment'
    type_lower = str(type_str).lower()
    if 'småhus' in type_lower:
        return 'detached_house'
    elif 'special' in type_lower:
        return 'special_housing'
    return 'apartment'


def normalize_building_type(building_type) -> str:
    """Normalise building type string to Swedish Hustyp."""
    if pd.isna(building_type):
        return 'Flerbostadshus'

    type_lower = str(building_type).lower()

    if any(t in type_lower for t in (
        'småhus', 'villa', 'small house', 'detached', 'semi-detached',
        'radhus', 'townhouse', 'single-family', 'enfamiljs',
    )):
        return 'Småhus'

    if any(t in type_lower for t in (
        'special', 'student', 'elderly', 'äldre', 'gruppbostad',
        'servicehus', 'care',
    )):
        return 'Specialbostad'

    return 'Flerbostadshus'


def english_to_swedish_hustyp(house_type: Optional[str]) -> str:
    """Convert English house-type label to Swedish Hustyp."""
    if house_type is None:
        return 'Flerbostadshus'
    return {
        'detached_house': 'Småhus',
        'apartment': 'Flerbostadshus',
        'special_housing': 'Specialbostad',
    }.get(house_type, 'Flerbostadshus')


# ── Housing-type distribution ────────────────────────────────────────────

def build_hustyp_distributions(
    hh_hustyp_dist: pd.DataFrame,
    config: Config,
) -> Dict[int, Dict[str, float]]:
    """Build ``{household_size: {hustyp: probability}}`` lookup."""
    from gbgsynth.helpers.household_factory import parse_household_size

    distributions: Dict[int, Dict[str, float]] = {}

    size_col = type_col = count_col = None
    for col in hh_hustyp_dist.columns:
        cl = col.lower()
        if 'storlek' in cl or 'size' in cl:
            size_col = col
        elif 'hustyp' in cl or 'house' in cl:
            type_col = col
        elif 'antal' in cl or 'count' in cl:
            count_col = col
    if count_col is None:
        count_col = hh_hustyp_dist.columns[-1]

    if size_col is None or type_col is None:
        logger.warning("Could not identify size/type columns in household data")
        return distributions

    for _, row in hh_hustyp_dist.iterrows():
        size = parse_household_size(row[size_col], config)
        hustyp = str(row[type_col])
        count = float(row[count_col]) if pd.notna(row[count_col]) else 0

        if size not in distributions:
            distributions[size] = {}
        distributions[size][hustyp] = distributions[size].get(hustyp, 0) + count

    for size in distributions:
        total = sum(distributions[size].values())
        if total > 0:
            distributions[size] = {k: v / total for k, v in distributions[size].items()}

    logger.debug("Built housing type distributions for sizes: %s", list(distributions.keys()))
    return distributions


# ── Assignment ───────────────────────────────────────────────────────────

def assign_housing_types(
    hh_hustyp_dist: pd.DataFrame,
    config: Config,
    households: Optional[List[Household]] = None,
) -> List[Household]:
    """Assign Hustyp labels to *households* using size-conditioned probs.

    Also sets ``hh.house_type`` (English label) to stay consistent.

    Returns:
        The same list of households (mutated in place).
    """
    if households is None or len(households) == 0:
        logger.warning("No households to assign housing types to")
        return []

    size_distributions = build_hustyp_distributions(hh_hustyp_dist, config)

    assigned_count: Dict[str, int] = {
        'Småhus': 0, 'Flerbostadshus': 0, 'Specialbostad': 0, 'unknown': 0,
    }

    for hh in households:
        if hh.size in size_distributions:
            dist = size_distributions[hh.size]
            hustyp = random.choices(
                list(dist.keys()), weights=list(dist.values())
            )[0]
        else:
            hustyp = 'Flerbostadshus'
            logger.debug(
                "No distribution for size %d, defaulting to Flerbostadshus",
                hh.size,
            )

        hh.assigned_hustyp = hustyp
        hh.house_type = parse_house_type(hustyp)
        assigned_count[hustyp] = assigned_count.get(hustyp, 0) + 1

    logger.info("Assigned housing types: %s", assigned_count)
    return households


# ── Building linkage ─────────────────────────────────────────────────────

def link_to_buildings(
    buildings: pd.DataFrame,
    households: Optional[List[Household]] = None,
    building_id_col: str = 'building_id',
    building_type_col: str = 'type',
    capacity_col: Optional[str] = None,
    income_weighted: bool = False,
) -> List[Household]:
    """Link households to building footprints by house-type matching.

    Returns:
        The same list of households (mutated in place).
    """
    if households is None or len(households) == 0:
        logger.warning("No households to link to buildings")
        return []

    unassigned = [hh for hh in households if hh.assigned_hustyp is None]
    if unassigned:
        logger.warning(
            "%d households missing assigned_hustyp, using house_type",
            len(unassigned),
        )
        for hh in unassigned:
            hh.assigned_hustyp = english_to_swedish_hustyp(hh.house_type)

    buildings = buildings.copy()
    buildings['_hustyp'] = buildings[building_type_col].apply(normalize_building_type)

    building_occupancy: Dict[int, int] = {}
    building_capacity: Dict[int, int] = {}

    for _, bldg in buildings.iterrows():
        bid = bldg[building_id_col]
        if capacity_col and capacity_col in bldg:
            building_capacity[bid] = int(bldg[capacity_col])
        else:
            building_capacity[bid] = 1 if bldg['_hustyp'] == 'Småhus' else 999
        building_occupancy[bid] = 0

    if income_weighted:
        households = sorted(households, key=lambda h: h.income, reverse=True)

    for hh in households:
        target_hustyp = hh.assigned_hustyp or 'Flerbostadshus'

        compatible = buildings[
            (buildings['_hustyp'] == target_hustyp)
            & buildings[building_id_col].apply(
                lambda bid: building_occupancy.get(bid, 0) < building_capacity.get(bid, 1)
            )
        ]

        if len(compatible) == 0:
            compatible = buildings[
                buildings[building_id_col].apply(
                    lambda bid: building_occupancy.get(bid, 0) < building_capacity.get(bid, 1)
                )
            ]
            if len(compatible) > 0:
                logger.debug(
                    "No %s buildings available for HH %s, using fallback",
                    target_hustyp,
                    hh.household_id,
                )

        if len(compatible) > 0:
            selected_idx = random.choice(compatible.index.tolist())
            raw_bid = compatible.loc[selected_idx, building_id_col]
            bid_key = int(str(raw_bid))  # type: ignore[arg-type]
            hh.building_id = str(raw_bid)
            building_occupancy[bid_key] = building_occupancy.get(bid_key, 0) + 1
        else:
            logger.warning("No available buildings for household %s", hh.household_id)

    assigned = sum(1 for hh in households if hh.building_id is not None)
    logger.info("Linked %d/%d households to buildings", assigned, len(households))
    return households


# ── Building-unit estimation ─────────────────────────────────────────────

def estimate_building_units(
    building_area: float,
    building_height: float,
    floors: Optional[int] = None,
    avg_unit_area: float = 80.0,
    floor_height: float = 3.0,
) -> int:
    """Estimate dwelling units from building dimensions."""
    if floors is None:
        floors = max(1, int(building_height / floor_height))
    total_area = building_area * floors
    units = int(total_area / avg_unit_area)
    return max(1, units)
