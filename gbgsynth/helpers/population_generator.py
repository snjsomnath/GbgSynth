"""
Individual generation from census marginals.

This module generates synthetic individuals (Agent objects) from census
population tables.  Two strategies are supported:

1. **Position data** (``60_FolkmHHStallning_PRI.px``) — yields *exact*
   role counts (cohabiting, single, child, …).
2. **Population data** with role *sampling* — fallback when position
   data is unavailable.

Helper functions for translating Swedish census labels to internal
values, age-group sampling, and role probability tables live here.
"""

import logging
import random
import re
from typing import Dict, List, Optional

import pandas as pd

from gbgsynth.config import Config
from gbgsynth.models import Agent

logger = logging.getLogger(__name__)


# ── Translation helpers ─────────────────────────────────────────────────
# All translations delegate to Config, which builds exact lookup dicts
# from table_mapping.json at init time.  No substring matching here.

_cfg = Config()


def translate_sex(sex_str, *, strict: bool = False) -> str:
    """Translate Swedish sex label to internal ``'male'``/``'female'``."""
    if pd.isna(sex_str):
        logger.warning("NaN sex value — defaulting to 'male'")
        if strict:
            raise ValueError("NaN sex value in population data")
        return 'male'

    key = str(sex_str).strip().lower()
    mapped = _cfg._sex_lookup.get(key)
    if mapped:
        return mapped
    logger.warning("Unrecognised sex value '%s' — defaulting to 'male'", sex_str)
    return 'male'


def translate_hh_role(role_str) -> str:
    """Translate aggregate household type from Swedish."""
    return _cfg.translate_hh_role(role_str)


def translate_hh_position(position_str) -> str:
    """Translate detailed household position from Swedish."""
    return _cfg.translate_position(position_str)


# ── Age helpers ──────────────────────────────────────────────────────────

def age_to_group(age: int) -> str:
    """Convert numeric age to census age-group string."""
    if age <= 5:
        return '0-5 år'
    elif age <= 15:
        return '6-15 år'
    elif age <= 18:
        return '16-18 år'
    elif age <= 24:
        return '19-24 år'
    elif age <= 34:
        return '25-34 år'
    elif age <= 44:
        return '35-44 år'
    elif age <= 54:
        return '45-54 år'
    elif age <= 64:
        return '55-64 år'
    elif age <= 74:
        return '65-74 år'
    elif age <= 84:
        return '75-84 år'
    else:
        return '85- år'


def sample_age_from_group(
    age_group,
    config: Config,
    *,
    strict: bool = False,
) -> int:
    """Sample a specific age uniformly from an age-group range.

    Tries the config's ``age_group_mappings`` first, then parses
    ``"X-Y år"`` / ``"X- år"`` patterns.
    """
    if pd.isna(age_group):
        logger.warning("NaN age group encountered — using random age 25–65")
        if strict:
            raise ValueError("NaN age group in population data")
        return random.randint(25, 65)

    age_mappings = config.age_group_mappings

    if age_group in age_mappings:
        range_dict = age_mappings[age_group]
        return random.randint(range_dict['min'], range_dict['max'])

    age_str = str(age_group).strip()

    match = re.match(r'(\d+)-(\d+)\s*år', age_str)
    if match:
        return random.randint(int(match.group(1)), int(match.group(2)))

    match = re.match(r'(\d+)[-+]\s*år', age_str)
    if match:
        min_age = int(match.group(1))
        return random.randint(min_age, min_age + 15)

    logger.warning("Could not parse age group '%s' — using random age 25–65", age_group)
    if strict:
        raise ValueError(f"Unparseable age group: '{age_group}'")
    return random.randint(25, 65)


# ── Role probability table ──────────────────────────────────────────────

def build_role_probability_table(
    position_data: pd.DataFrame,
    *,
    strict: bool = False,
) -> Dict:
    """Build a ``{(age_group, sex): {role: probability}}`` lookup.

    Returns:
        Dict keyed by ``(age_group_label, sex_en)``.
    """
    role_probs: Dict = {}

    age_col = 'Ålder' if 'Ålder' in position_data.columns else 'age_group'
    sex_col = 'Kön' if 'Kön' in position_data.columns else 'sex'
    pos_col = (
        'Hushållsställning'
        if 'Hushållsställning' in position_data.columns
        else 'hh_position'
    )
    count_col = 'Antal' if 'Antal' in position_data.columns else 'count'

    if count_col not in position_data.columns:
        for col in position_data.columns:
            if position_data[col].dtype in ('int64', 'float64'):
                count_col = col
                break

    for (age_grp, sex), group in position_data.groupby([age_col, sex_col]):
        role_counts: Dict[str, int] = {}
        total = 0

        for _, row in group.iterrows():
            position = row.get(pos_col, '')
            count = int(row.get(count_col, 0)) if pd.notna(row.get(count_col)) else 0
            if count <= 0:
                continue

            role = translate_hh_position(position)
            if role != 'unknown':
                role_counts[role] = role_counts.get(role, 0) + count
                total += count

        if total > 0:
            sex_eng = translate_sex(sex, strict=strict)
            role_probs[(age_grp, sex_eng)] = {
                r: c / total for r, c in role_counts.items()
            }

    logger.info("Built role probability table for %d age/sex groups", len(role_probs))
    return role_probs


def sample_role_for_agent(
    age: int,
    sex: str,
    role_probs: Dict,
) -> str:
    """Sample a household role using census probabilities."""
    if not role_probs:
        return 'child' if age < 18 else random.choice(['single', 'cohabiting'])

    age_group_label = age_to_group(age)
    probs = role_probs.get((age_group_label, sex))

    if not probs:
        # Try alternative sex labels
        for key, p in role_probs.items():
            if key[0] == age_group_label:
                probs = p
                break

    if not probs:
        return 'child' if age < 18 else random.choice(['single', 'cohabiting'])

    roles = list(probs.keys())
    weights = [probs[r] for r in roles]
    return random.choices(roles, weights=weights, k=1)[0]


# ── Individual generators ────────────────────────────────────────────────

def generate_individuals_from_position_data(
    position_data: pd.DataFrame,
    config: Config,
    start_id: int = 1,
    *,
    strict: bool = False,
) -> tuple:
    """Generate individuals with *exact* role counts from position data.

    Args:
        position_data: DataFrame from ``60_FolkmHHStallning_PRI.px``.
        config: Config object (for age-group mappings).
        start_id: Starting agent ID.
        strict: Raise on parse errors.

    Returns:
        ``(pool, next_id)`` — list of :class:`Agent` and next unused ID.
    """
    age_col = 'Ålder' if 'Ålder' in position_data.columns else 'age_group'
    sex_col = 'Kön' if 'Kön' in position_data.columns else 'sex'
    pos_col = (
        'Hushållsställning'
        if 'Hushållsställning' in position_data.columns
        else 'hh_position'
    )
    count_col = 'Antal' if 'Antal' in position_data.columns else 'count'

    if count_col not in position_data.columns:
        for col in position_data.columns:
            if position_data[col].dtype in ('int64', 'float64'):
                count_col = col
                break

    pool: List[Agent] = []
    role_counts: Dict[str, int] = {}
    next_id = start_id

    for _, row in position_data.iterrows():
        count = int(row[count_col]) if pd.notna(row[count_col]) else 0
        if count <= 0:
            continue

        age_group = row[age_col]
        sex_label = row[sex_col]
        position = row[pos_col]

        sex = translate_sex(sex_label, strict=strict)
        hh_role = translate_hh_position(position)

        if hh_role == 'unknown':
            continue

        role_counts[hh_role] = role_counts.get(hh_role, 0) + count

        for _ in range(count):
            age = sample_age_from_group(age_group, config, strict=strict)
            agent = Agent(agent_id=next_id, age=age, sex=sex, hh_role=hh_role)
            pool.append(agent)
            next_id += 1

    logger.info("Exact role counts from position data: %s", role_counts)
    return pool, next_id


def generate_individuals_from_population_data(
    population_data: pd.DataFrame,
    config: Config,
    start_id: int = 1,
    role_probs: Optional[Dict] = None,
    *,
    strict: bool = False,
) -> tuple:
    """Generate individuals from population data with role sampling.

    This is the fallback when detailed position data is unavailable.

    Args:
        population_data: DataFrame with age/sex/role counts.
        config: Config object.
        start_id: Starting agent ID.
        role_probs: Optional role-probability lookup built by
            :func:`build_role_probability_table`.
        strict: Raise on parse errors.

    Returns:
        ``(pool, next_id)``.
    """
    age_col = 'Ålder' if 'Ålder' in population_data.columns else 'age_group'
    sex_col = 'Kön' if 'Kön' in population_data.columns else 'sex'
    role_col = (
        'Hushållstyp'
        if 'Hushållstyp' in population_data.columns
        else 'hh_role'
    )
    count_col = (
        'Antal'
        if 'Antal' in population_data.columns
        else population_data.columns[-1]
    )

    pool: List[Agent] = []
    next_id = start_id

    for _, row in population_data.iterrows():
        count = int(row[count_col]) if pd.notna(row[count_col]) else 0
        if count <= 0:
            continue

        age_group = row[age_col]
        sex_label = row[sex_col]
        role_label = row[role_col]

        sex = translate_sex(sex_label, strict=strict)

        for _ in range(count):
            age = sample_age_from_group(age_group, config, strict=strict)

            if role_probs:
                hh_role = sample_role_for_agent(age, sex, role_probs)
            else:
                hh_role = translate_hh_role(role_label)
                if age < 18 and hh_role != 'cohabiting':
                    hh_role = 'child'

            agent = Agent(agent_id=next_id, age=age, sex=sex, hh_role=hh_role)
            pool.append(agent)
            next_id += 1

    return pool, next_id
