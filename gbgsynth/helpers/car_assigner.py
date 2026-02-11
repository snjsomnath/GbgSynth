"""
Car ownership assignment using a propensity-based model.

The propensity model distributes the census-reported total number of
cars across households, using socio-economic scoring.
"""

import logging
import random
from typing import Dict, List, Optional

import pandas as pd

from gbgsynth.models import Agent, Household

logger = logging.getLogger(__name__)


def assign_cars_simple(
    house_type: str,
    hh_size: int,
    car_data: Optional[pd.DataFrame],
) -> int:
    """Stub car assignment — always returns 0.

    Actual assignment happens via :func:`assign_cars_propensity`.
    Retained for backward compatibility with the IPF engines.
    """
    return 0


def assign_cars_propensity(
    households: List[Household],
    agents: List[Agent],
    car_data: Optional[pd.DataFrame],
    constraints: Dict,
    *,
    rng: Optional[random.Random] = None,
) -> None:
    """Assign cars to households using propensity scoring.

    The exact total car count from the census is distributed to
    households ranked by socio-economic propensity.

    Args:
        households: All synthesized households.
        agents: All synthesized agents (used for fallback pop count).
        car_data: DataFrame with ``Personbilar`` count.
        constraints: Constraints dict (for ``max_cars_per_household``).
        rng: Optional local ``random.Random`` instance.
            Falls back to module-level ``random`` if not provided.
    """
    _rng = rng or random
    if not households:
        return

    # --- Target car count ---
    total_cars_target = 0
    if car_data is not None and not car_data.empty:
        car_row = car_data[car_data['Tabellvärde'] == 'Personbilar']
        if not car_row.empty:
            count_col = (
                'NoContent'
                if 'NoContent' in car_data.columns
                else car_data.columns[-1]
            )
            total_cars_target = int(car_row[count_col].iloc[0])

    if total_cars_target == 0:
        total_pop = len(agents)
        total_cars_target = int(total_pop * 0.19)
        logger.info("No car data — estimating %d cars based on population", total_cars_target)
    else:
        logger.info("Target car count from census: %d", total_cars_target)

    # --- Propensity scores ---
    hh_scores = []
    for hh in households:
        score = 1.0  # Base propensity

        # 1. Housing Type Factor (strongest predictor)
        hustyp = hh.assigned_hustyp or ''
        if 'Småhus' in hustyp:
            score += 15.0  # Villas almost always have cars
        elif 'Specialbostad' in hustyp:
            score -= 3.0   # Student/elderly housing - lowest priority
        elif 'Flerbostadshus' in hustyp:
            score += 0.5   # Apartments - moderate

        # 2. Family Structure Factor
        has_young_children = any(m.age <= 5 for m in hh.members)
        has_school_children = any(6 <= m.age <= 15 for m in hh.members)
        if has_young_children:
            score += 4.0  # High priority - need car for daycare/errands
        if has_school_children:
            score += 2.5  # Moderate priority - school activities

        # 3. Income Factor
        has_low_income = any(getattr(m, 'low_income', False) for m in hh.members)
        if has_low_income:
            score -= 2.0
        else:
            incomes = [getattr(m, 'income', 0) or 0 for m in hh.members]
            avg_income = sum(incomes) / len(incomes) if incomes else 0
            if avg_income > 400000:
                score += 2.0
            elif avg_income > 300000:
                score += 1.0

        # 4. Household Size Factor
        if hh.size >= 3:
            score += 1.0
        elif hh.size == 1:
            score -= 0.5  # Singles less likely to own cars in urban areas

        # 5. Age Factor - working age adults more likely
        working_age_adults = sum(1 for m in hh.members if 25 <= m.age <= 64)
        score += working_age_adults * 0.5

        # Elderly single households less likely
        if hh.size == 1 and any(m.age >= 75 for m in hh.members):
            score -= 2.0

        hh_scores.append((hh, max(score, 0.1)))  # Ensure positive score

    hh_scores.sort(
        key=lambda x: x[1] * _rng.uniform(0.8, 1.2),
        reverse=True,
    )

    # --- Distribute ---
    cars_distributed = 0

    for hh, _score in hh_scores:
        if cars_distributed >= total_cars_target:
            break
        hh.cars = 1
        cars_distributed += 1

    max_cars = constraints.get('max_cars_per_household', 2)

    if cars_distributed < total_cars_target:
        for hh, score in hh_scores:
            if cars_distributed >= total_cars_target:
                break
            if score > 10.0 and hh.cars == 1:
                hh.cars = 2
                cars_distributed += 1

    remaining = total_cars_target - cars_distributed
    if remaining > 0:
        for hh, _score in hh_scores:
            if remaining <= 0:
                break
            if hh.cars < max_cars:
                hh.cars += 1
                remaining -= 1

    # --- Log ---
    total_assigned = sum(hh.cars for hh in households)
    hh_with_cars = sum(1 for hh in households if hh.cars > 0)
    if households:
        pct = 100 * hh_with_cars / len(households)
        logger.info(
            "Assigned %d cars to %d households (%.1f%% car ownership)",
            total_assigned, hh_with_cars, pct,
        )
    else:
        logger.warning("No households to assign cars to")
