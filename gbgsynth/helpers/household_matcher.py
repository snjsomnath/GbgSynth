"""
Constrained matching of individuals to household containers.

This module implements the matching phases of the top-down synthesis:
couple formation, child placement with biological constraints, single-parent
assignment, overflow redistribution, and children-only household repair.
"""

import logging
import random
from typing import Dict, List, Optional, Set

from gbgsynth.models import Agent, Household

logger = logging.getLogger(__name__)


# ── Couple formation ─────────────────────────────────────────────────────

def form_couples(
    cohabiting: List[Agent],
    multi_hh: List[Household],
    constraints: Dict,
) -> int:
    """Place male-female pairs in multi-person households.

    Matches couples using sorted pools and first-fit pairing.
    For each household needing a couple, the algorithm
    finds the first unplaced male and the first compatible female
    within ``partner_age_difference_max`` (default 15).

    Returns:
        Number of couples formed.
    """
    max_age_diff = constraints.get('partner_age_difference_max', 15)

    # Sort adults by age for better matching (matches original behaviour)
    males = sorted([a for a in cohabiting if a.sex == 'male'], key=lambda x: x.age)
    females = sorted([a for a in cohabiting if a.sex == 'female'], key=lambda x: x.age)

    couples_formed = 0

    for hh in multi_hh:
        if not hh.can_fit(2):
            continue

        # Find first compatible pair (first-fit, matching original logic)
        matched = False
        for male in males:
            if male.household_id is not None:
                continue
            for female in females:
                if female.household_id is not None:
                    continue
                if abs(male.age - female.age) <= max_age_diff:
                    hh.add_member(male)
                    hh.add_member(female)
                    couples_formed += 1
                    matched = True
                    break
            if matched:
                break

    return couples_formed


# ── Single-parent placement ─────────────────────────────────────────────

def place_single_parents(
    single_parents: List[Agent],
    multi_hh: List[Household],
) -> int:
    """Place single parents in empty multi-person households.

    Each single parent gets its own household (placed before children
    so children can later be matched to them).

    Returns:
        Number of single parents placed.
    """
    placed = 0

    parents = sorted(single_parents, key=lambda a: -a.age)

    for parent in parents:
        best_hh = None
        for hh in multi_hh:
            if len(hh.members) > 0:
                continue
            if not hh.can_fit():
                continue
            if hh.size >= 2:
                best_hh = hh
                break

        if best_hh:
            best_hh.add_member(parent)
            placed += 1

    return placed


# ── Child placement ──────────────────────────────────────────────────────

def place_children(
    children: List[Agent],
    multi_hh: List[Household],
    constraints: Dict,
) -> int:
    """Place children in households with suitable parents.

    Priorities (in order):
    1. Single-parent households
    2. Couple households
    3. Any household with a qualifying adult

    Constraints enforced:
    - Minimum parent-child age gap (default 18)
    - Maximum parent-child age gap (45 mothers / 65 fathers)
    - Sibling spacing preference (2–5 years)
    - Household must contain ≥1 adult with a parental role

    Returns:
        Number of children placed.
    """
    min_parent_gap = constraints.get('parent_child_age_gap_min', 18)
    max_mother_gap = constraints.get('parent_child_age_gap_max_mother', 45)
    max_father_gap = constraints.get('parent_child_age_gap_max_father', 65)

    def _is_suitable_parent(adult: Agent, child: Agent) -> bool:
        gap = adult.age - child.age
        if gap < min_parent_gap:
            return False
        max_gap = max_mother_gap if adult.sex == 'female' else max_father_gap
        return gap <= max_gap

    def _has_eligible_adult(hh: Household) -> bool:
        return any(
            m.age >= 18 and m.hh_role in ('cohabiting', 'single_parent')
            for m in hh.members
        )

    def _sibling_affinity(hh: Household, child: Agent) -> float:
        existing_children = [m for m in hh.members if m.age < 18]
        if not existing_children:
            return 0.0
        spacings = [abs(child.age - c.age) for c in existing_children]
        return sum(
            1.0 if 2 <= s <= 5 else 0.3 if s <= 8 else 0.05
            for s in spacings
        )

    children = sorted(children, key=lambda c: c.age)

    single_parent_hh = [
        hh for hh in multi_hh
        if any(m.hh_role == 'single_parent' for m in hh.members)
        and sum(1 for m in hh.members if m.hh_role == 'cohabiting') < 2
    ]
    couple_hh = [
        hh for hh in multi_hh
        if sum(1 for m in hh.members if m.hh_role == 'cohabiting') >= 2
    ]

    placed = 0

    # --- Pass 1: single-parent households ---
    remaining = list(children)
    still_remaining: List[Agent] = []
    for child in remaining:
        placed_this = False
        candidates = [
            hh for hh in single_parent_hh
            if hh.can_fit() and _has_eligible_adult(hh)
        ]
        candidates.sort(key=lambda hh: _sibling_affinity(hh, child), reverse=True)
        for hh in candidates:
            adults = [m for m in hh.members if _is_suitable_parent(m, child)]
            if adults:
                hh.add_member(child)
                placed += 1
                placed_this = True
                break
        if not placed_this:
            still_remaining.append(child)
    remaining = still_remaining

    # --- Pass 2: couple households ---
    still_remaining = []
    for child in remaining:
        placed_this = False
        candidates = [
            hh for hh in couple_hh
            if hh.can_fit() and _has_eligible_adult(hh)
        ]
        candidates.sort(key=lambda hh: _sibling_affinity(hh, child), reverse=True)
        for hh in candidates:
            adults = [m for m in hh.members if _is_suitable_parent(m, child)]
            if adults:
                hh.add_member(child)
                placed += 1
                placed_this = True
                break
        if not placed_this:
            still_remaining.append(child)
    remaining = still_remaining

    # --- Pass 3: any household with a qualifying adult ---
    still_remaining = []
    for child in remaining:
        placed_this = False
        for hh in multi_hh:
            if not hh.can_fit():
                continue
            if not _has_eligible_adult(hh):
                continue
            adults = [m for m in hh.members if _is_suitable_parent(m, child)]
            if adults:
                hh.add_member(child)
                placed += 1
                placed_this = True
                break
        if not placed_this:
            still_remaining.append(child)

    return placed


# ── Other / remaining / single placement ─────────────────────────────────

def place_other(
    other: List[Agent],
    multi_hh: List[Household],
) -> int:
    """Place 'other' role individuals (roommates, multi-gen, etc.)."""
    placed = 0
    for agent in other:
        for hh in multi_hh:
            if hh.can_fit():
                hh.add_member(agent)
                placed += 1
                break
    return placed


def fill_remaining_slots(
    remaining: List[Agent],
    multi_hh: List[Household],
) -> int:
    """Fill remaining slots in multi-person households."""
    placed = 0
    for agent in remaining:
        for hh in multi_hh:
            if hh.can_fit():
                hh.add_member(agent)
                placed += 1
                break
    return placed


def place_singles(
    singles: List[Agent],
    single_hh: List[Household],
) -> int:
    """Place single adults in 1-person households."""
    placed = 0
    for hh in single_hh:
        if not singles:
            break
        if hh.can_fit():
            agent = singles.pop()
            hh.add_member(agent)
            placed += 1
    return placed


# ── Overflow redistribution ──────────────────────────────────────────────

def redistribute_unplaced(
    unplaced: List[Agent],
    all_hh: List[Household],
) -> None:
    """Redistribute unplaced individuals into households.

    Adults are placed first (any household with capacity), then children
    (only into households that already have an adult).  Overflow is
    spread uniformly by bumping ``hh.size`` as needed.
    """
    children = [a for a in unplaced if a.hh_role == 'child' or a.age < 18]
    children_ids: Set[int] = set(id(c) for c in children)
    adults = [a for a in unplaced if id(a) not in children_ids]

    placed_count = 0
    still_unplaced_adults: List[Agent] = []
    still_unplaced_children: List[Agent] = []

    # Phase 1: adults
    for agent in adults:
        placed = False
        for hh in all_hh:
            if hh.can_fit():
                hh.add_member(agent)
                placed = True
                placed_count += 1
                break
        if not placed:
            still_unplaced_adults.append(agent)

    # Phase 2: children → households with adults only
    for child in children:
        placed = False
        for hh in all_hh:
            if not hh.can_fit():
                continue
            if any(m.age >= 18 for m in hh.members):
                hh.add_member(child)
                placed = True
                placed_count += 1
                break
        if not placed:
            still_unplaced_children.append(child)

    still_unplaced = still_unplaced_adults + still_unplaced_children

    if still_unplaced:
        logger.info(
            "Spreading %d overflow individuals across households "
            "(census privacy rounding adjustment)",
            len(still_unplaced),
        )

        overflow_adults = [a for a in still_unplaced if id(a) not in children_ids]
        overflow_children = [a for a in still_unplaced if id(a) in children_ids]

        shuffled_hh = list(all_hh)
        random.shuffle(shuffled_hh)

        for i, agent in enumerate(overflow_adults):
            hh = shuffled_hh[i % len(shuffled_hh)]
            hh.size += 1
            hh.add_member(agent)
            placed_count += 1

        hh_with_adults = [
            hh for hh in shuffled_hh if any(m.age >= 18 for m in hh.members)
        ]
        if not hh_with_adults:
            hh_with_adults = shuffled_hh

        for i, child in enumerate(overflow_children):
            hh = hh_with_adults[i % len(hh_with_adults)]
            hh.size += 1
            hh.add_member(child)
            placed_count += 1

        logger.info("Placed all %d overflow individuals", placed_count)


# ── Post-hoc repair ─────────────────────────────────────────────────────

def fix_children_only_households(
    agents: List[Agent],
    households: List[Household],
) -> List[Household]:
    """Move children out of households that have no adults.

    Returns:
        Pruned household list (empty households removed).
    """
    children_only_hh: List[Household] = []
    hh_with_adults: List[Household] = []

    for hh in households:
        if not hh.members:
            continue
        if any(m.age >= 18 for m in hh.members):
            hh_with_adults.append(hh)
        else:
            children_only_hh.append(hh)

    if not children_only_hh:
        return households

    affected = sum(len(hh.members) for hh in children_only_hh)
    logger.warning(
        "Found %d children-only households (%d children). "
        "Redistributing to family households.",
        len(children_only_hh),
        affected,
    )

    children_to_move: List[Agent] = []
    for hh in children_only_hh:
        children_to_move.extend(list(hh.members))
        for child in list(hh.members):
            hh.members.remove(child)
            child.household_id = None

    hh_with_adults.sort(key=lambda h: h.size - len(h.members), reverse=True)

    moved = 0
    for child in children_to_move:
        for hh in hh_with_adults:
            if hh.can_fit():
                hh.add_member(child)
                moved += 1
                break
        else:
            if hh_with_adults:
                target = random.choice(hh_with_adults)
                target.size += 1
                target.add_member(child)
                moved += 1

    pruned = [hh for hh in households if hh.members]

    logger.info(
        "Moved %d children to family households. Remaining households: %d",
        moved,
        len(pruned),
    )
    return pruned
