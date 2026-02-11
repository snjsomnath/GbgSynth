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
    """Place couples in multi-person households.

    Supports both opposite-sex and same-sex couples.  The fraction of
    same-sex couples is controlled by the ``same_sex_couple_fraction``
    constraint (default 2.5%, matching Swedish registry statistics).

    Uses a two-pointer sweep on age-sorted pools to *maximise* the number
    of couples formed.  The previous first-fit O(n²) approach exhausted
    candidate pools prematurely when age distributions did not align,
    causing Sammanboende (cohabiting) undercounts of 40-60%.

    Algorithm (inspired by GenSynthPop de Mooij et al., 2024):
    1. Sort males and females by age.
    2. For each male, find the closest-age available female within
       ``partner_age_difference_max`` (default 15) using a sliding-window
       pointer, preferring the female whose age is closest.
    3. Collect all valid pairs, then assign them to households that can
       fit ≥2 members.
    4. If ``allow_same_sex_couples`` is enabled, a fraction of pairs are
       formed as same-sex from remaining unmatched cohabiting adults.

    This separates *pairing* from *placement*, ensuring we maximise the
    number of formed couples regardless of household iteration order.

    Returns:
        Number of couples formed.
    """
    max_age_diff = constraints.get('partner_age_difference_max', 15)
    allow_same_sex = constraints.get('allow_same_sex_couples', False)
    ss_fraction = constraints.get('same_sex_couple_fraction', 0.025)

    males = sorted(
        [a for a in cohabiting if a.sex == 'male'],
        key=lambda x: x.age,
    )
    females = sorted(
        [a for a in cohabiting if a.sex == 'female'],
        key=lambda x: x.age,
    )

    if not males or not females:
        return 0

    # Count how many households can fit a couple
    hh_with_space = [hh for hh in multi_hh if hh.can_fit(2)]
    if not hh_with_space:
        return 0

    # --- Phase 1: build opposite-sex pairs using greedy closest-age matching ---
    paired_males: Set[int] = set()
    paired_females: Set[int] = set()
    pairs: List[tuple] = []  # (agent_a, agent_b)

    # For each male, find the closest compatible female
    fem_ptr = 0
    for male in males:
        best_female = None
        best_gap = max_age_diff + 1

        # Advance pointer to first female that could be compatible
        while fem_ptr < len(females) and females[fem_ptr].age < male.age - max_age_diff:
            fem_ptr += 1

        # Scan forward from pointer for all compatible females
        for j in range(fem_ptr, len(females)):
            female = females[j]
            if female.age > male.age + max_age_diff:
                break  # rest are too old
            if id(female) in paired_females:
                continue
            gap = abs(male.age - female.age)
            if gap < best_gap:
                best_gap = gap
                best_female = female

        if best_female is not None:
            pairs.append((male, best_female))
            paired_males.add(id(male))
            paired_females.add(id(best_female))

    # --- Phase 1b: form same-sex pairs from remaining cohabiting adults ---
    if allow_same_sex and ss_fraction > 0:
        total_couples = len(pairs)
        n_same_sex = max(1, int(round(total_couples * ss_fraction / (1 - ss_fraction))))

        unpaired_males = [a for a in males if id(a) not in paired_males]
        unpaired_females = [a for a in females if id(a) not in paired_females]

        ss_paired: Set[int] = set()
        ss_pairs: List[tuple] = []

        # Form male-male pairs
        mm_target = n_same_sex // 2
        unpaired_males.sort(key=lambda a: a.age)
        for i in range(0, len(unpaired_males) - 1, 2):
            if len(ss_pairs) >= mm_target:
                break
            a, b = unpaired_males[i], unpaired_males[i + 1]
            if abs(a.age - b.age) <= max_age_diff:
                ss_pairs.append((a, b))
                ss_paired.add(id(a))
                ss_paired.add(id(b))

        # Form female-female pairs
        ff_target = n_same_sex - len(ss_pairs)
        unpaired_females.sort(key=lambda a: a.age)
        for i in range(0, len(unpaired_females) - 1, 2):
            if len(ss_pairs) >= n_same_sex:
                break
            a, b = unpaired_females[i], unpaired_females[i + 1]
            if abs(a.age - b.age) <= max_age_diff:
                ss_pairs.append((a, b))
                ss_paired.add(id(a))
                ss_paired.add(id(b))

        pairs.extend(ss_pairs)
        if ss_pairs:
            logger.info(
                "Formed %d same-sex couples (%.1f%% of total)",
                len(ss_pairs),
                100 * len(ss_pairs) / len(pairs) if pairs else 0,
            )

    # --- Phase 2: assign pairs to households ---
    couples_formed = 0
    hh_idx = 0
    for agent_a, agent_b in pairs:
        if hh_idx >= len(hh_with_space):
            break
        hh = hh_with_space[hh_idx]
        hh.add_member(agent_a)
        hh.add_member(agent_b)
        couples_formed += 1
        hh_idx += 1

    if couples_formed < len(pairs):
        logger.info(
            "Formed %d couples but %d pairs had no household slot",
            couples_formed,
            len(pairs) - couples_formed,
        )

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
        random.Random().shuffle(shuffled_hh)  # local shuffle — non-critical path

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
                target = random.Random().choice(hh_with_adults)  # non-critical overflow
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
