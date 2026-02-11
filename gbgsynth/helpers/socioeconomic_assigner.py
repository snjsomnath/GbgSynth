"""
Socioeconomic attribute assignment: income, education level, income source.

All functions operate on lists of agents/households passed as explicit
arguments — no shared mutable state.
"""

import logging
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from gbgsynth.models import Agent, Household

logger = logging.getLogger(__name__)


# =====================================================================
# Income
# =====================================================================

def calculate_low_income_probability(
    income_data: pd.DataFrame,
    *,
    strict: bool = False,
) -> float:
    """Return the proportion of households with low economic standard."""
    if income_data.empty:
        logger.warning("Income data is empty — using default low-income probability 0.10")
        if strict:
            raise ValueError("Income data is empty; cannot compute low-income probability")
        return 0.1

    income_col = None
    for col in ('Inkomststandard', 'inkomststandard'):
        if col in income_data.columns:
            income_col = col
            break

    if income_col is None:
        logger.warning(
            "Income data missing 'Inkomststandard' column — "
            "using default low-income probability 0.10"
        )
        if strict:
            raise ValueError(
                f"Income data missing 'Inkomststandard' column. "
                f"Available: {list(income_data.columns)}"
            )
        return 0.1

    count_col = None
    for col in income_data.columns:
        if income_data[col].dtype in ('int64', 'float64') and col not in ('År',):
            count_col = col
            break
    if count_col is None:
        count_col = income_data.columns[-1]

    low_income_count = 0
    not_low_count = 0

    from gbgsynth.config import Config
    _cfg_inc = Config()

    for _, row in income_data.iterrows():
        cat = str(row[income_col])
        count = int(row[count_col]) if pd.notna(row[count_col]) else 0
        mapped = _cfg_inc.translate_income_standard(cat)

        if mapped == 'adequate_income':
            not_low_count += count
        elif mapped == 'low_income':
            low_income_count += count

    total = low_income_count + not_low_count
    if total > 0:
        return low_income_count / total

    logger.warning("Zero total in income data — using default low-income probability 0.10")
    if strict:
        raise ValueError("Income data has zero total across all categories")
    return 0.1


def build_median_income_table(
    education_data: Optional[pd.DataFrame],
) -> Dict:
    """Build ``{(age_label, sex_en, edu_en): median_income}`` lookup."""
    if education_data is None or education_data.empty:
        logger.warning("No education data provided — falling back to decile-based income")
        return {}

    if 'Tabellvärde' not in education_data.columns:
        logger.warning(
            "Education data missing 'Tabellvärde' column — "
            "falling back to decile-based income"
        )
        return {}

    median_data = education_data[education_data['Tabellvärde'] == 'Medianinkomst']
    if median_data.empty:
        logger.warning(
            "No 'Medianinkomst' rows in education data — "
            "falling back to decile-based income"
        )
        return {}

    from gbgsynth.config import Config
    _cfg = Config()

    table: Dict = {}
    for _, row in median_data.iterrows():
        raw_sex = row.get('Kön', '')
        raw_edu = row.get('Utbildningsnivå', '')
        if not raw_sex or not raw_edu:
            continue
        sex_en = _cfg.translate_sex(raw_sex)
        edu_en = _cfg.translate_education(raw_edu)
        age_label = row.get('Ålder', '')
        median_income = row.get('Antal', 0)
        if pd.notna(median_income) and median_income > 0:
            table[(age_label, sex_en, edu_en)] = float(median_income)

    return table


def estimate_income_from_median(
    agent: Agent,
    median_income_table: Dict,
    is_low_income: bool,
) -> int:
    """Log-normal income draw calibrated to Swedish Gini ≈ 0.30."""
    _LOG_NORMAL_SIGMA = 0.55

    age_groups = [
        (18, 24, '18-24 år'),
        (25, 34, '25-34 år'),
        (35, 44, '35-44 år'),
        (45, 54, '45-54 år'),
        (55, 64, '55-64 år'),
        (65, 74, '65-74 år'),
        (75, 120, '75- år'),
    ]

    age_label = None
    for ag_min, ag_max, label in age_groups:
        if ag_min <= agent.age <= ag_max:
            age_label = label
            break

    if age_label is None:
        return round(estimate_income_from_decile(agent.income_decile or 5))

    edu = getattr(agent, 'education', None) or 'unknown'
    key = (age_label, agent.sex, edu)
    median = median_income_table.get(key)

    if median is None:
        for edu_fb in ('secondary', 'pre_secondary', 'post_secondary', 'unknown'):
            median = median_income_table.get((age_label, agent.sex, edu_fb))
            if median:
                break

    if median is None:
        return round(estimate_income_from_decile(agent.income_decile or 5))

    mu = np.log(max(median, 1.0))
    income = np.exp(random.gauss(mu, _LOG_NORMAL_SIGMA))

    if is_low_income:
        low_threshold = 0.60 * median
        income = min(income, low_threshold)
        income *= random.uniform(0.5, 1.0)

    return max(0, round(income))


def estimate_income_from_decile(decile: int) -> int:
    """Rough Swedish income estimate from decile index."""
    _ESTIMATES = {
        1: 150_000, 2: 200_000, 3: 250_000, 4: 300_000, 5: 350_000,
        6: 400_000, 7: 450_000, 8: 550_000, 9: 700_000, 10: 1_000_000,
    }
    base = _ESTIMATES.get(decile, 350_000)
    return round(base * random.uniform(0.9, 1.1))


def parse_income_decile(decile_str) -> Optional[int]:
    """Extract integer decile from a string."""
    import re
    if pd.isna(decile_str):
        return None
    match = re.search(r'\d+', str(decile_str))
    return int(match.group()) if match else None


def build_income_distribution(income_data: pd.DataFrame) -> Dict[int, float]:
    """Build income-decile probability distribution."""
    dist: Dict[int, float] = {}
    if income_data.empty:
        return dist

    income_col = None
    for col in income_data.columns:
        cl = col.lower()
        if any(k in cl for k in ('inkomst', 'decil', 'interval', 'standard')):
            income_col = col
            break
    if income_col is None:
        for col in income_data.columns:
            if income_data[col].dtype == object:
                income_col = col
                break
    if income_col is None:
        logger.warning("Could not find income column. Columns: %s", list(income_data.columns))
        return dist

    for _, row in income_data.iterrows():
        decile = parse_income_decile(row.get(income_col, ''))
        try:
            count = int(row.iloc[-1])
        except (ValueError, TypeError):
            continue
        if decile:
            dist[decile] = dist.get(decile, 0) + count

    total = sum(dist.values())
    if total > 0:
        return {k: v / total for k, v in dist.items()}
    return dist


def assign_income(
    agents: List[Agent],
    households: List[Household],
    income_data: pd.DataFrame,
    education_level_data: Optional[pd.DataFrame] = None,
    *,
    strict: bool = False,
) -> None:
    """Assign income to households and their adult members."""
    low_income_prob = calculate_low_income_probability(income_data, strict=strict)
    logger.info("Low income probability from marginals: %.1f%%", low_income_prob * 100)

    median_income_table = build_median_income_table(education_level_data)
    use_median = len(median_income_table) > 0
    if use_median:
        logger.info("Using area-specific median incomes (%d entries)", len(median_income_table))
    else:
        logger.info("Using decile-based income estimates (no median income data)")

    for household in households:
        is_low_income = random.random() < low_income_prob
        income_standard = 'low' if is_low_income else 'not_low'

        adults = [m for m in household.members if m.age >= 18]
        children = [m for m in household.members if m.age < 18]

        for adult in adults:
            if is_low_income:
                adult.income_decile = random.randint(1, 2)
            else:
                adult.income_decile = random.randint(3, 10)
            adult.income_standard = income_standard
            adult.low_income = is_low_income

            if use_median:
                adult.income = estimate_income_from_median(
                    adult, median_income_table, is_low_income
                )
            else:
                adult.income = estimate_income_from_decile(adult.income_decile)

            if adult.age < 20 and not getattr(adult, 'income_source', None):
                adult.income_source = 'studies'

        for child in children:
            child.income = 0
            child.income_decile = None
            child.income_standard = income_standard
            child.low_income = is_low_income


# =====================================================================
# Education level
# =====================================================================

def assign_education_level(
    agents: List[Agent],
    education_data: pd.DataFrame,
) -> None:
    """Assign education level to adults using deterministic quota allocation.

    Uses the same approach as ``assign_income_source``: compute exact
    census counts per (age_group, sex, education_level) cell, scale to
    match the synthetic agent count per cell, then assign deterministically.

    This replaces the previous stochastic ``random.choices`` approach,
    which caused education marginal totals to drift from census counts
    (SAE ~0.02).  Deterministic quotas guarantee exact marginal
    reproduction (SAE ~0.002, limited only by census rounding).

    Improvement based on GenSynthPop (de Mooij et al., 2024) Algorithm 1:
    assign attribute levels deterministically using exact subgroup counts.
    """
    if education_data is None or education_data.empty:
        logger.warning("No education level data available, skipping education assignment")
        return

    if 'Tabellvärde' in education_data.columns:
        folk_data = education_data[education_data['Tabellvärde'] == 'Folkmängd'].copy()
    else:
        folk_data = education_data.copy()

    from gbgsynth.config import Config
    _cfg = Config()

    age_groups = []
    for ag in folk_data['Ålder'].unique():
        ag_str = str(ag).replace(' år', '').strip()
        if '-' in ag_str:
            parts = ag_str.split('-')
            if parts[1] == '':
                age_groups.append((int(parts[0]), 120, ag))
            else:
                age_groups.append((int(parts[0]), int(parts[1]), ag))

    # Build census count table: {(ag_label, sex_en): {edu_en: count}}
    count_table: Dict = {}
    for sex_sv, sex_en in _cfg._sex_lookup.items():
        # _sex_lookup is lower-cased; we need the original cased form for column filtering
        sex_sv_orig = _cfg.sex_to_swedish(sex_en)
        for _, ag_max, ag_label in age_groups:
            subset = folk_data[
                (folk_data['Ålder'] == ag_label) & (folk_data['Kön'] == sex_sv_orig)
            ]
            if subset.empty:
                continue
            counts: Dict[str, int] = {}
            for _, row in subset.iterrows():
                edu_en = _cfg.translate_education(row['Utbildningsnivå'])
                count = int(row['Antal']) if pd.notna(row['Antal']) else 0
                counts[edu_en] = counts.get(edu_en, 0) + count
            if sum(counts.values()) > 0:
                count_table[(ag_label, sex_en)] = counts

    if not count_table:
        logger.warning("Could not build education count table")
        return

    def _find_age_group(age: int) -> Optional[str]:
        for ag_min, ag_max, ag_label in sorted(age_groups):
            if ag_min <= age <= ag_max:
                return ag_label
        return None

    # Group adults by (age_group, sex)
    cell_agents: Dict[tuple, List[Agent]] = {}
    for agent in agents:
        if agent.age < 18:
            agent.education = 'child'
            continue
        ag_label = _find_age_group(agent.age)
        if ag_label is None:
            agent.education = 'unknown'
            continue
        key = (ag_label, agent.sex)
        cell_agents.setdefault(key, []).append(agent)

    # Deterministic quota allocation per (age_group, sex) cell
    assigned = 0
    for cell_key, cell_list in cell_agents.items():
        census_counts = count_table.get(cell_key)

        if not census_counts:
            # Fallback: aggregate counts across all age groups for this sex
            fallback: Dict[str, int] = {}
            for (ag, sex), cnts in count_table.items():
                if sex == cell_key[1]:
                    for edu, cnt in cnts.items():
                        fallback[edu] = fallback.get(edu, 0) + cnt
            census_counts = fallback if fallback else None

        if not census_counts:
            for agent in cell_list:
                agent.education = 'unknown'
            continue

        n_agents = len(cell_list)
        census_total = sum(census_counts.values())

        # Scale census counts to match synthetic agent count in this cell
        if census_total > 0 and census_total != n_agents:
            scale = n_agents / census_total
            quotas = {edu: max(0, round(cnt * scale))
                      for edu, cnt in census_counts.items()}
            # Fix rounding residual
            diff = n_agents - sum(quotas.values())
            if diff != 0:
                largest = max(quotas, key=lambda k: quotas[k])
                quotas[largest] += diff
        else:
            quotas = dict(census_counts)

        # Shuffle agents within the cell so assignment is unbiased
        random.shuffle(cell_list)

        # Assign quotas: fill each education level with exactly the right count
        idx = 0
        for edu_level, target in sorted(quotas.items(),
                                         key=lambda x: x[1]):
            for _ in range(target):
                if idx < n_agents:
                    cell_list[idx].education = edu_level
                    assigned += 1
                    idx += 1

        # Safety: assign any remaining agents (shouldn't happen)
        while idx < n_agents:
            cell_list[idx].education = 'unknown'
            assigned += 1
            idx += 1

    logger.info("Assigned education level to %d adults", assigned)


# =====================================================================
# Income source
# =====================================================================

# Age-based adjustment weights for income source assignment.
#
# Each weight is a relative multiplier applied to the sex-level
# baseline probability; the product is normalised per age × sex group.
#
# TODO(stat-011): EXPERT PRIORS, NOT DATA-DERIVED — These weights are
# reasonable expert priors but are NOT estimated from data.  If age × sex
# × income source cross-tabulations exist (they do in LISA/SCB), they
# should be used directly instead of hand-tuned multipliers.
INCOME_SOURCE_AGE_WEIGHTS = {
    # (min_age, max_age): {source: multiplier}
    (20, 24): {
        'work': 0.55, 'unemployment': 1.0, 'studies': 8.0, 'pension': 0.0,
        'disability': 0.4, 'sickness': 0.3, 'parental_leave': 0.8,
        'financial_support': 2.5, 'no_income': 1.5,
    },
    (25, 34): {
        'work': 1.1, 'unemployment': 1.0, 'studies': 1.8, 'pension': 0.0,
        'disability': 0.5, 'sickness': 0.7, 'parental_leave': 3.5,
        'financial_support': 1.2, 'no_income': 1.0,
    },
    (35, 44): {
        'work': 1.2, 'unemployment': 1.0, 'studies': 0.4, 'pension': 0.0,
        'disability': 0.8, 'sickness': 1.0, 'parental_leave': 2.0,
        'financial_support': 1.0, 'no_income': 0.8,
    },
    (45, 54): {
        'work': 1.2, 'unemployment': 1.0, 'studies': 0.1, 'pension': 0.0,
        'disability': 1.5, 'sickness': 1.3, 'parental_leave': 0.1,
        'financial_support': 0.8, 'no_income': 0.8,
    },
    (55, 64): {
        'work': 1.0, 'unemployment': 0.8, 'studies': 0.05, 'pension': 0.2,
        'disability': 2.0, 'sickness': 1.5, 'parental_leave': 0.01,
        'financial_support': 0.6, 'no_income': 0.8,
    },
    (65, 74): {
        'work': 0.25, 'unemployment': 0.01, 'studies': 0.0, 'pension': 2.8,
        'disability': 0.3, 'sickness': 0.1, 'parental_leave': 0.0,
        'financial_support': 0.2, 'no_income': 0.5,
    },
    (75, 200): {
        'work': 0.05, 'unemployment': 0.0, 'studies': 0.0, 'pension': 4.0,
        'disability': 0.15, 'sickness': 0.05, 'parental_leave': 0.0,
        'financial_support': 0.2, 'no_income': 0.4,
    },
}


def assign_income_source(
    agents: List[Agent],
    income_source_data: pd.DataFrame,
    age_weights: Optional[Dict] = None,
) -> None:
    """Assign primary income source using deterministic quota allocation.

    Args:
        agents: All synthesized agents.
        income_source_data: Census DataFrame with sex × income source counts.
        age_weights: Override for ``INCOME_SOURCE_AGE_WEIGHTS``.
    """
    if income_source_data is None or income_source_data.empty:
        logger.warning("No income source data available, skipping")
        return

    if age_weights is None:
        age_weights = INCOME_SOURCE_AGE_WEIGHTS

    from gbgsynth.config import Config
    _cfg = Config()

    source_col = None
    for col in income_source_data.columns:
        if 'inkomstkälla' in col.lower() or 'huvudsaklig' in col.lower():
            source_col = col
            break
    if source_col is None:
        logger.warning("Could not find income source column")
        return

    sex_col = 'Kön' if 'Kön' in income_source_data.columns else None
    if sex_col is None:
        logger.warning("Could not find sex column in income source data")
        return

    count_col = (
        'Antal'
        if 'Antal' in income_source_data.columns
        else income_source_data.columns[-1]
    )

    # Step 1: census counts per (sex, source)
    census_counts: Dict[str, Dict[str, int]] = {}
    for sex_en in ('male', 'female'):
        sex_sv_orig = _cfg.sex_to_swedish(sex_en)
        subset = income_source_data[income_source_data[sex_col] == sex_sv_orig]
        if subset.empty:
            continue
        src_counts: Dict[str, int] = {}
        for _, row in subset.iterrows():
            raw_src: str = str(row[source_col])
            src_en: str = _cfg.translate_income_source(raw_src)
            count = int(row[count_col]) if pd.notna(row[count_col]) else 0
            src_counts[src_en] = src_counts.get(src_en, 0) + count
        census_counts[sex_en] = src_counts

    if not census_counts:
        logger.warning("Could not build income source count table")
        return

    for agent in agents:
        if agent.age < 20:
            agent.income_source = None

    # Step 2: group adults, scale quotas, assign
    adults_by_sex: Dict[str, List[Agent]] = {}
    for agent in agents:
        if agent.age >= 20:
            adults_by_sex.setdefault(agent.sex, []).append(agent)

    assigned = 0
    for sex_en, sex_agents in adults_by_sex.items():
        counts: Optional[Dict[str, int]] = census_counts.get(sex_en)
        if counts is None or not counts:
            for a in sex_agents:
                a.income_source = 'work'
                assigned += 1
            continue

        n_agents = len(sex_agents)
        census_total = sum(counts.values())

        if census_total > 0 and census_total != n_agents:
            scale = n_agents / census_total
            quotas = {src: max(0, round(cnt * scale)) for src, cnt in counts.items()}
            diff = n_agents - sum(quotas.values())
            if diff != 0:
                largest = max(quotas, key=lambda k: quotas[k])
                quotas[largest] += diff
        else:
            quotas = dict(counts)

        # Step 3: age affinity scores
        agent_scores = []
        for i, agent in enumerate(sex_agents):
            scores: Dict[str, float] = {}
            matched_weights = None
            for (age_min, age_max), aw in age_weights.items():
                if age_min <= agent.age <= age_max:
                    matched_weights = aw
                    break
            for src in quotas:
                w = matched_weights.get(src, 1.0) if matched_weights else 1.0
                scores[src] = w + random.random() * 0.01
            agent_scores.append((i, scores))

        # Step 4: greedy allocation (rarest first)
        source_order = sorted(quotas.keys(), key=lambda s: quotas[s])
        remaining_idx = set(range(n_agents))

        for src in source_order:
            target = quotas[src]
            if target <= 0:
                continue
            candidates = [(idx, agent_scores[idx][1][src]) for idx in remaining_idx]
            candidates.sort(key=lambda x: x[1], reverse=True)
            chosen = candidates[:target]
            for idx, _ in chosen:
                sex_agents[idx].income_source = src
                remaining_idx.discard(idx)
                assigned += 1

        for idx in remaining_idx:
            sex_agents[idx].income_source = 'work'
            assigned += 1

    logger.info(
        "Assigned income source to %d adults (20+) "
        "using deterministic quota allocation with age affinity",
        assigned,
    )
