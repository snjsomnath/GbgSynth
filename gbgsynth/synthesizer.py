"""
Population synthesis engine with pluggable algorithms.

This module contains the core :class:`PopulationSynthesizer` class —
a thin orchestrator that delegates to six focused sub-modules:

- ``household_factory``       — Parse census tables, create HH containers
- ``population_generator``    — Generate individuals from marginals
- ``household_matcher``       — Constrained assignment (couples, children, …)
- ``socioeconomic_assigner``  — Income, education, income source
- ``car_assigner``            — Propensity-based car ownership
- ``housing_assigner``        — Hustyp assignment, building linkage

Three synthesis engines are available (selected via the ``engine`` parameter):

- **topdown** (default) — top-down constrained matching.  Fast, exact
  household-size distribution, but treats marginals as independent.
- **ipf** — Iterative Proportional Fitting via ``gbgsynth.ipf``.
- **constrained_ipf** — Constrained IPF with valid-by-construction
  household archetypes via ``gbgsynth.ipf``.
"""

import random
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

from gbgsynth.models import Agent, Household
from gbgsynth.config import Config

# ── Decomposed sub-modules (arch-001) ──────────────────────────────────
from gbgsynth.helpers import household_factory as _hf
from gbgsynth.helpers import population_generator as _pg
from gbgsynth.helpers import household_matcher as _hm
from gbgsynth.helpers import socioeconomic_assigner as _se
from gbgsynth.helpers import car_assigner as _ca
from gbgsynth.helpers import housing_assigner as _ha

logger = logging.getLogger(__name__)


class PopulationSynthesizer:
    """
    Population synthesizer using top-down constrained matching.

    The algorithm proceeds in phases:
    1. Create exact household containers from census size distribution
    2. Generate individuals from age/sex/role marginals
    3. Form couples in multi-person households
    4. Assign children to family households
    5. Fill remaining slots and handle overflow
    6. Assign socioeconomic attributes (income, cars)
    """

    # arch-003: supported synthesis engines
    VALID_ENGINES = {'topdown', 'ipf', 'constrained_ipf'}

    def __init__(self, config: Optional[Config] = None,
                 random_seed: Optional[int] = None,
                 strict: bool = False,
                 engine: str = 'topdown'):
        """
        Initialize the synthesizer.

        Args:
            config: Configuration object (will create default if None)
            random_seed: Optional seed for reproducible synthesis.
                If provided, local ``random.Random()`` and
                ``np.random.default_rng()`` instances are created at the
                start of every ``synthesize()`` invocation, ensuring
                repeated runs with the same seed yield identical
                synthetic populations without affecting global RNG state.
            strict: If True, raise ``ValueError`` on data quality issues
                instead of logging warnings and falling back to defaults.
            engine: Synthesis algorithm to use.  One of:
                - ``'topdown'`` (default) — top-down constrained matching.
                  Fast, exact household-size distribution, but treats
                  marginals as independent.
                - ``'ipf'`` — Iterative Proportional Fitting.
                  Better joint-distribution preservation.
                - ``'constrained_ipf'`` — Constrained IPF with valid-by-
                  construction household archetypes.  Statistically the
                  most rigorous, but requires ``gbgsynth.ipf`` module.
        """
        if engine not in self.VALID_ENGINES:
            raise ValueError(
                f"Unknown engine '{engine}'. "
                f"Choose from {sorted(self.VALID_ENGINES)}"
            )
        self.config = config or Config()
        self.constraints = self.config.constraints
        self.random_seed = random_seed
        self.strict = strict
        self.engine = engine

        # Synthesis state
        self.agents: List[Agent] = []
        self.households: List[Household] = []
        self.next_agent_id = 1
        self.next_household_id = 1

        # Synthesis statistics
        self.stats: Dict = {}

        # Local RNG instances — avoids mutating global random state.
        # Created fresh at each synthesize() call when a seed is provided.
        self._rng: random.Random = random.Random()
        self._np_rng: np.random.Generator = np.random.default_rng()

        # Declared up-front so the class interface is transparent (eng-003)
        self._household_position_data: Optional[pd.DataFrame] = None
        self._role_probs: Dict = {}
        self.ipf_stats: Dict = {}

    def synthesize(
        self,
        population_data: pd.DataFrame,
        household_data: pd.DataFrame,
        income_data: Optional[pd.DataFrame] = None,
        car_data: Optional[pd.DataFrame] = None,
        buildings: Optional[pd.DataFrame] = None,
        household_position_data: Optional[pd.DataFrame] = None,
        education_level_data: Optional[pd.DataFrame] = None,
        income_source_data: Optional[pd.DataFrame] = None
    ) -> Tuple[List[Agent], List[Household]]:
        """
        Generate synthetic population from census marginals.

        Args:
            population_data: DataFrame with age/sex/hh_role counts
            household_data: DataFrame with household size/type counts
            income_data: Optional DataFrame with income distribution
            car_data: Optional DataFrame with car ownership
            buildings: Optional DataFrame/GeoDataFrame with building footprints
            household_position_data: Optional DataFrame with detailed household
                      positions (including child role) by age/sex
            education_level_data: Optional DataFrame with education level counts
                      and income statistics by age group and sex
                      (from 23_InkomsterUtbildning table)
            income_source_data: Optional DataFrame with primary income source
                      distribution by sex (from 20_HuvudInk table)

        Returns:
            Tuple of (agents, households)
        """
        logger.info("Starting population synthesis")

        # ── State reset (eng-004) ────────────────────────────────────────
        self.agents = []
        self.households = []
        self.next_agent_id = 1
        self.next_household_id = 1
        self.stats = {}
        self.ipf_stats = {}
        self._household_position_data = None
        self._role_probs = {}

        # ── Seed RNGs for reproducibility (eng-001) ─────────────────────
        if self.random_seed is not None:
            self._rng = random.Random(self.random_seed)
            self._np_rng = np.random.default_rng(self.random_seed)
            logger.info(f"RNG seeded with {self.random_seed} for reproducibility")
        else:
            self._rng = random.Random()
            self._np_rng = np.random.default_rng()

        # ── Validate inputs (eng-005) ───────────────────────────────────
        self._validate_inputs(
            population_data, household_data,
            income_data=income_data,
            car_data=car_data,
            education_level_data=education_level_data,
            income_source_data=income_source_data,
        )

        # Store household position data for role assignment
        self._household_position_data = household_position_data
        if household_position_data is not None:
            self._role_probs = _pg.build_role_probability_table(household_position_data)
            logger.debug("Built role probability table from household position data")

        # ── Engine dispatch (arch-003) ─────────────────────────────────────
        if self.engine == 'ipf':
            logger.info("Using IPF synthesis engine")
            from gbgsynth.ipf import run_ipf_engine
            self.agents, self.households, self.next_agent_id, \
                self.next_household_id, self.ipf_stats = run_ipf_engine(
                    population_data, household_data, car_data,
                    self.constraints, self.config,
                    start_agent_id=self.next_agent_id,
                    start_household_id=self.next_household_id,
                )
        elif self.engine == 'constrained_ipf':
            logger.info("Using Constrained IPF synthesis engine")
            from gbgsynth.ipf import run_constrained_ipf_engine
            self.agents, self.households, self.next_agent_id, \
                self.next_household_id, self.ipf_stats = run_constrained_ipf_engine(
                    population_data, household_data, car_data,
                    self.constraints, self.config,
                    start_agent_id=self.next_agent_id,
                    start_household_id=self.next_household_id,
                )
        else:  # 'topdown'
            logger.info("Using top-down constrained synthesis engine")
            self._synthesize_topdown(population_data, household_data, car_data)
        
        logger.info(f"Matched {len(self.agents)} individuals to {len(self.households)} households")

        # Assign education levels from census distribution
        if education_level_data is not None:
            _se.assign_education_level(self.agents, education_level_data)

        # Assign income using education-based median incomes if available
        if income_data is not None:
            _se.assign_income(
                self.agents, self.households, income_data,
                education_level_data, strict=self.strict,
            )

        # Assign primary income source
        if income_source_data is not None:
            _se.assign_income_source(self.agents, income_source_data)

        # Assign housing types using size-conditioned distribution.
        _ha.assign_housing_types(household_data, self.config, self.households,
                                 rng=self._rng)

        # Assign cars using propensity model with exact target
        _ca.assign_cars_propensity(
            self.households, self.agents, car_data, self.constraints,
            rng=self._rng,
        )

        # Link to building footprints (if provided)
        if buildings is not None:
            _ha.link_to_buildings(buildings, self.households, rng=self._rng)
            logger.debug("Linked households to building footprints")

        # Validation
        self._validate_synthesis()

        return self.agents, self.households

    # =========================================================================
    # Input Validation (eng-005)
    # =========================================================================

    def _validate_inputs(
        self,
        population_data: pd.DataFrame,
        household_data: pd.DataFrame,
        *,
        income_data: Optional[pd.DataFrame] = None,
        car_data: Optional[pd.DataFrame] = None,
        education_level_data: Optional[pd.DataFrame] = None,
        income_source_data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Validate input DataFrames before synthesis.

        Checks:
        - Required DataFrames are not empty
        - Required columns exist
        - Count columns are non-negative
        - Population total ≈ household capacity total (within 10%)

        In strict mode (``self.strict=True``), raises ``ValueError``.
        Otherwise logs warnings and continues.
        """
        issues: List[str] = []

        # --- population_data ---
        if population_data is None or population_data.empty:
            issues.append("population_data is None or empty")
        else:
            # Check for a count column
            count_col = None
            for col in ['Antal', 'count']:
                if col in population_data.columns:
                    count_col = col
                    break
            if count_col is None:
                count_col = population_data.columns[-1]

            if population_data[count_col].dtype in ('int64', 'float64'):
                neg = (population_data[count_col] < 0).sum()
                if neg > 0:
                    issues.append(
                        f"population_data has {neg} negative counts in '{count_col}'"
                    )

        # --- household_data ---
        if household_data is None or household_data.empty:
            issues.append("household_data is None or empty")
        else:
            hh_size_col = (
                'Hushållsstorlek'
                if 'Hushållsstorlek' in household_data.columns
                else 'hh_size'
            )
            if hh_size_col not in household_data.columns:
                issues.append(
                    f"household_data missing size column "
                    f"(tried 'Hushållsstorlek' and 'hh_size'). "
                    f"Available: {list(household_data.columns)}"
                )

        # --- Cross-check population vs household capacity ---
        if (
            population_data is not None
            and not population_data.empty
            and household_data is not None
            and not household_data.empty
        ):
            try:
                pop_count_col = None
                for col in ['Antal', 'count']:
                    if col in population_data.columns:
                        pop_count_col = col
                        break
                if pop_count_col is None:
                    pop_count_col = population_data.columns[-1]
                total_pop = int(population_data[pop_count_col].sum())

                hh_count_col = None
                for col in ['Antal', 'count']:
                    if col in household_data.columns:
                        hh_count_col = col
                        break
                if hh_count_col is None:
                    hh_count_col = household_data.columns[-1]

                hh_size_col = (
                    'Hushållsstorlek'
                    if 'Hushållsstorlek' in household_data.columns
                    else 'hh_size'
                )
                if hh_size_col in household_data.columns:
                    total_capacity = 0
                    for _, row in household_data.iterrows():
                        sz = _hf.parse_household_size(row[hh_size_col], self.config)
                        cnt = int(row[hh_count_col]) if pd.notna(row[hh_count_col]) else 0
                        total_capacity += sz * cnt

                    if total_capacity > 0:
                        ratio = total_pop / total_capacity
                        if abs(ratio - 1.0) > 0.10:
                            issues.append(
                                f"Population total ({total_pop}) differs from "
                                f"household capacity ({total_capacity}) by "
                                f"{abs(ratio - 1.0):.0%}. Census marginals "
                                f"may need reconciliation."
                            )
            except Exception as exc:
                logger.debug(f"Could not cross-check pop/hh totals: {exc}")

        # --- Report ---
        if issues:
            for issue in issues:
                logger.warning(f"Input validation: {issue}")
            if self.strict:
                raise ValueError(
                    "Input validation failed (strict mode):\n  - "
                    + "\n  - ".join(issues)
                )

    # =========================================================================
    # Synthesis Engines
    # =========================================================================

    def _synthesize_topdown(
        self,
        population_data: pd.DataFrame,
        household_data: pd.DataFrame,
        car_data: Optional[pd.DataFrame]
    ) -> None:
        """
        Top-down constrained synthesis: anchor households first, then fill.
        
        This approach solves the "orphan individual" problem by:
        1. Creating EXACT household containers from the actual size distribution
        2. Generating individuals from population marginals (age × sex × role)
        3. Using constrained assignment to place individuals into households
        4. Ensuring structural constraints (couples together, children with parents)
        
        Key insight: By creating containers first, we guarantee the household
        size distribution is exact. Then we fit individuals into those containers
        using role-aware assignment.


        
        Args:
            population_data: DataFrame with age/sex/hh_role counts
            household_data: DataFrame with household size/type counts
            car_data: Optional DataFrame with car ownership statistics
        """
        # === Phase 1: Create EXACT household containers ===
        logger.info("Top-down Phase 1: Creating exact household containers")
        
        household_containers, self.next_household_id = _hf.create_household_containers(
            household_data, self.config, start_id=self.next_household_id,
        )
        self.households.extend(household_containers)
        
        # === Phase 2: Generate individual pool from marginals ===
        logger.info("Top-down Phase 2: Generating individuals from marginals")
        
        # Check if we have detailed position data - use it directly for exact role counts
        use_position_data_directly = (
            self._household_position_data is not None
            and len(self._household_position_data) > 0
        )
        
        if use_position_data_directly:
            logger.info("Using position data DIRECTLY for exact role counts")
            assert self._household_position_data is not None  # narrowed above
            individual_pool, self.next_agent_id = _pg.generate_individuals_from_position_data(
                self._household_position_data, self.config,
                start_id=self.next_agent_id, strict=self.strict,
                rng=self._rng,
            )
        else:
            logger.info("Falling back to population data with role sampling")
            individual_pool, self.next_agent_id = _pg.generate_individuals_from_population_data(
                population_data, self.config,
                start_id=self.next_agent_id, role_probs=self._role_probs,
                strict=self.strict,
                rng=self._rng,
            )
        
        logger.info(f"Generated {len(individual_pool)} individuals from marginals")
        
        # Categorize individuals by role
        singles = [a for a in individual_pool if a.hh_role == 'single']
        single_parents = [a for a in individual_pool if a.hh_role == 'single_parent']
        cohabiting = [a for a in individual_pool if a.hh_role == 'cohabiting']
        children = [a for a in individual_pool if a.hh_role == 'child']
        other = [a for a in individual_pool if a.hh_role not in ['single', 'single_parent', 'cohabiting', 'child']]
        
        logger.info(f"Pool: {len(singles)} singles, {len(single_parents)} single_parents, "
                   f"{len(cohabiting)} cohabiting, {len(children)} children, {len(other)} other")
        
        # === Phase 3: Constrained assignment ===
        logger.info("Top-down Phase 3: Constrained assignment to containers")
        
        # Sort households by size descending (fill largest first)
        containers_by_size = sorted(household_containers, key=lambda h: h.size, reverse=True)
        
        # Separate into single-person and multi-person households
        single_hh = [h for h in containers_by_size if h.size == 1]
        multi_hh = [h for h in containers_by_size if h.size >= 2]
        
        # Shuffle for randomization
        self._rng.shuffle(singles)
        self._rng.shuffle(single_parents)
        self._rng.shuffle(cohabiting)
        self._rng.shuffle(children)
        self._rng.shuffle(other)
        self._rng.shuffle(multi_hh)
        
        # Step 3a: Form couples in multi-person households
        logger.info("Step 3a: Forming couples in multi-person households")
        couples_formed = _hm.form_couples(cohabiting, multi_hh, self.constraints)
        logger.info(f"Formed {couples_formed} couples")
        
        # Step 3a.5: Place single parents in multi-person households
        logger.info("Step 3a.5: Placing single parents in multi-person households")
        single_parents_placed = _hm.place_single_parents(single_parents, multi_hh)
        logger.info(f"Placed {single_parents_placed} single parents")
        
        # Step 3b: Assign children to households with adults
        logger.info("Step 3b: Assigning children to family households")
        children_placed = _hm.place_children(children, multi_hh, self.constraints)
        logger.info(f"Placed {children_placed} children")
        
        # Step 3c: Assign "other" role individuals (roommates, etc.)
        logger.info("Step 3c: Assigning 'other' role individuals")
        other_placed = _hm.place_other(other, multi_hh)
        logger.info(f"Placed {other_placed} 'other' individuals")
        
        # Step 3d: Fill remaining multi-person slots with unassigned cohabiting adults
        logger.info("Step 3d: Filling remaining multi-person slots")
        remaining_cohabiting = [a for a in cohabiting if a.household_id is None]
        extra_placed = _hm.fill_remaining_slots(remaining_cohabiting, multi_hh)
        logger.info(f"Placed {extra_placed} additional cohabiting adults")
        
        # Step 3e: Assign singles to single-person households
        logger.info("Step 3e: Assigning singles to 1-person households")
        singles_placed = _hm.place_singles(singles, single_hh)
        logger.info(f"Placed {singles_placed} singles")
        
        # Step 3f: Handle any overflow - people who couldn't be placed
        unplaced = [a for a in individual_pool if a.household_id is None]
        if unplaced:
            logger.info(f"Step 3f: Redistributing {len(unplaced)} unplaced individuals")
            _hm.redistribute_unplaced(unplaced, household_containers)
        
        # Finalize: add all placed agents to self.agents
        self.agents = [a for a in individual_pool if a.household_id is not None]
        
        # Validate and fix household compositions
        self.households = _hm.fix_children_only_households(self.agents, self.households)
        
        # Calculate capacity stats
        total_capacity = sum(h.size for h in self.households)
        total_placed = len(self.agents)
        if total_capacity > 0:
            logger.info(f"Top-down synthesis complete: {total_placed}/{total_capacity} slots filled "
                       f"({100*total_placed/total_capacity:.1f}%)")
        else:
            logger.warning("Top-down synthesis complete: no capacity (area may have zero population)")
        
        # Store stats
        self.stats = {
            'method': 'topdown',
            'households_created': len(self.households),
            'individuals_placed': len(self.agents),
            'individuals_generated': len(individual_pool),
            'unplaced': len(individual_pool) - len(self.agents)
        }
    

    # =========================================================================
    # Test-facing wrappers (used by test_synthesizer.py)
    # =========================================================================

    def _assign_income(self, income_data: pd.DataFrame,
                       education_level_data: Optional[pd.DataFrame] = None) -> None:
        """Assign income to households and adult members."""
        _se.assign_income(
            self.agents, self.households, income_data,
            education_level_data, strict=self.strict,
        )
    
    def _build_median_income_table(self, education_data: Optional[pd.DataFrame]) -> Dict:
        """Build a lookup table of median income by (age_group, sex, education_level)."""
        return _se.build_median_income_table(education_data)
    
    def _estimate_income_from_median(self, agent, median_income_table: Dict,
                                      is_low_income: bool) -> int:
        """Estimate income for an agent using area-specific median income data."""
        return _se.estimate_income_from_median(agent, median_income_table, is_low_income)

    def _calculate_low_income_probability(self, income_data: pd.DataFrame) -> float:
        """Calculate the probability of low income from income standard data."""
        return _se.calculate_low_income_probability(income_data, strict=self.strict)

    def _build_income_distribution(self, income_data: pd.DataFrame) -> Dict[int, float]:
        """Build income decile probability distribution."""
        return _se.build_income_distribution(income_data)

    def _assign_education_level(self, education_data: pd.DataFrame) -> None:
        """Assign education level to adult individuals based on census distributions."""
        _se.assign_education_level(self.agents, education_data)

    # Age-based adjustment weights for income source assignment.
    # Delegated to socioeconomic_assigner module (arch-001).
    _INCOME_SOURCE_AGE_WEIGHTS = _se.INCOME_SOURCE_AGE_WEIGHTS

    def _assign_income_source(self, income_source_data: pd.DataFrame) -> None:
        """Assign primary income source to adults based on census distributions."""
        _se.assign_income_source(
            self.agents, income_source_data, self._INCOME_SOURCE_AGE_WEIGHTS,
        )

    def _validate_synthesis(self) -> None:
        """Validate and clean up the synthesized population."""
        issues: List[str] = []

        # Remove empty households
        empty_hhs = [h for h in self.households if len(h.members) == 0]
        if empty_hhs:
            issues.append(f"Removed {len(empty_hhs)} empty households")
            logger.warning(issues[-1])
            self.households = [h for h in self.households if len(h.members) > 0]

        # Check all agents have households
        orphaned = [a for a in self.agents if a.household_id is None]
        if orphaned:
            issues.append(f"{len(orphaned)} orphaned agents (no household_id)")
            logger.warning(issues[-1])

        # eng-013: Check for children-only households
        children_only = [
            h for h in self.households
            if h.members and all(m.age < 18 for m in h.members)
        ]
        if children_only:
            issues.append(f"{len(children_only)} children-only households remain")
            logger.warning(issues[-1])

        # eng-013: Check for duplicate IDs
        agent_ids = [a.agent_id for a in self.agents]
        if len(agent_ids) != len(set(agent_ids)):
            issues.append("Duplicate agent IDs detected")
            logger.warning(issues[-1])

        hh_ids = [h.household_id for h in self.households]
        if len(hh_ids) != len(set(hh_ids)):
            issues.append("Duplicate household IDs detected")
            logger.warning(issues[-1])

        # eng-013: Check all agents have required attributes
        missing_attrs = sum(
            1 for a in self.agents
            if a.age is None or a.sex is None or a.hh_role is None
        )
        if missing_attrs:
            issues.append(f"{missing_attrs} agents missing required attributes")
            logger.warning(issues[-1])

        # Log summary statistics
        logger.info(f"Final population: {len(self.agents)} individuals "
                   f"in {len(self.households)} households")
        # eng-012: guard division by zero
        if self.households:
            avg_size = len(self.agents) / len(self.households)
            logger.info(f"Average household size: {avg_size:.2f}")
        else:
            logger.warning("No households in final population")

        # Store validation results
        self.stats['validation_issues'] = issues
