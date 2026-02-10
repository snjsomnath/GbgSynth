"""
Population synthesis engine using top-down constrained matching.

This module contains the core logic for generating synthetic individuals
and households based on census marginals.

The algorithm:
1. Create exact household containers from census size distribution
2. Generate individuals from age/sex/role marginals
3. Use constrained assignment to place individuals into households
4. Ensure structural constraints (couples together, children with parents)
"""

import random
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

from gbgsynth.models import Agent, Household
from gbgsynth.config import Config

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

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the synthesizer.

        Args:
            config: Configuration object (will create default if None)
        """
        self.config = config or Config()
        self.constraints = self.config.constraints

        # Synthesis state
        self.agents: List[Agent] = []
        self.households: List[Household] = []
        self.next_agent_id = 1
        self.next_household_id = 1
        
        # Synthesis statistics
        self.stats: Dict = {}

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
        
        # Store household position data for role assignment
        self._household_position_data = household_position_data
        if household_position_data is not None:
            self._build_role_probability_table(household_position_data)
            logger.debug("Built role probability table from household position data")

        # Top-down constrained synthesis: anchor households first, then fill
        self._synthesize_topdown(population_data, household_data, car_data)
        
        logger.info(f"Matched {len(self.agents)} individuals to {len(self.households)} households")

        # Assign education levels from census distribution
        if education_level_data is not None:
            self._assign_education_level(education_level_data)

        # Assign income using education-based median incomes if available
        if income_data is not None:
            self._assign_income(income_data, education_level_data)

        # Assign primary income source
        if income_source_data is not None:
            self._assign_income_source(income_source_data)
        self.assign_housing_types(household_data)

        # Assign cars using propensity model with exact target
        self._assign_cars_propensity(car_data)

        # Link to building footprints (if provided)
        if buildings is not None:
            self.link_to_buildings(buildings)
            logger.debug("Linked households to building footprints")

        # Validation
        self._validate_synthesis()

        return self.agents, self.households

    def _synthesize_with_ipf(
        self,
        population_data: pd.DataFrame,
        household_data: pd.DataFrame,
        car_data: Optional[pd.DataFrame]
    ) -> None:
        """
        IPF-based synthesis that better matches marginal distributions.
        
        This approach:
        1. Uses IPF to fit joint distribution of household size × housing type
        2. Uses IPF to fit joint distribution of age × sex × household role
        3. Creates households from IPF-fitted distribution
        4. Creates individuals from IPF-fitted distribution
        5. Uses constraint-aware matching to assign individuals to households
        
        Args:
            population_data: DataFrame with age/sex/hh_role counts
            household_data: DataFrame with household size/type counts
            car_data: Optional DataFrame with car ownership statistics
        """
        from gbgsynth.ipf import IPFSynthesizer
        
        # === Step 1: Fit household distribution with IPF ===
        logger.info("IPF Step 1: Fitting household distribution")
        
        # Extract marginals
        hh_size_col = 'Hushållsstorlek' if 'Hushållsstorlek' in household_data.columns else 'hh_size'
        hh_type_col = 'Hustyp' if 'Hustyp' in household_data.columns else 'house_type'
        count_col = 'Antal' if 'Antal' in household_data.columns else household_data.columns[-1]
        
        hh_size_marginal = household_data.groupby(hh_size_col)[count_col].sum()
        hh_type_marginal = household_data.groupby(hh_type_col)[count_col].sum()
        
        # Fit 2D IPF for household size × housing type
        hh_ipf = IPFSynthesizer()
        hh_ipf.fit_2d(hh_size_marginal, hh_type_marginal)
        
        # Sample households
        total_hh = int(hh_size_marginal.sum())
        hh_samples = hh_ipf.sample(total_hh)
        
        hh_fit_stats = hh_ipf.compute_fit_statistics({
            'row': hh_size_marginal,
            'col': hh_type_marginal
        })
        logger.info(f"Household IPF: RMSE={hh_fit_stats['rmse']:.2f}, converged={hh_fit_stats['converged']}")
        
        # === Step 2: Create household objects from IPF samples ===
        logger.info("IPF Step 2: Creating households from IPF samples")
        
        for idx, row in hh_samples.iterrows():
            size = self._parse_household_size(row['row'])
            house_type = self._parse_house_type(row['col'])
            
            hh = Household(
                household_id=self.next_household_id,
                size=size,
                house_type=house_type,
                cars=self._assign_cars(house_type, size, car_data),
                assigned_hustyp=row['col']  # Store original hustyp label
            )
            self.households.append(hh)
            self.next_household_id += 1
        
        logger.info(f"Created {len(self.households)} households from IPF")
        
        # === Step 3: Fit population distribution with IPF ===
        logger.info("IPF Step 3: Fitting population distribution")
        
        age_col = 'Ålder' if 'Ålder' in population_data.columns else 'age_group'
        sex_col = 'Kön' if 'Kön' in population_data.columns else 'sex'
        role_col = 'Hushållstyp' if 'Hushållstyp' in population_data.columns else 'hh_role'
        
        age_marginal = population_data.groupby(age_col)[count_col].sum()
        sex_marginal = population_data.groupby(sex_col)[count_col].sum()
        role_marginal = population_data.groupby(role_col)[count_col].sum()
        
        pop_ipf = IPFSynthesizer()
        pop_ipf.fit({
            'age': age_marginal,
            'sex': sex_marginal,
            'role': role_marginal
        })
        
        # Sample population
        total_pop = int(age_marginal.sum())
        pop_samples = pop_ipf.sample(total_pop)
        
        pop_fit_stats = pop_ipf.compute_fit_statistics({
            'age': age_marginal,
            'sex': sex_marginal,
            'role': role_marginal
        })
        logger.info(f"Population IPF: RMSE={pop_fit_stats['rmse']:.2f}, converged={pop_fit_stats['converged']}")
        
        # === Step 4: Create agent objects from IPF samples ===
        logger.info("IPF Step 4: Creating agents from IPF samples")
        
        individual_pool = []
        for idx, row in pop_samples.iterrows():
            age = self._sample_age_from_group(row['age'])
            sex = self._translate_sex(row['sex'])
            hh_role = self._translate_hh_role(row['role'])
            
            agent = Agent(
                agent_id=self.next_agent_id,
                age=age,
                sex=sex,
                hh_role=hh_role
            )
            individual_pool.append(agent)
            self.next_agent_id += 1
        
        logger.info(f"Created {len(individual_pool)} agents from IPF")
        
        # === Step 5: Match individuals to households ===
        logger.info("IPF Step 5: Matching individuals to households")
        self._match_individuals_to_households_ipf(individual_pool)
        
        # Store IPF statistics
        self.ipf_stats = {
            'household': hh_fit_stats,
            'population': pop_fit_stats
        }
    
    def _synthesize_with_constrained_ipf(
        self,
        population_data: pd.DataFrame,
        household_data: pd.DataFrame,
        car_data: Optional[pd.DataFrame]
    ) -> None:
        """
        Constrained IPF synthesis that generates complete valid households.
        
        This approach:
        1. Defines valid household archetypes (compositions)
        2. Uses IPF to fit archetype counts to marginals
        3. Samples complete households from fitted distribution
        4. Every household is valid by construction (no post-hoc matching)
        
        Args:
            population_data: DataFrame with age/sex/hh_role counts
            household_data: DataFrame with household size/type counts
            car_data: Optional DataFrame with car ownership statistics
        """
        from gbgsynth.ipf import ConstrainedIPF
        
        # Initialize constrained IPF with our constraints
        min_gap = self.constraints.get('parent_child_age_gap_min', 18)
        max_diff = self.constraints.get('partner_age_difference_max', 15)
        
        constrained_ipf = ConstrainedIPF(
            min_parent_age_gap=min_gap,
            max_partner_age_diff=max_diff
        )
        
        # === Step 1: Fit constrained IPF to marginals ===
        logger.info("Constrained IPF Step 1: Fitting archetype counts to marginals")
        
        archetype_counts = constrained_ipf.fit(household_data, population_data)
        
        logger.info(f"Fitted {len(archetype_counts)} archetypes")
        for name, count in sorted(archetype_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                logger.info(f"  {name}: {count}")
        
        # === Step 2: Sample complete households ===
        logger.info("Constrained IPF Step 2: Sampling complete households")
        
        sampled_households = constrained_ipf.sample_households(
            archetype_counts, 
            random_state=None
        )
        
        logger.info(f"Sampled {len(sampled_households)} complete households")
        
        # === Step 3: Create Household and Agent objects ===
        logger.info("Constrained IPF Step 3: Creating household and agent objects")
        
        # Extract housing type marginal for assignment
        hh_type_col = 'Hustyp' if 'Hustyp' in household_data.columns else 'house_type'
        count_col = 'Antal' if 'Antal' in household_data.columns else household_data.columns[-1]
        hh_type_marginal = household_data.groupby(hh_type_col)[count_col].sum()
        
        # Create probability distribution for housing types
        total_hh_type = hh_type_marginal.sum()
        hh_type_probs = hh_type_marginal / total_hh_type
        
        for hh_data in sampled_households:
            # Sample housing type
            house_type_label = np.random.choice(
                hh_type_probs.index,
                p=hh_type_probs.values
            )
            house_type = self._parse_house_type(house_type_label)
            
            # Create household
            hh = Household(
                household_id=self.next_household_id,
                size=hh_data['size'],
                house_type=house_type,
                cars=self._assign_cars(house_type, hh_data['size'], car_data),
                assigned_hustyp=house_type_label
            )
            self.households.append(hh)
            self.next_household_id += 1
            
            # Create and assign members
            for member in hh_data['members']:
                agent = Agent(
                    agent_id=self.next_agent_id,
                    age=member['age'],
                    sex=member['sex'],
                    hh_role=member['role']
                )
                hh.add_member(agent)
                self.agents.append(agent)
                self.next_agent_id += 1
        
        logger.info(f"Created {len(self.households)} households with {len(self.agents)} agents")
        
        # === Step 4: Compute and store fit statistics ===
        fit_stats = constrained_ipf.compute_fit_statistics(household_data, population_data)
        
        self.ipf_stats = {
            'constrained_ipf': fit_stats,
            'archetype_counts': archetype_counts,
            'iterations': fit_stats.get('iterations', 0),
            'converged': fit_stats.get('converged', False)
        }
        
        logger.info(f"Constrained IPF fit: RMSE={fit_stats['rmse']:.2f}, "
                   f"HH error={fit_stats['fitted_households'] - fit_stats['target_households']}, "
                   f"Pop error={fit_stats['fitted_population'] - fit_stats['target_population']}")

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
        
        hh_size_col = 'Hushållsstorlek' if 'Hushållsstorlek' in household_data.columns else 'hh_size'
        hh_type_col = 'Hustyp' if 'Hustyp' in household_data.columns else 'house_type'
        count_col = 'Antal' if 'Antal' in household_data.columns else household_data.columns[-1]
        
        # Get exact counts by size
        size_counts = household_data.groupby(hh_size_col)[count_col].sum().to_dict()
        
        # Get housing type distribution for assignment
        type_counts = household_data.groupby(hh_type_col)[count_col].sum()
        type_probs = type_counts / type_counts.sum()
        
        # Create household containers (largest first for priority filling)
        household_containers = []
        for size_label, count in sorted(size_counts.items(), 
                                        key=lambda x: self._parse_household_size(x[0]), 
                                        reverse=True):
            size = self._parse_household_size(size_label)
            if size == 0 or count == 0:
                continue
                
            for _ in range(int(count)):
                # Sample housing type
                house_type_label = np.random.choice(type_probs.index, p=type_probs.values)
                house_type = self._parse_house_type(house_type_label)
                
                hh = Household(
                    household_id=self.next_household_id,
                    size=size,
                    house_type=house_type,
                    cars=self._assign_cars(house_type, size, car_data),
                    assigned_hustyp=house_type_label
                )
                household_containers.append(hh)
                self.households.append(hh)
                self.next_household_id += 1
        
        logger.info(f"Created {len(self.households)} exact household containers")
        
        # Log size distribution
        size_dist = {}
        for hh in self.households:
            size_dist[hh.size] = size_dist.get(hh.size, 0) + 1
        logger.info(f"Household size distribution: {dict(sorted(size_dist.items()))}")
        
        # === Phase 2: Generate individual pool from marginals ===
        logger.info("Top-down Phase 2: Generating individuals from marginals")
        
        # Check if we have detailed position data - use it directly for exact role counts
        use_position_data_directly = (
            hasattr(self, '_household_position_data') and 
            self._household_position_data is not None and
            len(self._household_position_data) > 0
        )
        
        if use_position_data_directly:
            logger.info("Using position data DIRECTLY for exact role counts")
            individual_pool = self._generate_individuals_from_position_data()
        else:
            logger.info("Falling back to population data with role sampling")
            individual_pool = self._generate_individuals_from_population_data(population_data)
        
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
        random.shuffle(singles)
        random.shuffle(single_parents)
        random.shuffle(cohabiting)
        random.shuffle(children)
        random.shuffle(other)
        random.shuffle(multi_hh)
        
        # Step 3a: Form couples in multi-person households
        logger.info("Step 3a: Forming couples in multi-person households")
        couples_formed = self._form_couples_topdown(cohabiting, multi_hh)
        logger.info(f"Formed {couples_formed} couples")
        
        # Step 3a.5: Place single parents in multi-person households
        # Single parents MUST be placed before children, so children can join them
        logger.info("Step 3a.5: Placing single parents in multi-person households")
        single_parents_placed = self._place_single_parents_topdown(single_parents, multi_hh)
        logger.info(f"Placed {single_parents_placed} single parents")
        
        # Step 3b: Assign children to households with adults
        logger.info("Step 3b: Assigning children to family households")
        children_placed = self._place_children_topdown(children, multi_hh)
        logger.info(f"Placed {children_placed} children")
        
        # Step 3c: Assign "other" role individuals (roommates, etc.)
        logger.info("Step 3c: Assigning 'other' role individuals")
        other_placed = self._place_other_topdown(other, multi_hh)
        logger.info(f"Placed {other_placed} 'other' individuals")
        
        # Step 3d: Fill remaining multi-person slots with unassigned cohabiting adults
        logger.info("Step 3d: Filling remaining multi-person slots")
        remaining_cohabiting = [a for a in cohabiting if a.household_id is None]
        extra_placed = self._fill_remaining_slots_topdown(remaining_cohabiting, multi_hh)
        logger.info(f"Placed {extra_placed} additional cohabiting adults")
        
        # Step 3e: Assign singles to single-person households
        logger.info("Step 3e: Assigning singles to 1-person households")
        singles_placed = self._place_singles_topdown(singles, single_hh)
        logger.info(f"Placed {singles_placed} singles")
        
        # Step 3f: Handle any overflow - people who couldn't be placed
        unplaced = [a for a in individual_pool if a.household_id is None]
        if unplaced:
            logger.info(f"Step 3f: Redistributing {len(unplaced)} unplaced individuals")
            self._redistribute_unplaced_topdown(unplaced, household_containers)
        
        # Finalize: add all placed agents to self.agents
        self.agents = [a for a in individual_pool if a.household_id is not None]
        
        # Validate and fix household compositions
        self._fix_children_only_households()
        
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
    
    def _form_couples_topdown(self, cohabiting: List[Agent], multi_hh: List[Household]) -> int:
        """Form couples in multi-person households."""
        max_age_diff = self.constraints.get('partner_age_difference_max', 15)
        
        males = [a for a in cohabiting if a.sex == 'male']
        females = [a for a in cohabiting if a.sex == 'female']
        
        couples_formed = 0
        
        for hh in multi_hh:
            if not hh.can_fit(2):
                continue
            
            # Find compatible pair
            for male in males:
                if male.household_id is not None:
                    continue
                for female in females:
                    if female.household_id is not None:
                        continue
                    if abs(male.age - female.age) <= max_age_diff:
                        hh.add_member(male)
                        hh.add_member(female)
                        self.agents.extend([male, female])
                        couples_formed += 1
                        break
                else:
                    continue
                break
        
        return couples_formed
    
    def _place_single_parents_topdown(self, single_parents: List[Agent], multi_hh: List[Household]) -> int:
        """
        Place single parents in multi-person households.
        
        Single parents (Ensamstående förälder) must go to multi-person households
        because they live with children. They should be placed BEFORE children
        so that children can then be assigned to their households.
        
        Each single parent household should have EXACTLY ONE single parent.
        Prioritize empty households to ensure proper single-parent family formation.
        """
        placed = 0
        
        # Sort single parents by age descending (older parents first)
        single_parents = sorted(single_parents, key=lambda a: -a.age)
        
        for parent in single_parents:
            # Find an EMPTY multi-person household (no members yet)
            # This ensures each single parent gets their own household
            best_hh = None
            for hh in multi_hh:
                if len(hh.members) > 0:
                    continue  # Skip households that already have members
                
                if not hh.can_fit():
                    continue
                
                # Need at least 2 total capacity (parent + child)
                if hh.size >= 2:
                    best_hh = hh
                    break
            
            if best_hh:
                best_hh.add_member(parent)
                placed += 1
        
        return placed
    
    def _place_children_topdown(self, children: List[Agent], multi_hh: List[Household]) -> int:
        """
        Place children in households that have adults.
        
        Prioritizes:
        1. Single-parent households (ensure single parents get children first)
        2. Couple households (with suitable parents)
        """
        min_parent_gap = self.constraints.get('parent_child_age_gap_min', 18)
        
        # Sort children youngest first
        children = sorted(children, key=lambda c: c.age)
        
        # Separate single-parent households from couple households
        single_parent_hh = [hh for hh in multi_hh 
                          if any(m.hh_role == 'single_parent' for m in hh.members)
                          and sum(1 for m in hh.members if m.hh_role == 'cohabiting') < 2]
        couple_hh = [hh for hh in multi_hh 
                    if sum(1 for m in hh.members if m.hh_role == 'cohabiting') >= 2]
        
        placed = 0
        
        # First pass: prioritize single-parent households
        for child in children:
            if child.household_id is not None:
                continue
                
            for hh in single_parent_hh:
                if not hh.can_fit():
                    continue
                
                # Check if there's a suitable parent
                adults = [m for m in hh.members if m.age >= child.age + min_parent_gap]
                if adults:
                    hh.add_member(child)
                    placed += 1
                    break
        
        # Second pass: fill remaining children into couple households
        for child in children:
            if child.household_id is not None:
                continue
                
            for hh in couple_hh:
                if not hh.can_fit():
                    continue
                
                # Check if there's a suitable parent
                adults = [m for m in hh.members if m.age >= child.age + min_parent_gap]
                if adults:
                    hh.add_member(child)
                    placed += 1
                    break
        
        # Third pass: any remaining children go to any household with adults
        for child in children:
            if child.household_id is not None:
                continue
                
            for hh in multi_hh:
                if not hh.can_fit():
                    continue
                
                adults = [m for m in hh.members if m.age >= child.age + min_parent_gap]
                if adults:
                    hh.add_member(child)
                    placed += 1
                    break
        
        return placed
    
    def _place_other_topdown(self, other: List[Agent], multi_hh: List[Household]) -> int:
        """Place 'other' role individuals (roommates, multi-gen, etc.)."""
        placed = 0
        
        for agent in other:
            for hh in multi_hh:
                if hh.can_fit():
                    hh.add_member(agent)
                    placed += 1
                    break
        
        return placed
    
    def _fix_children_only_households(self) -> None:
        """
        Fix households that contain only children (no adults).
        
        This is a post-synthesis validation step that corrects any
        households that ended up with only children. Such households
        are invalid and would be misclassified as "Övriga hushåll"
        when they should be part of family structures.
        
        Strategy:
        1. Identify households with only children
        2. Move children to other households that have adults
        3. If source household becomes empty, remove it
        """
        children_only_hh = []
        hh_with_adults = []
        
        for hh in self.households:
            if not hh.members:
                continue
            has_adult = any(m.age >= 18 for m in hh.members)
            if has_adult:
                hh_with_adults.append(hh)
            else:
                # All members are children
                children_only_hh.append(hh)
        
        if not children_only_hh:
            return  # No issues found
        
        # Count affected children
        affected_children = sum(len(hh.members) for hh in children_only_hh)
        logger.warning(f"Found {len(children_only_hh)} children-only households "
                      f"({affected_children} children). Redistributing to family households.")
        
        # Redistribute children from invalid households to valid ones
        children_to_move = []
        for hh in children_only_hh:
            children_to_move.extend(list(hh.members))
            # Clear the household
            for child in list(hh.members):
                hh.members.remove(child)
                child.household_id = None
        
        # Sort target households by capacity (prefer households with more space)
        hh_with_adults.sort(key=lambda h: h.size - len(h.members), reverse=True)
        
        moved = 0
        for child in children_to_move:
            for hh in hh_with_adults:
                if hh.can_fit():
                    hh.add_member(child)
                    moved += 1
                    break
            else:
                # No capacity - force add to a random family household
                if hh_with_adults:
                    target = random.choice(hh_with_adults)
                    target.size += 1
                    target.add_member(child)
                    moved += 1
        
        # Remove empty households
        self.households = [hh for hh in self.households if hh.members]
        
        logger.info(f"Moved {moved} children to family households. "
                   f"Remaining households: {len(self.households)}")
    
    def _fill_remaining_slots_topdown(self, remaining: List[Agent], multi_hh: List[Household]) -> int:
        """Fill remaining slots in multi-person households."""
        placed = 0
        
        for agent in remaining:
            for hh in multi_hh:
                if hh.can_fit():
                    hh.add_member(agent)
                    placed += 1
                    break
        
        return placed
    
    def _place_singles_topdown(self, singles: List[Agent], single_hh: List[Household]) -> int:
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
    
    def _redistribute_unplaced_topdown(self, unplaced: List[Agent], all_hh: List[Household]) -> None:
        """
        Redistribute unplaced individuals to households.
        
        IMPORTANT: Places adults before children, and ensures children are
        only placed in households that have at least one adult. This prevents
        creating invalid households with only children (which would be 
        mis-classified as "Övriga hushåll" instead of proper family types).
        
        For any overflow (due to census data privacy rounding), randomly
        spreads individuals across households to distribute error uniformly.
        
        Note: Census data often has small discrepancies (~1-2%) between
        population tables and household tables due to statistical disclosure
        control (privacy protection through rounding/suppression).
        """
        # Separate children from adults - children need households WITH adults
        children = [a for a in unplaced if a.hh_role == 'child' or a.age < 18]
        adults = [a for a in unplaced if a not in children]
        
        placed_count = 0
        still_unplaced_adults = []
        still_unplaced_children = []
        
        # Phase 1: Place adults first (they can go anywhere with capacity)
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
        
        # Phase 2: Place children ONLY in households that already have adults
        for child in children:
            placed = False
            for hh in all_hh:
                if not hh.can_fit():
                    continue
                # Check if household has at least one adult
                has_adult = any(m.age >= 18 for m in hh.members)
                if has_adult:
                    hh.add_member(child)
                    placed = True
                    placed_count += 1
                    break
            if not placed:
                still_unplaced_children.append(child)
        
        still_unplaced = still_unplaced_adults + still_unplaced_children
        
        # For overflow individuals, spread across households
        # But maintain the rule: children only go to households with adults
        if still_unplaced:
            logger.info(f"Spreading {len(still_unplaced)} overflow individuals across households "
                       f"(census privacy rounding adjustment)")
            
            # Separate adults and children for overflow handling too
            overflow_adults = [a for a in still_unplaced if a not in children]
            overflow_children = [a for a in still_unplaced if a in children]
            
            # Shuffle households to spread uniformly
            shuffled_hh = list(all_hh)
            random.shuffle(shuffled_hh)
            
            # Place overflow adults in any household
            for i, agent in enumerate(overflow_adults):
                hh = shuffled_hh[i % len(shuffled_hh)]
                hh.size += 1
                hh.add_member(agent)
                placed_count += 1
            
            # Place overflow children only in households with adults
            hh_with_adults = [hh for hh in shuffled_hh if any(m.age >= 18 for m in hh.members)]
            if not hh_with_adults:
                # Fallback: if no households have adults yet, use any
                hh_with_adults = shuffled_hh
                
            for i, child in enumerate(overflow_children):
                hh = hh_with_adults[i % len(hh_with_adults)]
                hh.size += 1
                hh.add_member(child)
                placed_count += 1
            
            logger.info(f"Placed all {placed_count} overflow individuals")

    def _match_individuals_to_households_ipf(self, pool: List[Agent]) -> None:
        """
        IPF-aware matching that respects household size constraints exactly.
        
        This uses a more sophisticated assignment:
        1. Sort households by size (largest first) to fill complex HHs first
        2. Match couples to 2+ person households (respecting capacity)
        3. Match children to family households
        4. Match singles to remaining 1-person households
        5. For any leftover individuals, redistribute to under-filled households
        
        Args:
            pool: List of Agent objects to assign
        """
        # Build capacity tracking
        hh_by_size = {}
        for hh in self.households:
            if hh.size not in hh_by_size:
                hh_by_size[hh.size] = []
            hh_by_size[hh.size].append(hh)
        
        # Categorize individuals
        adults_cohabiting = [a for a in pool if a.is_adult() and a.hh_role == 'cohabiting']
        adults_single = [a for a in pool if a.is_adult() and a.hh_role == 'single']
        children = [a for a in pool if a.is_child()]
        
        random.shuffle(adults_cohabiting)
        random.shuffle(adults_single)
        random.shuffle(children)
        
        logger.info(f"Pool: {len(adults_cohabiting)} cohabiting adults, {len(adults_single)} single adults, {len(children)} children")
        
        # Phase 1: Form couples in multi-person households (largest first)
        # Sort households by size descending to fill bigger HHs first
        multi_hhs = sorted([hh for hh in self.households if hh.size >= 2], 
                          key=lambda h: h.size, reverse=True)
        
        couples_formed = 0
        max_age_diff = self.constraints.get('partner_age_difference_max', 15)
        
        males = [a for a in adults_cohabiting if a.sex == 'male']
        females = [a for a in adults_cohabiting if a.sex == 'female']
        
        for hh in multi_hhs:
            if not hh.can_fit(2):
                continue
            
            # Find compatible pair
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
                        break
                else:
                    continue
                break
        
        logger.info(f"Formed {couples_formed} couples")
        
        # Phase 2: Assign children to households with adults (prioritize big families)
        min_parent_gap = self.constraints.get('parent_child_age_gap_min', 18)
        children_assigned = 0
        
        # Sort children youngest first
        children = sorted(children, key=lambda c: c.age)
        
        # Sort households by remaining capacity descending
        multi_hhs_with_adults = [hh for hh in multi_hhs if len(hh.members) > 0 and hh.can_fit()]
        multi_hhs_with_adults = sorted(multi_hhs_with_adults, 
                                       key=lambda h: h.size - len(h.members), reverse=True)
        
        for child in children:
            if child.household_id is not None:
                continue
            
            for hh in multi_hhs_with_adults:
                if not hh.can_fit():
                    continue
                
                # Check for suitable parent
                head = hh.head
                if head and (head.age - child.age) >= min_parent_gap:
                    hh.add_member(child)
                    children_assigned += 1
                    break
        
        logger.info(f"Assigned {children_assigned} children")
        
        # Phase 3: Fill remaining capacity with unassigned cohabiting adults
        # (for households that got a couple but have room for more)
        remaining_cohab = [a for a in adults_cohabiting if a.household_id is None]
        cohab_added = 0
        
        for hh in multi_hhs:
            if not hh.can_fit():
                continue
            for adult in remaining_cohab:
                if adult.household_id is None and hh.can_fit():
                    hh.add_member(adult)
                    cohab_added += 1
        
        if cohab_added > 0:
            logger.info(f"Added {cohab_added} additional cohabiting adults")
        
        # Phase 4: Fill single-person households with single adults
        single_hhs = hh_by_size.get(1, [])
        singles_assigned = 0
        
        for single in adults_single:
            if single.household_id is not None:
                continue
            
            for hh in single_hhs:
                if hh.is_full():
                    continue
                hh.add_member(single)
                singles_assigned += 1
                break
        
        logger.info(f"Assigned {singles_assigned} singles")
        
        # Phase 5: Fill remaining capacity in multi-person households
        remaining = [a for a in pool if a.household_id is None]
        redistributed = 0
        
        for agent in remaining:
            for hh in self.households:
                if hh.can_fit():
                    hh.add_member(agent)
                    redistributed += 1
                    break
        
        if redistributed > 0:
            logger.info(f"Redistributed {redistributed} remaining individuals")
        
        # Phase 5: Handle truly unmatched by creating new single households
        # (IPF ensures totals match, but matching constraints may leave some unassigned)
        still_remaining = [a for a in pool if a.household_id is None]
        if still_remaining:
            logger.warning(f"{len(still_remaining)} individuals could not be matched, creating additional households")
            for agent in still_remaining:
                hh = Household(
                    household_id=self.next_household_id,
                    size=1
                )
                hh.add_member(agent)
                self.households.append(hh)
                self.next_household_id += 1
        
        # Collect all agents
        self.agents = []
        for hh in self.households:
            self.agents.extend(hh.members)

    def _create_households(
        self,
        household_data: pd.DataFrame,
        car_data: Optional[pd.DataFrame]
    ) -> None:
        """
        Create empty household shells based on size/type statistics.

        Args:
            household_data: DataFrame with household size and type counts
            car_data: Optional DataFrame with car ownership statistics
        """
        # Parse household data (expected columns: hh_size, house_type, count)
        for _, row in household_data.iterrows():
            size = self._parse_household_size(row.get('Hushållsstorlek', row.get('hh_size')))
            house_type = self._parse_house_type(row.get('Hustyp', row.get('house_type')))
            count = int(row.get('Antal', row.get('count', row.iloc[-1])))

            # Create households
            for _ in range(count):
                hh = Household(
                    household_id=self.next_household_id,
                    size=size,
                    house_type=house_type,
                    cars=self._assign_cars(house_type, size, car_data)
                )
                self.households.append(hh)
                self.next_household_id += 1

    def _generate_individual_pool(self, population_data: pd.DataFrame) -> List[Agent]:
        """
        Generate a pool of individuals from demographic statistics.

        Args:
            population_data: DataFrame with age/sex/hh_role counts

        Returns:
            List of Agent objects (not yet assigned to households)
        """
        pool = []

        for _, row in population_data.iterrows():
            # Parse row data
            age_group = row.get('Ålder', row.get('age_group'))
            sex = row.get('Kön', row.get('sex'))
            hh_role = row.get('Hushållstyp', row.get('hh_role'))
            count = int(row.get('Antal', row.get('count', row.iloc[-1])))

            # Translate values
            sex_eng = self._translate_sex(sex)
            hh_role_eng = self._translate_hh_role(hh_role)

            # Generate individuals for this category
            for _ in range(count):
                age = self._sample_age_from_group(age_group)
                agent = Agent(
                    agent_id=self.next_agent_id,
                    age=age,
                    sex=sex_eng,
                    hh_role=hh_role_eng
                )
                pool.append(agent)
                self.next_agent_id += 1

        return pool

    def _match_individuals_to_households(self, pool: List[Agent]) -> None:
        """
        Greedy matching algorithm to assign individuals to households.

        Algorithm phases:
        1. Form couples in 2+ person households
        2. Assign children to families
        3. Fill single-person households
        4. Handle remaining unmatched individuals
        """
        # Separate pool by type
        adults_cohabiting = [a for a in pool if a.is_adult() and a.hh_role == 'cohabiting']
        adults_single = [a for a in pool if a.is_adult() and a.hh_role == 'single']
        children = [a for a in pool if a.is_child()]

        # Sort households: couples first, then families, then singles
        couple_hhs = [h for h in self.households if h.size >= 2]
        single_hhs = [h for h in self.households if h.size == 1]

        # Phase 1: Form couples
        logger.info("Phase 1: Forming couples")
        couples_formed = self._form_couples(adults_cohabiting, couple_hhs)
        logger.info(f"Formed {couples_formed} couples")

        # Phase 2: Assign children
        logger.info("Phase 2: Assigning children")
        children_assigned = self._assign_children(children, couple_hhs)
        logger.info(f"Assigned {children_assigned} children")

        # Phase 3: Fill single households
        logger.info("Phase 3: Filling single households")
        singles_assigned = self._fill_single_households(adults_single, single_hhs)
        logger.info(f"Assigned {singles_assigned} singles")

        # Phase 4: Handle remaining
        remaining = [a for a in pool if a.household_id is None]
        if remaining:
            logger.warning(f"{len(remaining)} individuals could not be matched")
            self._handle_unmatched(remaining)

        # Collect all agents from households into self.agents
        self.agents = []
        for hh in self.households:
            self.agents.extend(hh.members)

    def _form_couples(self, adults: List[Agent], households: List[Household]) -> int:
        """
        Form couples in multi-person households.

        Args:
            adults: List of cohabiting adults
            households: List of households needing couples

        Returns:
            Number of couples formed
        """
        couples_formed = 0
        max_age_diff = self.constraints['partner_age_difference_max']

        # Sort adults by age for better matching
        males = sorted([a for a in adults if a.sex == 'male'], key=lambda x: x.age)
        females = sorted([a for a in adults if a.sex == 'female'], key=lambda x: x.age)

        # Match opposite-sex pairs
        for hh in households:
            if hh.can_fit(2):
                # Try to find a compatible pair
                for i, male in enumerate(males):
                    if male.household_id is not None:
                        continue

                    for j, female in enumerate(females):
                        if female.household_id is not None:
                            continue

                        if male.can_be_partner_with(female, max_age_diff):
                            hh.add_member(male)
                            hh.add_member(female)
                            couples_formed += 1
                            break
                    else:
                        continue
                    break

        return couples_formed

    def _assign_children(self, children: List[Agent], households: List[Household]) -> int:
        """
        Assign children to households with parents.

        Args:
            children: List of child agents
            households: List of households that can accommodate children

        Returns:
            Number of children assigned
        """
        assigned = 0
        min_age_gap = self.constraints['parent_child_age_gap_min']

        # Sort children by age (younger first for biological plausibility)
        children = sorted(children, key=lambda x: x.age)

        for child in children:
            if child.household_id is not None:
                continue

            # Find suitable household
            for hh in households:
                if not hh.can_fit():
                    continue

                # Check if household has adults who can be parents
                head = hh.head
                if head and head.can_be_parent_of(child, min_age_gap):
                    hh.add_member(child)
                    assigned += 1
                    break

        return assigned

    def _fill_single_households(self, singles: List[Agent], households: List[Household]) -> int:
        """
        Fill single-person households with remaining singles.

        Args:
            singles: List of single adults
            households: List of 1-person households

        Returns:
            Number of singles assigned
        """
        assigned = 0

        for hh in households:
            if hh.is_full():
                continue

            # Try to find a single
            for single in singles:
                if single.household_id is None:
                    hh.add_member(single)
                    assigned += 1
                    break

        return assigned

    def _handle_unmatched(self, unmatched: List[Agent]) -> None:
        """
        Create additional households for unmatched individuals.

        Args:
            unmatched: List of agents without households
        """
        logger.info(f"Creating {len(unmatched)} additional single households")

        for agent in unmatched:
            hh = Household(
                household_id=self.next_household_id,
                size=1
            )
            hh.add_member(agent)
            self.households.append(hh)
            self.next_household_id += 1

    def _assign_income(self, income_data: pd.DataFrame,
                       education_level_data: Optional[pd.DataFrame] = None) -> None:
        """
        Assign income to households and adult members.
        
        Uses two data sources:
        1. Income standard data (10_InkStandard): determines low/not-low income
           probability for each household
        2. Education level data (23_InkomsterUtbildning): provides area-specific
           median income by education × age × sex for realistic SEK amounts
        
        If education data with Medianinkomst is available, income is assigned
        based on the agent's education level, age group, and sex. Otherwise,
        falls back to crude decile-based estimates.
        
        Income is assigned to:
        - Adults (age >= 18): Individual income based on education/age/sex
        - Children (age < 18): No individual income (income = 0)
        
        Args:
            income_data: DataFrame with income standard distribution
            education_level_data: Optional DataFrame with median income by
                                 education × age × sex
        """
        # Calculate low income probability from actual data
        low_income_prob = self._calculate_low_income_probability(income_data)
        logger.info(f"Low income probability from marginals: {low_income_prob:.1%}")
        
        # Build median income lookup from education data if available
        median_income_table = self._build_median_income_table(education_level_data)
        use_median = len(median_income_table) > 0
        if use_median:
            logger.info(f"Using area-specific median incomes ({len(median_income_table)} entries)")
        else:
            logger.info("Using decile-based income estimates (no median income data)")

        # Assign income at the household level, then distribute to adult members
        for household in self.households:
            # Determine household's income standard
            is_low_income = random.random() < low_income_prob
            income_standard = 'low' if is_low_income else 'not_low'
            
            # Get adults in this household
            adults = [m for m in household.members if m.age >= 18]
            children = [m for m in household.members if m.age < 18]
            
            # Assign income to adults
            for adult in adults:
                if is_low_income:
                    adult.income_decile = random.randint(1, 2)
                else:
                    adult.income_decile = random.randint(3, 10)
                adult.income_standard = income_standard
                
                # Try education-based median income first
                if use_median:
                    adult.income = self._estimate_income_from_median(
                        adult, median_income_table, is_low_income
                    )
                else:
                    adult.income = self._estimate_income_from_decile(adult.income_decile)
            
            # Children get no individual income
            for child in children:
                child.income = 0
                child.income_decile = None
                child.income_standard = income_standard  # Inherit household's standard
    
    def _build_median_income_table(self, education_data: Optional[pd.DataFrame]) -> Dict:
        """
        Build a lookup table of median income by (age_group, sex, education_level).
        
        Extracts Medianinkomst rows from the education table and creates
        a dictionary for fast lookup during income assignment.
        
        Args:
            education_data: DataFrame with Tabellvärde, Ålder, Kön,
                           Utbildningsnivå, Antal columns
        
        Returns:
            Dict mapping (age_group_label, sex_en, edu_en) -> median_income_sek
        """
        if education_data is None or education_data.empty:
            return {}
        
        if 'Tabellvärde' not in education_data.columns:
            return {}
        
        median_data = education_data[education_data['Tabellvärde'] == 'Medianinkomst']
        if median_data.empty:
            return {}
        
        sex_map = {'Man': 'male', 'Kvinna': 'female'}
        edu_map = {
            'Förgymnasial utbildning': 'pre_secondary',
            'Gymnasial utbildning': 'secondary',
            'Eftergymnasial utbildning': 'post_secondary',
            'Uppgift saknas': 'unknown',
        }
        
        table = {}
        for _, row in median_data.iterrows():
            sex_sv = row.get('Kön', '')
            sex_en = sex_map.get(sex_sv)
            if not sex_en:
                continue
            
            edu_sv = row.get('Utbildningsnivå', '')
            edu_en = edu_map.get(edu_sv)
            if not edu_en:
                continue
            
            age_label = row.get('Ålder', '')
            median_income = row.get('Antal', 0)  # Antal holds the value for all metrics
            
            if pd.notna(median_income) and median_income > 0:
                table[(age_label, sex_en, edu_en)] = float(median_income)
        
        return table
    
    def _estimate_income_from_median(self, agent, median_income_table: Dict,
                                      is_low_income: bool) -> float:
        """
        Estimate income for an agent using area-specific median income data.
        
        Looks up the median income for the agent's age group × sex × education
        level, then applies randomness to create realistic variation.
        
        For low-income households, income is scaled down by 40-70%.
        
        Args:
            agent: The Agent to estimate income for
            median_income_table: Lookup dict (age_label, sex, edu) -> median_sek
            is_low_income: Whether the household is flagged as low income
        
        Returns:
            Estimated income in SEK
        """
        # Define age group boundaries matching the education table
        age_groups = [
            (18, 24, '18-24 år'),
            (25, 34, '25-34 år'),
            (35, 44, '35-44 år'),
            (45, 54, '45-54 år'),
            (55, 64, '55-64 år'),
            (65, 74, '65-74 år'),
            (75, 120, '75- år'),
        ]
        
        # Find matching age group
        age_label = None
        for ag_min, ag_max, label in age_groups:
            if ag_min <= agent.age <= ag_max:
                age_label = label
                break
        
        if age_label is None:
            return self._estimate_income_from_decile(agent.income_decile or 5)
        
        edu = getattr(agent, 'education', None) or 'unknown'
        key = (age_label, agent.sex, edu)
        
        median = median_income_table.get(key)
        
        if median is None:
            # Try without education specificity
            for edu_fallback in ['secondary', 'pre_secondary', 'post_secondary', 'unknown']:
                fallback_key = (age_label, agent.sex, edu_fallback)
                median = median_income_table.get(fallback_key)
                if median:
                    break
        
        if median is None:
            return self._estimate_income_from_decile(agent.income_decile or 5)
        
        # Apply variation: ±30% around median (log-normal-ish)
        variation = random.gauss(0, 0.15)  # ~15% std dev
        income = median * (1 + variation)
        
        # Low income households get significantly less
        if is_low_income:
            income *= random.uniform(0.3, 0.6)
        
        return max(0, round(income))

    def _calculate_low_income_probability(self, income_data: pd.DataFrame) -> float:
        """Calculate the probability of low income from income standard data."""
        if income_data.empty:
            return 0.1  # Default fallback
        
        # Find Inkomststandard column
        income_col = None
        for col in ['Inkomststandard', 'inkomststandard']:
            if col in income_data.columns:
                income_col = col
                break
        
        if income_col is None:
            return 0.1
        
        # Find count column
        count_col = None
        for col in income_data.columns:
            if income_data[col].dtype in ['int64', 'float64'] and col not in ['År']:
                count_col = col
                break
        
        if count_col is None:
            count_col = income_data.columns[-1]
        
        # Sum up low income vs not low income
        low_income_count = 0
        not_low_count = 0
        
        for _, row in income_data.iterrows():
            cat = str(row[income_col]).lower()
            count = int(row[count_col]) if pd.notna(row[count_col]) else 0
            
            if 'inte har låg' in cat:  # Not low income
                not_low_count += count
            elif 'har låg' in cat:  # Low income
                low_income_count += count
            # Skip "Ingår ej i helårshushåll"
        
        total = low_income_count + not_low_count
        if total > 0:
            return low_income_count / total
        
        return 0.1  # Fallback

    def _build_income_distribution(self, income_data: pd.DataFrame) -> Dict[int, float]:
        """Build income decile probability distribution."""
        dist = {}
        
        if income_data.empty:
            return dist
        
        # Find the column that might contain income interval/decile info
        income_col = None
        for col in income_data.columns:
            col_lower = col.lower()
            if 'inkomst' in col_lower or 'decil' in col_lower or 'interval' in col_lower or 'standard' in col_lower:
                income_col = col
                break
        
        # If no specific column found, try first non-numeric column
        if income_col is None:
            for col in income_data.columns:
                if income_data[col].dtype == object:
                    income_col = col
                    break
        
        if income_col is None:
            logger.warning(f"Could not find income column in data. Columns: {list(income_data.columns)}")
            return dist
        
        for _, row in income_data.iterrows():
            decile = self._parse_income_decile(row.get(income_col, ''))
            # Get count from last column (typically the value column)
            try:
                count = int(row.iloc[-1])
            except (ValueError, TypeError):
                continue
            if decile:
                dist[decile] = dist.get(decile, 0) + count

        # Normalize to probabilities
        total = sum(dist.values())
        if total > 0:
            return {k: v / total for k, v in dist.items()}
        return dist

    def _assign_education_level(self, education_data: pd.DataFrame) -> None:
        """
        Assign education level to adult individuals based on census distributions.
        
        Uses the 23_InkomsterUtbildning table which provides population counts
        (Folkmängd) by education level, age group, and sex for adults 18+.
        
        Education levels assigned:
        - pre_secondary (Förgymnasial utbildning)
        - secondary (Gymnasial utbildning) 
        - post_secondary (Eftergymnasial utbildning)
        - unknown (Uppgift saknas)
        
        Children (<18) get education='child'.
        
        The probability distribution is conditioned on age group × sex,
        so the synthetic population should closely match the census
        education level distribution.
        
        Args:
            education_data: DataFrame with columns Ålder, Kön, Utbildningsnivå,
                           Tabellvärde, Antal. Contains multiple metrics
                           (Folkmängd, Medianinkomst, etc.) — we filter to
                           Folkmängd for probability computation.
        """
        if education_data is None or education_data.empty:
            logger.warning("No education level data available, skipping education assignment")
            return
        
        # Filter to population counts only (not income statistics)
        if 'Tabellvärde' in education_data.columns:
            folk_data = education_data[education_data['Tabellvärde'] == 'Folkmängd'].copy()
        else:
            folk_data = education_data.copy()
        
        # Map Swedish education levels to internal values
        edu_level_map = {
            'Förgymnasial utbildning': 'pre_secondary',
            'Gymnasial utbildning': 'secondary',
            'Eftergymnasial utbildning': 'post_secondary',
            'Uppgift saknas': 'unknown',
        }
        
        # Map Swedish sex to internal values
        sex_map = {'Man': 'male', 'Kvinna': 'female'}
        
        # Build age group boundaries from the data
        # Age groups in the table: "18-24 år", "25-34 år", ..., "75- år"
        age_groups = []
        for ag in folk_data['Ålder'].unique():
            ag_str = str(ag).replace(' år', '').strip()
            if '-' in ag_str:
                parts = ag_str.split('-')
                if parts[1] == '':
                    # Open-ended like "75-"
                    age_groups.append((int(parts[0]), 120, ag))
                else:
                    age_groups.append((int(parts[0]), int(parts[1]), ag))
        
        # Build probability lookup: (age_group_label, sex) -> {edu_level: probability}
        prob_table = {}
        for sex_sv, sex_en in sex_map.items():
            for _, ag_max, ag_label in age_groups:
                subset = folk_data[
                    (folk_data['Ålder'] == ag_label)
                    & (folk_data['Kön'] == sex_sv)
                ]
                if subset.empty:
                    continue
                
                counts = {}
                for _, row in subset.iterrows():
                    edu_sv = row['Utbildningsnivå']
                    edu_en = edu_level_map.get(edu_sv, 'unknown')
                    count = int(row['Antal']) if pd.notna(row['Antal']) else 0
                    counts[edu_en] = count
                
                total = sum(counts.values())
                if total > 0:
                    probs = {k: v / total for k, v in counts.items()}
                else:
                    # Fallback: equal distribution
                    n = len(counts)
                    probs = {k: 1.0 / n for k in counts} if n > 0 else {}
                
                prob_table[(ag_label, sex_en)] = probs
        
        if not prob_table:
            logger.warning("Could not build education probability table")
            return
        
        def _find_age_group(age: int) -> Optional[str]:
            """Find the matching age group label for a given age."""
            for ag_min, ag_max, ag_label in sorted(age_groups):
                if ag_min <= age <= ag_max:
                    return ag_label
            return None
        
        # Assign education to each agent
        edu_levels = list(edu_level_map.values())
        assigned = 0
        for agent in self.agents:
            if agent.age < 18:
                agent.education = 'child'
                continue
            
            ag_label = _find_age_group(agent.age)
            key = (ag_label, agent.sex)
            
            probs = prob_table.get(key)
            if probs:
                levels = list(probs.keys())
                weights = list(probs.values())
                agent.education = random.choices(levels, weights=weights, k=1)[0]
                assigned += 1
            else:
                # Fallback: use overall distribution for this sex
                fallback_probs = {}
                for (ag, sex), p in prob_table.items():
                    if sex == agent.sex:
                        for edu, prob in p.items():
                            fallback_probs[edu] = fallback_probs.get(edu, 0) + prob
                if fallback_probs:
                    total = sum(fallback_probs.values())
                    levels = list(fallback_probs.keys())
                    weights = [fallback_probs[l] / total for l in levels]
                    agent.education = random.choices(levels, weights=weights, k=1)[0]
                    assigned += 1
                else:
                    agent.education = 'unknown'
        
        logger.info(f"Assigned education level to {assigned} adults")

    # Age-based adjustment weights for income source assignment.
    #
    # The 20_HuvudInk_PRI.px table gives only sex-level marginals (no age
    # breakdown).  These weights encode the strong age dependence that is
    # inherent in the category definitions:
    #   • Pension – by definition retirement income (inkomstpension,
    #     garantipension, tjänstepension, …) → almost exclusively 65 +
    #   • Studies – studiestöd, barnbidrag vid förlängd skolgång → 18–29
    #   • Parental leave – föräldrapenning, tillfällig fp → peak 25–44
    #   • Disability – aktivitetsersättning 19–29, sjukersättning 30–64
    #   • Sickness – sjukpenning, rehab-penning → working-age
    #   • Work – kontant bruttolön etc. → broad working-age
    #   • Unemployment – a-kassa, aktivitetsstöd → working-age
    #   • Financial support – ekonomiskt stöd → all ages, skewed young
    #   • No income – kapitalinkomster, barnbidrag only, emigrated → any age
    #
    # Each weight is a relative multiplier applied to the sex-level
    # baseline probability; the product is normalised per age × sex group.
    _INCOME_SOURCE_AGE_WEIGHTS = {
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

    def _assign_income_source(self, income_source_data: pd.DataFrame) -> None:
        """
        Assign primary income source to adults based on census distributions.
        
        Uses the 20_HuvudInk_PRI.px table which provides **exact counts**
        by primary income source and sex for adults aged 20+.
        
        Algorithm (deterministic quota allocation):
        
        1. Extract exact census counts per (sex, source) — e.g. 2,800 males
           with "work", 850 males with "pension", etc.
        2. Scale those counts proportionally to match the actual number of
           synth agents per sex (census and synth may differ by a few %).
        3. Compute an **age-based affinity score** for each agent × source,
           using ``_INCOME_SOURCE_AGE_WEIGHTS``.  A 70-year-old gets a high
           affinity for "pension" and near-zero for "studies".
        4. For each source (from rarest to most common), greedily assign the
           agents with the highest affinity, drawing exactly the target count.
        5. The result matches the census sex-level marginal exactly while
           distributing sources across ages realistically.
        
        Income source categories (9):
        - work, unemployment, studies, pension, disability,
          sickness, parental_leave, financial_support, no_income
        
        Children (<20) get income_source=None.
        
        Args:
            income_source_data: DataFrame with columns Kön, 
                               Huvudsaklig inkomstkälla, Antal.
        """
        if income_source_data is None or income_source_data.empty:
            logger.warning("No income source data available, skipping")
            return
        
        # Map Swedish income source names to internal values
        source_map = {
            'Ersättning för arbete': 'work',
            'Ersättning vid arbetslöshet': 'unemployment',
            'Ersättning för studier': 'studies',
            'Pension': 'pension',
            'Ersättning vid långvarigt nedsatt arbetsförmåga': 'disability',
            'Ersättning vid sjukdom': 'sickness',
            'Ersättning vid föräldraledighet eller närståendeomvårdnad': 'parental_leave',
            'Ekonomiskt stöd': 'financial_support',
            'Saknar ersättningar': 'no_income',
        }
        
        # Map sex values
        sex_map = {'Man': 'male', 'Kvinna': 'female'}
        
        # Find columns
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
        
        count_col = 'Antal' if 'Antal' in income_source_data.columns else income_source_data.columns[-1]
        
        # ── Step 1: Extract exact census counts per (sex, source) ──
        census_counts = {}  # sex_en -> {source_en: count}
        for sex_sv, sex_en in sex_map.items():
            subset = income_source_data[income_source_data[sex_col] == sex_sv]
            if subset.empty:
                continue
            counts = {}
            for _, row in subset.iterrows():
                src_sv = row[source_col]
                src_en = source_map.get(src_sv, src_sv)
                count = int(row[count_col]) if pd.notna(row[count_col]) else 0
                counts[src_en] = counts.get(src_en, 0) + count
            census_counts[sex_en] = counts
        
        if not census_counts:
            logger.warning("Could not build income source count table")
            return
        
        # Set children to None
        for agent in self.agents:
            if agent.age < 20:
                agent.income_source = None
        
        # ── Step 2: Group adults by sex, scale census quotas ──
        adults_by_sex = {}  # sex -> [agent, ...]
        for agent in self.agents:
            if agent.age >= 20:
                adults_by_sex.setdefault(agent.sex, []).append(agent)
        
        assigned = 0
        for sex_en, agents in adults_by_sex.items():
            counts = census_counts.get(sex_en)
            if not counts:
                # No census data for this sex; fall back to uniform 'work'
                for a in agents:
                    a.income_source = 'work'
                    assigned += 1
                continue
            
            n_agents = len(agents)
            census_total = sum(counts.values())
            
            # Scale census counts to match synth agent count
            if census_total > 0 and census_total != n_agents:
                scale = n_agents / census_total
                quotas = {src: max(0, round(cnt * scale))
                          for src, cnt in counts.items()}
                # Fix rounding so sum == n_agents
                diff = n_agents - sum(quotas.values())
                if diff != 0:
                    # Adjust the largest category
                    largest = max(quotas, key=quotas.get)
                    quotas[largest] += diff
            else:
                quotas = dict(counts)
            
            # ── Step 3: Compute age affinity scores ──
            # For each agent, compute a score per source using age weights.
            # Add small random jitter to break ties.
            agent_scores = []  # [(agent_idx, {source: score})]
            for i, agent in enumerate(agents):
                scores = {}
                # Find matching age band
                matched_weights = None
                for (age_min, age_max), aw in self._INCOME_SOURCE_AGE_WEIGHTS.items():
                    if age_min <= agent.age <= age_max:
                        matched_weights = aw
                        break
                
                for src in quotas:
                    if matched_weights:
                        w = matched_weights.get(src, 1.0)
                    else:
                        w = 1.0
                    # Score = weight + small jitter for randomness within band
                    scores[src] = w + random.random() * 0.01
                agent_scores.append((i, scores))
            
            # ── Step 4: Greedy allocation from rarest to most common ──
            # Process rarest sources first so they get their best-fit agents.
            source_order = sorted(quotas.keys(), key=lambda s: quotas[s])
            
            remaining = set(range(n_agents))  # indices of unassigned agents
            
            for src in source_order:
                target = quotas[src]
                if target <= 0:
                    continue
                
                # Among remaining agents, pick the ones with highest score for src
                candidates = [(idx, agent_scores[idx][1][src]) for idx in remaining]
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                chosen = candidates[:target]
                for idx, _ in chosen:
                    agents[idx].income_source = src
                    remaining.discard(idx)
                    assigned += 1
            
            # Any remaining agents (rounding edge case) get 'work'
            for idx in remaining:
                agents[idx].income_source = 'work'
                assigned += 1
        
        logger.info(f"Assigned income source to {assigned} adults (20+) "
                     f"using deterministic quota allocation with age affinity")

    def _validate_synthesis(self) -> None:
        """Validate and clean up the synthesized population."""
        # Remove empty households (shouldn't happen but clean up if it does)
        empty_hhs = [h for h in self.households if len(h.members) == 0]
        if empty_hhs:
            logger.warning(f"Removing {len(empty_hhs)} empty households")
            self.households = [h for h in self.households if len(h.members) > 0]

        # Check all agents have households
        orphaned = [a for a in self.agents if a.household_id is None]
        if orphaned:
            logger.warning(f"{len(orphaned)} orphaned agents")

        # Log summary statistics
        logger.info(f"Final population: {len(self.agents)} individuals in {len(self.households)} households")
        logger.info(f"Average household size: {len(self.agents) / len(self.households):.2f}")

    # Helper methods for parsing and translating values

    def _parse_household_size(self, size_str: str) -> int:
        """Parse household size from string."""
        if isinstance(size_str, int):
            return size_str
        mapping = self.config.household_size_mappings
        return mapping.get(size_str, 1)

    def _parse_house_type(self, type_str: str) -> str:
        """Parse house type."""
        if pd.isna(type_str):
            return 'apartment'
        
        type_lower = str(type_str).lower()
        if 'småhus' in type_lower:
            return 'detached_house'
        elif 'special' in type_lower:
            return 'special_housing'
        else:
            return 'apartment'

    def _translate_sex(self, sex_str: str) -> str:
        """Translate sex from Swedish."""
        if pd.isna(sex_str):
            return 'male'
        
        sex_lower = str(sex_str).lower()
        if 'kvinn' in sex_lower or 'female' in sex_lower:
            return 'female'
        return 'male'

    def _translate_hh_role(self, role_str: str) -> str:
        """Translate household role from Swedish (aggregate HHtyp table)."""
        if pd.isna(role_str):
            return 'single'
        
        role_lower = str(role_str).lower()
        if 'samman' in role_lower or 'cohab' in role_lower:
            return 'cohabiting'
        elif 'övrig' in role_lower or 'other' in role_lower:
            return 'other'
        return 'single'
    
    def _translate_hh_position(self, position_str: str) -> str:
        """
        Translate detailed household position from Swedish.
        
        Handles positions from 60_FolkmHHStallning_PRI.px table:
        - Person i gift par/registrerat partnerskap -> cohabiting
        - Personer i samboförhållande -> cohabiting
        - Ensamstående förälder -> single_parent (must live with children)
        - Barn -> child
        - Ensamboende -> single (must live alone in 1-person HH)
        - Ej ensamboende personer, övriga -> other
        
        The distinction between 'single' and 'single_parent' is critical:
        - 'single' (Ensamboende) MUST go to 1-person households
        - 'single_parent' (Ensamstående förälder) MUST go to multi-person HH with children
        
        IMPORTANT: Check 'övrig' BEFORE 'ensamboende' because 
        "Ej ensamboende personer, övriga" contains both substrings!
        """
        if pd.isna(position_str):
            return 'single'
        
        pos_lower = str(position_str).lower()
        
        # Order matters - check more specific patterns first
        # CRITICAL: Check 'övrig' before 'ensamboende' because
        # "Ej ensamboende personer, övriga" matches BOTH!
        if 'barn' == pos_lower.strip():
            return 'child'
        elif 'övrig' in pos_lower:
            return 'other'  # Must check BEFORE ensamboende!
        elif 'ensamboende' in pos_lower:
            return 'single'  # Lives alone
        elif 'ensamstående förälder' in pos_lower:
            return 'single_parent'  # Lives with children - DIFFERENT from single!
        elif 'gift par' in pos_lower or 'partnerskap' in pos_lower:
            return 'cohabiting'
        elif 'sambo' in pos_lower:
            return 'cohabiting'
        elif 'uppgift saknas' in pos_lower:
            return 'unknown'
        
        return 'single'
    
    def _generate_individuals_from_position_data(self) -> List[Agent]:
        """
        Generate individuals directly from position data for EXACT role counts.
        
        This is more accurate than sampling from probabilities because it
        guarantees the exact number of singles, cohabiting, children, and
        other roles match the census data.
        
        Returns:
            List of Agent objects with exact role distribution from census.
        """
        position_data = self._household_position_data
        
        age_col = 'Ålder' if 'Ålder' in position_data.columns else 'age_group'
        sex_col = 'Kön' if 'Kön' in position_data.columns else 'sex'
        pos_col = 'Hushållsställning' if 'Hushållsställning' in position_data.columns else 'hh_position'
        count_col = 'Antal' if 'Antal' in position_data.columns else 'count'
        
        if count_col not in position_data.columns:
            for col in position_data.columns:
                if position_data[col].dtype in ['int64', 'float64']:
                    count_col = col
                    break
        
        individual_pool = []
        role_counts = {}  # For logging
        
        for _, row in position_data.iterrows():
            count = int(row[count_col]) if pd.notna(row[count_col]) else 0
            if count <= 0:
                continue
            
            age_group = row[age_col]
            sex_label = row[sex_col]
            position = row[pos_col]
            
            sex = self._translate_sex(sex_label)
            hh_role = self._translate_hh_position(position)
            
            if hh_role == 'unknown':
                continue  # Skip unknown positions
            
            role_counts[hh_role] = role_counts.get(hh_role, 0) + count
            
            for _ in range(count):
                age = self._sample_age_from_group(age_group)
                
                agent = Agent(
                    agent_id=self.next_agent_id,
                    age=age,
                    sex=sex,
                    hh_role=hh_role
                )
                individual_pool.append(agent)
                self.next_agent_id += 1
        
        logger.info(f"Exact role counts from position data: {role_counts}")
        return individual_pool
    
    def _generate_individuals_from_population_data(self, population_data: pd.DataFrame) -> List[Agent]:
        """
        Generate individuals from population data with role sampling (fallback).
        
        Used when detailed position data is not available.
        
        Args:
            population_data: DataFrame with age/sex/role counts
            
        Returns:
            List of Agent objects
        """
        age_col = 'Ålder' if 'Ålder' in population_data.columns else 'age_group'
        sex_col = 'Kön' if 'Kön' in population_data.columns else 'sex'
        role_col = 'Hushållstyp' if 'Hushållstyp' in population_data.columns else 'hh_role'
        count_col = 'Antal' if 'Antal' in population_data.columns else population_data.columns[-1]
        
        individual_pool = []
        
        for _, row in population_data.iterrows():
            count = int(row[count_col]) if pd.notna(row[count_col]) else 0
            if count <= 0:
                continue
            
            age_group = row[age_col]
            sex_label = row[sex_col]
            role_label = row[role_col]
            
            sex = self._translate_sex(sex_label)
            
            for _ in range(count):
                age = self._sample_age_from_group(age_group)
                
                # Use probability sampling if available
                if hasattr(self, '_role_probs') and self._role_probs:
                    hh_role = self._sample_role_for_agent(age, sex)
                else:
                    hh_role = self._translate_hh_role(role_label)
                    if age < 18 and hh_role != 'cohabiting':
                        hh_role = 'child'
                
                agent = Agent(
                    agent_id=self.next_agent_id,
                    age=age,
                    sex=sex,
                    hh_role=hh_role
                )
                individual_pool.append(agent)
                self.next_agent_id += 1
        
        return individual_pool
    
    def _build_role_probability_table(self, position_data: pd.DataFrame) -> None:
        """
        Build lookup table of role probabilities by age group and sex.
        
        This enables assigning roles based on actual census proportions
        rather than just the aggregate HHtyp (single/cohabiting/other).
        """
        self._role_probs = {}  # (age_group, sex) -> {role: probability}
        
        # Identify columns
        age_col = 'Ålder' if 'Ålder' in position_data.columns else 'age_group'
        sex_col = 'Kön' if 'Kön' in position_data.columns else 'sex'
        pos_col = 'Hushållsställning' if 'Hushållsställning' in position_data.columns else 'hh_position'
        count_col = 'Antal' if 'Antal' in position_data.columns else 'count'
        
        if count_col not in position_data.columns:
            # Try to find any numeric column
            for col in position_data.columns:
                if position_data[col].dtype in ['int64', 'float64']:
                    count_col = col
                    break
        
        # Aggregate by age/sex/position
        for (age_grp, sex), group in position_data.groupby([age_col, sex_col]):
            role_counts = {}
            total = 0
            
            for _, row in group.iterrows():
                position = row.get(pos_col, '')
                count = int(row.get(count_col, 0)) if pd.notna(row.get(count_col)) else 0
                
                if count <= 0:
                    continue
                
                role = self._translate_hh_position(position)
                if role != 'unknown':
                    role_counts[role] = role_counts.get(role, 0) + count
                    total += count
            
            # Convert to probabilities
            if total > 0:
                sex_eng = self._translate_sex(sex)
                self._role_probs[(age_grp, sex_eng)] = {
                    role: count / total for role, count in role_counts.items()
                }
        
        logger.info(f"Built role probability table for {len(self._role_probs)} age/sex groups")
    
    def _sample_role_for_agent(self, age: int, sex: str) -> str:
        """
        Sample a role for an agent based on census probabilities.
        
        Uses the household position data to determine the probability
        that someone of a given age and sex has each role.
        """
        if not hasattr(self, '_role_probs') or not self._role_probs:
            # No detailed data, fall back to simple age-based rule
            if age < 18:
                return 'child'
            return random.choice(['single', 'cohabiting'])
        
        # Find matching age group
        age_group = self._age_to_group(age)
        
        role_probs = self._role_probs.get((age_group, sex))
        if not role_probs:
            # Try alternative sex labels
            for key, probs in self._role_probs.items():
                if key[0] == age_group:
                    role_probs = probs
                    break
        
        if not role_probs:
            # Fall back to age-based default
            if age < 18:
                return 'child'
            return random.choice(['single', 'cohabiting'])
        
        # Weighted random choice
        roles = list(role_probs.keys())
        weights = [role_probs[r] for r in roles]
        return random.choices(roles, weights=weights, k=1)[0]
    
    def _age_to_group(self, age: int) -> str:
        """Convert age to census age group string."""
        # These match the 60_FolkmHHStallning_PRI.px age groups
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

    def _sample_age_from_group(self, age_group: str) -> int:
        """Sample a specific age from an age group range."""
        import re
        
        if pd.isna(age_group):
            return random.randint(25, 65)
        
        age_mappings = self.config.age_group_mappings
        
        # Try direct lookup
        if age_group in age_mappings:
            range_dict = age_mappings[age_group]
            return random.randint(range_dict['min'], range_dict['max'])
        
        age_str = str(age_group).strip()
        
        # Handle "X-Y år" format (e.g., "25-34 år")
        match = re.match(r'(\d+)-(\d+)\s*år', age_str)
        if match:
            min_age = int(match.group(1))
            max_age = int(match.group(2))
            return random.randint(min_age, max_age)
        
        # Handle "X- år" or "X+ år" format (open-ended, e.g., "85- år")
        match = re.match(r'(\d+)[-+]\s*år', age_str)
        if match:
            min_age = int(match.group(1))
            # Sample from min_age to some reasonable max (e.g., min_age + 15)
            return random.randint(min_age, min_age + 15)
        
        # Default
        return random.randint(25, 65)

    def _assign_cars(
        self,
        house_type: str,
        hh_size: int,
        car_data: Optional[pd.DataFrame]
    ) -> int:
        """
        Simple car assignment during household creation.
        Returns 0 - actual assignment happens later via propensity model.
        """
        # Initial assignment is 0 - propensity model assigns exact totals later
        return 0

    def _assign_cars_propensity(self, car_data: Optional[pd.DataFrame]) -> None:
        """
        Assign cars to households using a propensity-based model.
        
        This ensures the exact number of cars from census data is distributed
        to households based on socio-economic propensity scores.
        
        Propensity factors:
        - Household income level (higher income = higher propensity)
        - Family structure (families with children = higher propensity)  
        - Housing type (Småhus = highest, Specialbostad = lowest)
        - Household size (larger households = slightly higher propensity)
        
        Args:
            car_data: DataFrame with 'Personbilar' count from census
        """
        if not self.households:
            return
            
        # Extract target car count from data
        total_cars_target = 0
        if car_data is not None and not car_data.empty:
            # Find the 'Personbilar' row
            car_row = car_data[car_data['Tabellvärde'] == 'Personbilar']
            if not car_row.empty:
                count_col = 'NoContent' if 'NoContent' in car_data.columns else car_data.columns[-1]
                total_cars_target = int(car_row[count_col].iloc[0])
        
        if total_cars_target == 0:
            # Fallback: estimate based on population (0.19 cars/person is typical for Haga)
            total_pop = len(self.agents)
            total_cars_target = int(total_pop * 0.19)
            logger.info(f"No car data - estimating {total_cars_target} cars based on population")
        else:
            logger.info(f"Target car count from census: {total_cars_target}")
        
        # Calculate propensity score for each household
        hh_scores = []
        
        for hh in self.households:
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
            has_young_children = any(
                m.age <= 5 for m in hh.members
            )
            has_school_children = any(
                6 <= m.age <= 15 for m in hh.members
            )
            if has_young_children:
                score += 4.0  # High priority - need car for daycare/errands
            if has_school_children:
                score += 2.5  # Moderate priority - school activities
            
            # 3. Income Factor
            # Check if household has low income flag
            has_low_income = any(
                getattr(m, 'low_income', False) for m in hh.members
            )
            if has_low_income:
                score -= 2.0
            else:
                # Check income attribute if available (handle None values)
                incomes = [getattr(m, 'income', 0) or 0 for m in hh.members]
                avg_income = sum(incomes) / len(incomes) if incomes else 0
                if avg_income > 400000:  # Higher income threshold
                    score += 2.0
                elif avg_income > 300000:
                    score += 1.0
            
            # 4. Household Size Factor
            if hh.size >= 3:
                score += 1.0
            elif hh.size == 1:
                score -= 0.5  # Singles less likely to own cars in urban areas
            
            # 5. Age Factor - working age adults more likely
            working_age_adults = sum(
                1 for m in hh.members if 25 <= m.age <= 64
            )
            score += working_age_adults * 0.5
            
            # Elderly single households less likely
            if hh.size == 1 and any(m.age >= 75 for m in hh.members):
                score -= 2.0
            
            hh_scores.append((hh, max(score, 0.1)))  # Ensure positive score
        
        # Sort by score with stochastic noise for realistic variation
        hh_scores.sort(
            key=lambda x: x[1] * random.uniform(0.8, 1.2),
            reverse=True
        )
        
        # Distribute cars - highest propensity households get cars first
        cars_distributed = 0
        
        # First pass: assign one car to highest propensity households
        for hh, score in hh_scores:
            if cars_distributed >= total_cars_target:
                break
            hh.cars = 1
            cars_distributed += 1
        
        # Second pass: some high-propensity households get a second car
        # (typically Småhus with families)
        if cars_distributed < total_cars_target:
            for hh, score in hh_scores:
                if cars_distributed >= total_cars_target:
                    break
                # Only give second car to high-scoring households
                if score > 10.0 and hh.cars == 1:
                    hh.cars = 2
                    cars_distributed += 1
        
        # Ensure any remaining capacity is filled
        remaining = total_cars_target - cars_distributed
        if remaining > 0:
            for hh, score in hh_scores:
                if remaining <= 0:
                    break
                if hh.cars < 2:  # Max 2 cars per household
                    hh.cars += 1
                    remaining -= 1
        
        # Log statistics
        total_assigned = sum(hh.cars for hh in self.households)
        hh_with_cars = sum(1 for hh in self.households if hh.cars > 0)
        logger.info(f"Assigned {total_assigned} cars to {hh_with_cars} households "
                   f"({100*hh_with_cars/len(self.households):.1f}% car ownership)")

    def _parse_income_decile(self, decile_str: str) -> Optional[int]:
        """Parse income decile from string."""
        if pd.isna(decile_str):
            return None
        
        # Try to extract number
        import re
        match = re.search(r'\d+', str(decile_str))
        if match:
            return int(match.group())
        return None

    def _estimate_income_from_decile(self, decile: int) -> float:
        """Estimate income amount from decile (rough Swedish estimates)."""
        # Simplified Swedish income distribution (SEK per year)
        decile_estimates = {
            1: 150000,
            2: 200000,
            3: 250000,
            4: 300000,
            5: 350000,
            6: 400000,
            7: 450000,
            8: 550000,
            9: 700000,
            10: 1000000
        }
        base = decile_estimates.get(decile, 350000)
        # Add some randomness
        return base * random.uniform(0.9, 1.1)

    # =========================================================================
    # Housing Type Assignment Methods
    # =========================================================================

    def assign_housing_types(
        self,
        hh_hustyp_dist: pd.DataFrame,
        households: Optional[List[Household]] = None
    ) -> List[Household]:
        """
        Assign Swedish house types (Hustyp) to households based on size distribution.

        Uses table 31_HHStorlHustyp_PRI.px data to probabilistically assign
        house types based on the observed distribution for each household size.

        Args:
            hh_hustyp_dist: DataFrame from table 31_HHStorlHustyp_PRI.px with columns:
                           - Hushållsstorlek (household size)
                           - Hustyp (house type: Småhus, Flerbostadshus, Specialbostad)
                           - Count/Antal (number of households)
            households: Optional list of households to assign (uses self.households if None)

        Returns:
            List of households with assigned_hustyp populated

        Example:
            >>> synthesizer.assign_housing_types(household_data)
            >>> for hh in synthesizer.households:
            ...     print(f"HH {hh.household_id}: {hh.assigned_hustyp}")
        """
        if households is None:
            households = self.households

        if households is None or len(households) == 0:
            logger.warning("No households to assign housing types to")
            return []

        # Build probability distributions by household size
        size_distributions = self._build_hustyp_distributions(hh_hustyp_dist)

        assigned_count = {
            'Småhus': 0,
            'Flerbostadshus': 0,
            'Specialbostad': 0,
            'unknown': 0
        }

        for hh in households:
            if hh.size in size_distributions:
                dist = size_distributions[hh.size]
                hustyp = random.choices(
                    list(dist.keys()),
                    weights=list(dist.values())
                )[0]
            else:
                # Default to apartment for unknown sizes
                hustyp = 'Flerbostadshus'
                logger.debug(f"No distribution for size {hh.size}, defaulting to Flerbostadshus")

            hh.assigned_hustyp = hustyp
            assigned_count[hustyp] = assigned_count.get(hustyp, 0) + 1

        logger.info(f"Assigned housing types: {assigned_count}")
        return households

    def _build_hustyp_distributions(
        self,
        hh_hustyp_dist: pd.DataFrame
    ) -> Dict[int, Dict[str, float]]:
        """
        Build probability distributions for house types by household size.

        Args:
            hh_hustyp_dist: DataFrame with Hushållsstorlek, Hustyp, and count columns

        Returns:
            Dictionary mapping household size to {hustyp: probability}
        """
        distributions: Dict[int, Dict[str, float]] = {}

        # Identify column names (handle Swedish/English variations)
        size_col = None
        type_col = None
        count_col = None

        for col in hh_hustyp_dist.columns:
            col_lower = col.lower()
            if 'storlek' in col_lower or 'size' in col_lower:
                size_col = col
            elif 'hustyp' in col_lower or 'house' in col_lower:
                type_col = col
            elif 'antal' in col_lower or 'count' in col_lower:
                count_col = col

        # Fallback to last column as count
        if count_col is None:
            count_col = hh_hustyp_dist.columns[-1]

        if size_col is None or type_col is None:
            logger.warning("Could not identify size/type columns in household data")
            return distributions

        # Group by size and type
        for _, row in hh_hustyp_dist.iterrows():
            size = self._parse_household_size(row[size_col])
            hustyp = str(row[type_col])
            count = float(row[count_col]) if pd.notna(row[count_col]) else 0

            if size not in distributions:
                distributions[size] = {}

            distributions[size][hustyp] = distributions[size].get(hustyp, 0) + count

        # Normalize to probabilities
        for size in distributions:
            total = sum(distributions[size].values())
            if total > 0:
                distributions[size] = {
                    k: v / total for k, v in distributions[size].items()
                }

        logger.debug(f"Built housing type distributions for sizes: {list(distributions.keys())}")
        return distributions

    def link_to_buildings(
        self,
        buildings: pd.DataFrame,
        households: Optional[List[Household]] = None,
        building_id_col: str = 'building_id',
        building_type_col: str = 'type',
        capacity_col: Optional[str] = None,
        income_weighted: bool = False
    ) -> List[Household]:
        """
        Link households to building footprints based on house type matching.

        Assigns each household to a specific building polygon, respecting
        house type compatibility and optional capacity constraints.

        Args:
            buildings: GeoDataFrame or DataFrame with building footprints containing:
                      - building_id: Unique building identifier
                      - type: Building type ('Småhus', 'villa', 'apartment', etc.)
                      - capacity (optional): Max households per building
            households: Optional list of households (uses self.households if None)
            building_id_col: Column name for building ID
            building_type_col: Column name for building type
            capacity_col: Optional column name for building capacity
            income_weighted: If True, assign higher-income households to
                           Småhus preferentially

        Returns:
            List of households with building_id populated

        Example:
            >>> buildings = gpd.read_file("buildings.shp")
            >>> synthesizer.link_to_buildings(buildings)
        """
        if households is None:
            households = self.households

        if households is None or len(households) == 0:
            logger.warning("No households to link to buildings")
            return []

        # Ensure all households have assigned_hustyp
        unassigned = [hh for hh in households if hh.assigned_hustyp is None]
        if unassigned:
            logger.warning(f"{len(unassigned)} households missing assigned_hustyp, using house_type")
            for hh in unassigned:
                hh.assigned_hustyp = self._english_to_swedish_hustyp(hh.house_type)

        # Map building types to Swedish Hustyp categories
        buildings = buildings.copy()
        buildings['_hustyp'] = buildings[building_type_col].apply(self._normalize_building_type)

        # Track building occupancy
        building_occupancy: Dict[int, int] = {}
        building_capacity: Dict[int, int] = {}

        for _, bldg in buildings.iterrows():
            bid = bldg[building_id_col]
            if capacity_col and capacity_col in bldg:
                building_capacity[bid] = int(bldg[capacity_col])
            else:
                # Estimate capacity: Småhus = 1, Flerbostadshus = based on size
                if bldg['_hustyp'] == 'Småhus':
                    building_capacity[bid] = 1
                else:
                    building_capacity[bid] = 999  # Unlimited for apartments
            building_occupancy[bid] = 0

        # Sort households by income if weighted assignment requested
        if income_weighted:
            households = sorted(
                households,
                key=lambda h: h.income,
                reverse=True
            )

        # Assign households to buildings by house type
        for hh in households:
            target_hustyp = hh.assigned_hustyp or 'Flerbostadshus'

            # Find compatible buildings with capacity
            compatible = buildings[
                (buildings['_hustyp'] == target_hustyp) &
                (buildings[building_id_col].apply(
                    lambda bid: building_occupancy.get(bid, 0) < building_capacity.get(bid, 1)
                ))
            ]

            if len(compatible) == 0:
                # Fallback: try any building with capacity
                compatible = buildings[
                    buildings[building_id_col].apply(
                        lambda bid: building_occupancy.get(bid, 0) < building_capacity.get(bid, 1)
                    )
                ]
                if len(compatible) > 0:
                    logger.debug(
                        f"No {target_hustyp} buildings available for HH {hh.household_id}, "
                        f"using fallback"
                    )

            if len(compatible) > 0:
                # Random assignment among compatible buildings
                selected_idx = random.choice(compatible.index.tolist())
                selected_bid = compatible.loc[selected_idx, building_id_col]

                hh.building_id = selected_bid
                building_occupancy[selected_bid] = building_occupancy.get(selected_bid, 0) + 1
            else:
                logger.warning(f"No available buildings for household {hh.household_id}")

        # Log summary
        assigned = sum(1 for hh in households if hh.building_id is not None)
        logger.info(f"Linked {assigned}/{len(households)} households to buildings")

        return households

    def _normalize_building_type(self, building_type: str) -> str:
        """
        Normalize building type strings to Swedish Hustyp categories.

        Args:
            building_type: Input building type string

        Returns:
            Standardized Swedish Hustyp: 'Småhus', 'Flerbostadshus', or 'Specialbostad'
        """
        if pd.isna(building_type):
            return 'Flerbostadshus'

        type_lower = str(building_type).lower()

        # Småhus variants
        if any(term in type_lower for term in [
            'småhus', 'villa', 'small house', 'detached', 'semi-detached',
            'radhus', 'townhouse', 'single-family', 'enfamiljs'
        ]):
            return 'Småhus'

        # Specialbostad variants
        if any(term in type_lower for term in [
            'special', 'student', 'elderly', 'äldre', 'gruppbostad',
            'servicehus', 'care'
        ]):
            return 'Specialbostad'

        # Default to Flerbostadshus (apartment building)
        return 'Flerbostadshus'

    def _english_to_swedish_hustyp(self, house_type: Optional[str]) -> str:
        """
        Convert English house type to Swedish Hustyp.

        Args:
            house_type: English house type string

        Returns:
            Swedish Hustyp category
        """
        if house_type is None:
            return 'Flerbostadshus'

        mapping = {
            'detached_house': 'Småhus',
            'apartment': 'Flerbostadshus',
            'special_housing': 'Specialbostad'
        }
        return mapping.get(house_type, 'Flerbostadshus')

    def estimate_building_units(
        self,
        building_area: float,
        building_height: float,
        floors: Optional[int] = None,
        avg_unit_area: float = 80.0,
        floor_height: float = 3.0
    ) -> int:
        """
        Estimate number of dwelling units in a building based on physical dimensions.

        Uses the formula: Units = (Building Area × Floors) / Average Unit Area

        Args:
            building_area: Building footprint area in m²
            building_height: Building height in meters
            floors: Number of floors (calculated from height if None)
            avg_unit_area: Average apartment size in m² (default 80)
            floor_height: Average floor height in meters (default 3.0)

        Returns:
            Estimated number of dwelling units

        Example:
            >>> # A 500m² footprint building, 15m tall
            >>> units = synthesizer.estimate_building_units(500, 15)
            >>> print(units)  # ~31 units
        """
        if floors is None:
            floors = max(1, int(building_height / floor_height))

        total_area = building_area * floors
        units = int(total_area / avg_unit_area)

        return max(1, units)  # At least 1 unit
