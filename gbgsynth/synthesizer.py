"""
Population synthesis engine implementing the greedy matching algorithm.

This module contains the core logic for generating synthetic individuals
and households based on census marginals.
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
    Implements greedy matching algorithm for synthetic population generation.

    The algorithm proceeds in phases:
    1. Create empty households based on size/type statistics
    2. Generate individual pool from demographic statistics
    3. Form couples in multi-person households
    4. Assign children to couple/single-parent households
    5. Fill single-person households
    6. Assign socioeconomic attributes
    """

    def __init__(self, config: Optional[Config] = None, use_ipf: bool = False, 
                 use_constrained_ipf: bool = False, use_topdown: bool = True):
        """
        Initialize the synthesizer.

        Args:
            config: Configuration object (will create default if None)
            use_ipf: If True, use IPF for better marginal fitting
            use_constrained_ipf: If True, use constrained IPF that generates
                                 valid household compositions directly
            use_topdown: If True, use top-down constrained synthesis that
                        anchors exact household containers first, then fills
                        with individuals. Best balance of structure + demographics.
        """
        self.config = config or Config()
        self.constraints = self.config.constraints
        self.use_ipf = use_ipf
        self.use_constrained_ipf = use_constrained_ipf
        self.use_topdown = use_topdown

        # Synthesis state
        self.agents: List[Agent] = []
        self.households: List[Household] = []
        self.next_agent_id = 1
        self.next_household_id = 1
        
        # IPF results (if used)
        self.ipf_stats: Dict = {}

    def synthesize(
        self,
        population_data: pd.DataFrame,
        household_data: pd.DataFrame,
        income_data: Optional[pd.DataFrame] = None,
        car_data: Optional[pd.DataFrame] = None,
        buildings: Optional[pd.DataFrame] = None,
        household_position_data: Optional[pd.DataFrame] = None
    ) -> Tuple[List[Agent], List[Household]]:
        """
        Main synthesis method coordinating all phases.

        Args:
            population_data: DataFrame with age/sex/hh_role counts
            household_data: DataFrame with household size/type counts
            income_data: Optional DataFrame with income distribution
            car_data: Optional DataFrame with car ownership
            buildings: Optional DataFrame/GeoDataFrame with building footprints
                      for spatial linking
            household_position_data: Optional DataFrame with detailed household 
                      positions (including child role) by age/sex

        Returns:
            Tuple of (agents, households)
        """
        logger.info("Starting population synthesis")
        
        # Store household position data for role assignment
        self._household_position_data = household_position_data
        if household_position_data is not None:
            self._build_role_probability_table(household_position_data)
            logger.info("Built role probability table from household position data")

        if self.use_topdown:
            # Top-down constrained synthesis: anchor households first, then fill
            logger.info("Using top-down constrained synthesis")
            self._synthesize_topdown(population_data, household_data, car_data)
        elif self.use_constrained_ipf:
            # Use constrained IPF that generates complete valid households
            logger.info("Using constrained IPF synthesis")
            self._synthesize_with_constrained_ipf(population_data, household_data, car_data)
        elif self.use_ipf:
            # Use IPF-based synthesis for better marginal fitting
            logger.info("Using IPF-based synthesis")
            self._synthesize_with_ipf(population_data, household_data, car_data)
        else:
            # Original greedy matching approach
            logger.info("Using greedy matching synthesis")
            
            # Phase 1: Create households
            self._create_households(household_data, car_data)
            logger.info(f"Created {len(self.households)} households")

            # Phase 2: Generate individual pool
            individual_pool = self._generate_individual_pool(population_data)
            logger.info(f"Generated pool of {len(individual_pool)} individuals")

            # Phase 3: Greedy matching
            self._match_individuals_to_households(individual_pool)
        
        logger.info(f"Matched {len(self.agents)} individuals to households")

        # Phase 4: Assign socioeconomic attributes
        if income_data is not None:
            self._assign_income(income_data)

        # Phase 5: Assign housing types (Hustyp) based on household size distribution
        self.assign_housing_types(household_data)
        logger.info("Assigned housing types to households")

        # Phase 6: Assign cars using propensity model with exact target
        self._assign_cars_propensity(car_data)
        logger.info("Assigned cars to households using propensity model")

        # Phase 7: Link to building footprints (if provided)
        if buildings is not None:
            self.link_to_buildings(buildings)
            logger.info("Linked households to building footprints")

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
        
        age_col = 'Ålder' if 'Ålder' in population_data.columns else 'age_group'
        sex_col = 'Kön' if 'Kön' in population_data.columns else 'sex'
        role_col = 'Hushållstyp' if 'Hushållstyp' in population_data.columns else 'hh_role'
        
        # Generate individuals exactly matching the marginal counts
        individual_pool = []
        
        for _, row in population_data.iterrows():
            count = int(row[count_col]) if pd.notna(row[count_col]) else 0
            if count <= 0:
                continue
            
            age_group = row[age_col]
            sex_label = row[sex_col]
            role_label = row[role_col]
            
            sex = self._translate_sex(sex_label)
            hh_role = self._translate_hh_role(role_label)
            
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
        
        logger.info(f"Generated {len(individual_pool)} individuals from marginals")
        
        # Categorize individuals by role
        singles = [a for a in individual_pool if a.hh_role == 'single']
        cohabiting = [a for a in individual_pool if a.hh_role == 'cohabiting']
        children = [a for a in individual_pool if a.hh_role == 'child']
        other = [a for a in individual_pool if a.hh_role not in ['single', 'cohabiting', 'child']]
        
        logger.info(f"Pool: {len(singles)} singles, {len(cohabiting)} cohabiting, "
                   f"{len(children)} children, {len(other)} other")
        
        # === Phase 3: Constrained assignment ===
        logger.info("Top-down Phase 3: Constrained assignment to containers")
        
        # Sort households by size descending (fill largest first)
        containers_by_size = sorted(household_containers, key=lambda h: h.size, reverse=True)
        
        # Separate into single-person and multi-person households
        single_hh = [h for h in containers_by_size if h.size == 1]
        multi_hh = [h for h in containers_by_size if h.size >= 2]
        
        # Shuffle for randomization
        random.shuffle(singles)
        random.shuffle(cohabiting)
        random.shuffle(children)
        random.shuffle(other)
        random.shuffle(multi_hh)
        
        # Step 3a: Form couples in multi-person households
        logger.info("Step 3a: Forming couples in multi-person households")
        couples_formed = self._form_couples_topdown(cohabiting, multi_hh)
        logger.info(f"Formed {couples_formed} couples")
        
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
        
        # Calculate capacity stats
        total_capacity = sum(h.size for h in self.households)
        total_placed = len(self.agents)
        if total_capacity > 0:
            logger.info(f"Top-down synthesis complete: {total_placed}/{total_capacity} slots filled "
                       f"({100*total_placed/total_capacity:.1f}%)")
        else:
            logger.warning("Top-down synthesis complete: no capacity (area may have zero population)")
        
        # Store stats
        self.ipf_stats = {
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
            if not hh.has_capacity(2):
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
    
    def _place_children_topdown(self, children: List[Agent], multi_hh: List[Household]) -> int:
        """Place children in households that have adults."""
        min_parent_gap = self.constraints.get('parent_child_age_gap_min', 18)
        
        # Sort children youngest first
        children = sorted(children, key=lambda c: c.age)
        
        placed = 0
        for child in children:
            # Find a household with capacity and an adult old enough
            for hh in multi_hh:
                if not hh.has_capacity():
                    continue
                
                # Check if there's a suitable parent
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
                if hh.has_capacity():
                    hh.add_member(agent)
                    placed += 1
                    break
        
        return placed
    
    def _fill_remaining_slots_topdown(self, remaining: List[Agent], multi_hh: List[Household]) -> int:
        """Fill remaining slots in multi-person households."""
        placed = 0
        
        for agent in remaining:
            for hh in multi_hh:
                if hh.has_capacity():
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
            if hh.has_capacity():
                agent = singles.pop()
                hh.add_member(agent)
                placed += 1
        
        return placed
    
    def _redistribute_unplaced_topdown(self, unplaced: List[Agent], all_hh: List[Household]) -> None:
        """
        Redistribute unplaced individuals to any household with capacity.
        
        This is a last resort - ideally we've sized everything correctly.
        """
        for agent in unplaced:
            placed = False
            
            # Try to find any household with remaining capacity
            for hh in all_hh:
                if hh.has_capacity():
                    hh.add_member(agent)
                    placed = True
                    break
            
            if not placed:
                # Log but don't create new households - this maintains exact HH count
                logger.warning(f"Could not place agent {agent.agent_id} (age={agent.age}, "
                             f"role={agent.hh_role}) - no capacity available")

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
            if not hh.has_capacity(2):
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
        multi_hhs_with_adults = [hh for hh in multi_hhs if len(hh.members) > 0 and hh.has_capacity()]
        multi_hhs_with_adults = sorted(multi_hhs_with_adults, 
                                       key=lambda h: h.size - len(h.members), reverse=True)
        
        for child in children:
            if child.household_id is not None:
                continue
            
            for hh in multi_hhs_with_adults:
                if not hh.has_capacity():
                    continue
                
                # Check for suitable parent
                head = hh.get_head()
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
            if not hh.has_capacity():
                continue
            for adult in remaining_cohab:
                if adult.household_id is None and hh.has_capacity():
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
                if hh.has_capacity():
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
            if hh.has_capacity(2):
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
                if not hh.has_capacity():
                    continue

                # Check if household has adults who can be parents
                head = hh.get_head()
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

    def _assign_income(self, income_data: pd.DataFrame) -> None:
        """
        Assign income to agents based on neighborhood income standard distribution.
        
        The income data from SCB uses "Inkomststandard" categories:
        - Low income ("har låg inkomststandard")
        - Not low income ("inte har låg inkomststandard")

        Args:
            income_data: DataFrame with income standard distribution
        """
        # Calculate low income probability from actual data
        low_income_prob = self._calculate_low_income_probability(income_data)
        
        logger.info(f"Low income probability from marginals: {low_income_prob:.1%}")

        for agent in self.agents:
            # Assign income standard based on marginal probability
            is_low_income = random.random() < low_income_prob
            
            if is_low_income:
                agent.income_decile = random.randint(1, 2)  # Low income = decile 1-2
            else:
                agent.income_decile = random.randint(3, 10)  # Not low income = decile 3-10
            
            agent.income = self._estimate_income_from_decile(agent.income_decile)
            agent.income_standard = 'low' if is_low_income else 'not_low'

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

    def _validate_synthesis(self) -> None:
        """Validate the synthesized population."""
        # Check all households have members
        empty_hhs = [h for h in self.households if len(h.members) == 0]
        if empty_hhs:
            logger.warning(f"{len(empty_hhs)} empty households")

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
        """Translate household role from Swedish."""
        if pd.isna(role_str):
            return 'single'
        
        role_lower = str(role_str).lower()
        if 'samman' in role_lower or 'cohab' in role_lower:
            return 'cohabiting'
        return 'single'

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
                key=lambda h: h.get_household_income(),
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
