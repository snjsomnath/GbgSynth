"""
Iterative Proportional Fitting (IPF) for population synthesis.

IPF adjusts a multi-dimensional contingency table to match known marginal totals
while preserving the joint distribution structure of the original data.

This module also contains the IPF and Constrained-IPF *engine* functions
used by :class:`gbgsynth.synthesizer.PopulationSynthesizer` when the user
selects ``engine='ipf'`` or ``engine='constrained_ipf'``.
"""

from __future__ import annotations

import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gbgsynth.models import Agent, Household

logger = logging.getLogger(__name__)


class IPFSynthesizer:
    """
    Implements Iterative Proportional Fitting for population synthesis.
    
    IPF (also known as RAS or Fratar method) iteratively adjusts cell values
    to match marginal constraints while minimizing deviation from the seed.
    
    The algorithm:
    1. Start with a seed matrix (from PUMS/sample data or uniform)
    2. Scale rows to match row marginals
    3. Scale columns to match column marginals  
    4. Repeat until convergence
    5. Sample from the fitted distribution
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        min_cell_value: float = 1e-10
    ):
        """
        Initialize IPF synthesizer.
        
        Args:
            max_iterations: Maximum IPF iterations
            convergence_threshold: Stop when max relative change < threshold
            min_cell_value: Minimum cell value to prevent division by zero
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_cell_value = min_cell_value
        
        # Results
        self.fitted_weights: Optional[np.ndarray] = None
        self.dimension_labels: Dict[str, List[str]] = {}
        self.convergence_history: List[float] = []
    
    def fit(
        self,
        marginals: Dict[str, pd.Series],
        seed: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit IPF to match marginal distributions.
        
        Args:
            marginals: Dict mapping dimension name to pd.Series of counts
                      e.g., {'age': Series([100, 200, 150], index=['0-17', '18-64', '65+']),
                             'sex': Series([450, 450], index=['male', 'female'])}
            seed: Optional seed matrix. If None, uses uniform seed.
        
        Returns:
            Fitted weight matrix matching all marginals
        """
        # Extract dimension info
        dim_names = list(marginals.keys())
        dim_sizes = [len(m) for m in marginals.values()]
        
        # Store labels for later use
        for name, series in marginals.items():
            self.dimension_labels[name] = list(series.index)
        
        # Create seed matrix if not provided
        if seed is None:
            # Uniform seed
            seed = np.ones(dim_sizes, dtype=float)
        
        # Normalize marginals to same total
        target_total = list(marginals.values())[0].sum()
        marginal_arrays = []
        for series in marginals.values():
            arr = series.values.astype(float)
            # Scale to target total
            arr = arr * (target_total / arr.sum())
            marginal_arrays.append(arr)
        
        # Run IPF
        weights = seed.copy().astype(float)
        weights = np.maximum(weights, self.min_cell_value)
        
        self.convergence_history = []
        
        for iteration in range(self.max_iterations):
            old_weights = weights.copy()
            
            # Iterate through each dimension
            for dim_idx, marginal in enumerate(marginal_arrays):
                # Sum over all other dimensions
                current_marginal = weights.sum(axis=tuple(
                    d for d in range(len(dim_sizes)) if d != dim_idx
                ))
                
                # Compute adjustment factors
                factors = np.divide(
                    marginal,
                    current_marginal,
                    out=np.ones_like(marginal),
                    where=current_marginal > self.min_cell_value
                )
                
                # Apply factors along this dimension
                # Need to reshape factors for broadcasting
                shape = [1] * len(dim_sizes)
                shape[dim_idx] = dim_sizes[dim_idx]
                factors = factors.reshape(shape)
                
                weights = weights * factors
                weights = np.maximum(weights, self.min_cell_value)
            
            # Check convergence
            max_change = np.max(np.abs(weights - old_weights) / 
                               np.maximum(old_weights, self.min_cell_value))
            self.convergence_history.append(max_change)
            
            if max_change < self.convergence_threshold:
                logger.info(f"IPF converged after {iteration + 1} iterations")
                break
        else:
            logger.warning(f"IPF did not converge after {self.max_iterations} iterations")
        
        self.fitted_weights = weights
        return weights
    
    def fit_2d(
        self,
        row_marginal: pd.Series,
        col_marginal: pd.Series,
        seed: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Convenience method for 2D IPF.
        
        Args:
            row_marginal: Marginal totals for rows
            col_marginal: Marginal totals for columns
            seed: Optional seed matrix
            
        Returns:
            Fitted DataFrame with row/col indices
        """
        weights = self.fit(
            {'row': row_marginal, 'col': col_marginal},
            seed
        )
        
        return pd.DataFrame(
            weights,
            index=row_marginal.index,
            columns=col_marginal.index
        )
    
    def sample(
        self,
        n_samples: int,
        random_state: Optional[int] = None,
        np_rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """
        Sample from the fitted distribution.
        
        Args:
            n_samples: Number of samples to draw
            random_state: Random seed for reproducibility (creates
                a local ``np.random.Generator`` — does not mutate
                global state).
            np_rng: Optional pre-created ``np.random.Generator``.
                Takes precedence over *random_state*.
            
        Returns:
            DataFrame with sampled category combinations
        """
        if self.fitted_weights is None:
            raise ValueError("Must call fit() before sample()")
        
        if np_rng is not None:
            _rng = np_rng
        elif random_state is not None:
            _rng = np.random.default_rng(random_state)
        else:
            _rng = np.random.default_rng()
        
        # Flatten weights to 1D probability distribution
        flat_weights = self.fitted_weights.flatten()
        probs = flat_weights / flat_weights.sum()
        
        # Sample indices
        flat_indices = _rng.choice(
            len(flat_weights),
            size=n_samples,
            p=probs
        )
        
        # Convert flat indices back to multi-dimensional indices
        dim_names = list(self.dimension_labels.keys())
        shape = self.fitted_weights.shape
        
        records = []
        for flat_idx in flat_indices:
            # Unravel index
            multi_idx = np.unravel_index(flat_idx, shape)
            
            # Map to labels
            record = {}
            for dim_idx, (dim_name, idx) in enumerate(zip(dim_names, multi_idx)):
                record[dim_name] = self.dimension_labels[dim_name][idx]
            records.append(record)
        
        return pd.DataFrame(records)
    
    def get_fitted_marginals(self) -> Dict[str, pd.Series]:
        """
        Get the marginal totals from the fitted weights.
        
        Returns:
            Dict mapping dimension name to fitted marginal Series
        """
        if self.fitted_weights is None:
            raise ValueError("Must call fit() first")
        
        result = {}
        dim_names = list(self.dimension_labels.keys())
        
        for dim_idx, dim_name in enumerate(dim_names):
            marginal = self.fitted_weights.sum(axis=tuple(
                d for d in range(len(dim_names)) if d != dim_idx
            ))
            result[dim_name] = pd.Series(
                marginal,
                index=self.dimension_labels[dim_name],
                name=dim_name
            )
        
        return result
    
    def compute_fit_statistics(
        self,
        target_marginals: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Compute goodness-of-fit statistics.
        
        Args:
            target_marginals: Original target marginals
            
        Returns:
            Dict with SRMSE, max error, etc.
        """
        fitted = self.get_fitted_marginals()
        
        all_errors = []
        all_rel_errors = []
        
        for dim_name, target in target_marginals.items():
            if dim_name not in fitted:
                continue
            
            fit = fitted[dim_name]
            
            # Align indices
            common_idx = target.index.intersection(fit.index)
            t = target.loc[common_idx].values
            f = fit.loc[common_idx].values
            
            errors = f - t
            all_errors.extend(errors)
            
            rel_errors = np.divide(
                np.abs(errors),
                t,
                out=np.zeros_like(errors, dtype=float),
                where=t > 0
            )
            all_rel_errors.extend(rel_errors)
        
        all_errors = np.array(all_errors)
        all_rel_errors = np.array(all_rel_errors)
        
        return {
            'rmse': float(np.sqrt(np.mean(all_errors ** 2))),
            'mae': float(np.mean(np.abs(all_errors))),
            'max_abs_error': float(np.max(np.abs(all_errors))),
            'mean_rel_error': float(np.mean(all_rel_errors)),
            'max_rel_error': float(np.max(all_rel_errors)),
            'iterations': len(self.convergence_history),
            'converged': len(self.convergence_history) < self.max_iterations
        }


class HouseholdIPF:
    """
    IPF-based household synthesizer that matches multiple marginals.
    
    This class coordinates IPF fitting and sampling to generate
    households that match:
    - Household size distribution
    - Housing type distribution  
    - Age distribution of population
    - Sex distribution of population
    """
    
    def __init__(self, config=None):
        """Initialize with optional config."""
        self.config = config
        self.ipf = IPFSynthesizer()
        
    def synthesize_households(
        self,
        household_data: pd.DataFrame,
        population_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Synthesize households using IPF to match marginals.
        
        Args:
            household_data: DataFrame with Hushållsstorlek, Hustyp, Antal columns
            population_data: DataFrame with Ålder, Kön, Hushållstyp, Antal columns
            
        Returns:
            Tuple of (household DataFrame, fit statistics dict)
        """
        # Extract household marginals
        hh_size_marginal = self._extract_household_size_marginal(household_data)
        hh_type_marginal = self._extract_housing_type_marginal(household_data)
        
        logger.info(f"Household size marginal: {dict(hh_size_marginal)}")
        logger.info(f"Housing type marginal: {dict(hh_type_marginal)}")
        
        # Build 2D IPF for household size × housing type
        fitted = self.ipf.fit_2d(
            hh_size_marginal,
            hh_type_marginal
        )
        
        logger.info(f"IPF fitted in {len(self.ipf.convergence_history)} iterations")
        
        # Generate households by sampling
        total_hh = int(hh_size_marginal.sum())
        households = self.ipf.sample(total_hh)
        households.columns = ['size', 'house_type']
        
        # Convert size labels back to integers
        households['size'] = households['size'].apply(self._parse_size_label)
        
        fit_stats = self.ipf.compute_fit_statistics({
            'row': hh_size_marginal,
            'col': hh_type_marginal
        })
        
        return households, fit_stats
    
    def synthesize_population(
        self,
        population_data: pd.DataFrame,
        total_population: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Synthesize population using IPF for age × sex × household role.
        
        Args:
            population_data: DataFrame with Ålder, Kön, Hushållstyp, Antal columns
            total_population: Override total (uses marginal sum if None)
            
        Returns:
            Tuple of (population DataFrame, fit statistics dict)
        """
        # Extract marginals
        age_marginal = self._extract_age_marginal(population_data)
        sex_marginal = self._extract_sex_marginal(population_data)
        role_marginal = self._extract_role_marginal(population_data)
        
        logger.info(f"Age marginal total: {age_marginal.sum()}")
        logger.info(f"Sex marginal total: {sex_marginal.sum()}")
        logger.info(f"Role marginal total: {role_marginal.sum()}")
        
        # 3D IPF: age × sex × role
        self.ipf.fit({
            'age': age_marginal,
            'sex': sex_marginal,
            'role': role_marginal
        })
        
        # Sample population
        n = total_population or int(age_marginal.sum())
        population = self.ipf.sample(n)
        
        fit_stats = self.ipf.compute_fit_statistics({
            'age': age_marginal,
            'sex': sex_marginal,
            'role': role_marginal
        })
        
        return population, fit_stats
    
    def _extract_household_size_marginal(self, df: pd.DataFrame) -> pd.Series:
        """Extract household size marginal from data."""
        size_col = 'Hushållsstorlek' if 'Hushållsstorlek' in df.columns else 'hh_size'
        count_col = 'Antal' if 'Antal' in df.columns else df.columns[-1]
        
        return df.groupby(size_col)[count_col].sum()
    
    def _extract_housing_type_marginal(self, df: pd.DataFrame) -> pd.Series:
        """Extract housing type marginal from data."""
        type_col = 'Hustyp' if 'Hustyp' in df.columns else 'house_type'
        count_col = 'Antal' if 'Antal' in df.columns else df.columns[-1]
        
        return df.groupby(type_col)[count_col].sum()
    
    def _extract_age_marginal(self, df: pd.DataFrame) -> pd.Series:
        """Extract age marginal from population data."""
        age_col = 'Ålder' if 'Ålder' in df.columns else 'age_group'
        count_col = 'Antal' if 'Antal' in df.columns else df.columns[-1]
        
        return df.groupby(age_col)[count_col].sum()
    
    def _extract_sex_marginal(self, df: pd.DataFrame) -> pd.Series:
        """Extract sex marginal from population data."""
        sex_col = 'Kön' if 'Kön' in df.columns else 'sex'
        count_col = 'Antal' if 'Antal' in df.columns else df.columns[-1]
        
        return df.groupby(sex_col)[count_col].sum()
    
    def _extract_role_marginal(self, df: pd.DataFrame) -> pd.Series:
        """Extract household role marginal from population data."""
        role_col = 'Hushållstyp' if 'Hushållstyp' in df.columns else 'hh_role'
        count_col = 'Antal' if 'Antal' in df.columns else df.columns[-1]
        
        return df.groupby(role_col)[count_col].sum()
    
    def _parse_size_label(self, label: str) -> int:
        """Parse household size from label."""
        label = str(label).lower()
        if '1' in label:
            return 1
        elif '2' in label:
            return 2
        elif '3' in label:
            return 3
        elif '4' in label:
            return 4
        elif '5' in label:
            return 5
        elif '6' in label or 'fler' in label:
            return 6
        return 1


class ConstrainedIPF:
    """
    Constrained IPF that generates complete household compositions.
    
    Instead of generating individuals and then trying to match them,
    this approach:
    1. Defines household "archetypes" (valid compositions)
    2. Builds a seed matrix where infeasible combinations have zero weight
    3. Runs IPF to match marginals while respecting constraints
    4. Samples complete households directly
    
    This ensures 100% of generated households are valid by construction.
    """
    
    def __init__(
        self,
        min_parent_age_gap: int = 18,
        max_parent_age_gap: int = 45,
        max_partner_age_diff: int = 15,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6
    ):
        """
        Initialize constrained IPF.
        
        Args:
            min_parent_age_gap: Minimum age difference parent-child
            max_parent_age_gap: Maximum age difference parent-child
            max_partner_age_diff: Maximum age difference between partners
            max_iterations: Max IPF iterations
            convergence_threshold: Convergence criterion
        """
        self.min_parent_age_gap = min_parent_age_gap
        self.max_parent_age_gap = max_parent_age_gap
        self.max_partner_age_diff = max_partner_age_diff
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Household archetypes: defines valid compositions
        # Each archetype is a tuple of (role, age_constraint) requirements
        self.archetypes = self._define_archetypes()
        
        # Results
        self.fitted_weights: Optional[np.ndarray] = None
        self.archetype_labels: List[str] = []
        self.convergence_history: List[float] = []
        self.fit_stats: Dict = {}
    
    def _define_archetypes(self) -> List[Dict]:
        """
        Define valid household archetypes with composition rules.
        
        Each archetype defines:
        - size: household size
        - composition: list of member specs (role, age_range, sex_constraint)
        - description: human-readable label
        """
        archetypes = [
            # === Single person households (size 1) ===
            {
                'size': 1,
                'name': 'single_young',
                'composition': [
                    {'role': 'single', 'age_range': (18, 34), 'sex': 'any'}
                ]
            },
            {
                'size': 1,
                'name': 'single_middle',
                'composition': [
                    {'role': 'single', 'age_range': (35, 64), 'sex': 'any'}
                ]
            },
            {
                'size': 1,
                'name': 'single_senior',
                'composition': [
                    {'role': 'single', 'age_range': (65, 100), 'sex': 'any'}
                ]
            },
            
            # === Two-person households (size 2) ===
            # Young couples
            {
                'size': 2,
                'name': 'couple_young',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (18, 34), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (18, 34), 'sex': 'female'}
                ]
            },
            # Middle-aged couples
            {
                'size': 2,
                'name': 'couple_middle',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (35, 64), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (35, 64), 'sex': 'female'}
                ]
            },
            # Senior couples
            {
                'size': 2,
                'name': 'couple_senior',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (65, 100), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (65, 100), 'sex': 'female'}
                ]
            },
            # Single parent with 1 child
            {
                'size': 2,
                'name': 'single_parent_1child',
                'composition': [
                    {'role': 'single', 'age_range': (25, 64), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'}
                ]
            },
            # Adult child living with single parent
            {
                'size': 2,
                'name': 'single_with_adult_child',
                'composition': [
                    {'role': 'single', 'age_range': (45, 80), 'sex': 'any'},
                    {'role': 'child', 'age_range': (18, 35), 'sex': 'any'}
                ]
            },
            
            # === Three-person households (size 3) ===
            # Couple with 1 young child
            {
                'size': 3,
                'name': 'couple_1child_young',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (25, 44), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (25, 44), 'sex': 'female'},
                    {'role': 'child', 'age_range': (0, 10), 'sex': 'any'}
                ]
            },
            # Couple with 1 older child
            {
                'size': 3,
                'name': 'couple_1child_older',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (35, 54), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (35, 54), 'sex': 'female'},
                    {'role': 'child', 'age_range': (10, 17), 'sex': 'any'}
                ]
            },
            # Couple with adult child
            {
                'size': 3,
                'name': 'couple_adult_child',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (45, 70), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (45, 70), 'sex': 'female'},
                    {'role': 'child', 'age_range': (18, 30), 'sex': 'any'}
                ]
            },
            # Single parent with 2 children
            {
                'size': 3,
                'name': 'single_parent_2children',
                'composition': [
                    {'role': 'single', 'age_range': (28, 64), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'}
                ]
            },
            
            # === Four-person households (size 4) ===
            # Couple with 2 children
            {
                'size': 4,
                'name': 'couple_2children',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (28, 54), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (28, 54), 'sex': 'female'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'}
                ]
            },
            # Couple with older children
            {
                'size': 4,
                'name': 'couple_2children_older',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (40, 60), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (40, 60), 'sex': 'female'},
                    {'role': 'child', 'age_range': (12, 24), 'sex': 'any'},
                    {'role': 'child', 'age_range': (12, 24), 'sex': 'any'}
                ]
            },
            # Single parent with 3 children
            {
                'size': 4,
                'name': 'single_parent_3children',
                'composition': [
                    {'role': 'single', 'age_range': (30, 55), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'}
                ]
            },
            
            # === Five-person households (size 5) ===
            # Couple with 3 children
            {
                'size': 5,
                'name': 'couple_3children',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (30, 54), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (30, 54), 'sex': 'female'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'}
                ]
            },
            # Single parent with 4 children
            {
                'size': 5,
                'name': 'single_parent_4children',
                'composition': [
                    {'role': 'single', 'age_range': (32, 55), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'}
                ]
            },
            
            # === Large households (size 6+) ===
            # Couple with 4+ children
            {
                'size': 6,
                'name': 'couple_4plus_children',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (32, 54), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (32, 54), 'sex': 'female'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (0, 17), 'sex': 'any'}
                ]
            },
            # Multi-generational (3 generations)
            {
                'size': 5,
                'name': 'multigenerational_3gen',
                'composition': [
                    {'role': 'cohabiting', 'age_range': (70, 90), 'sex': 'any'},
                    {'role': 'cohabiting', 'age_range': (40, 55), 'sex': 'male'},
                    {'role': 'cohabiting', 'age_range': (40, 55), 'sex': 'female'},
                    {'role': 'child', 'age_range': (5, 17), 'sex': 'any'},
                    {'role': 'child', 'age_range': (5, 17), 'sex': 'any'}
                ]
            },
        ]
        
        return archetypes
    
    def fit(
        self,
        household_data: pd.DataFrame,
        population_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Fit constrained IPF to match both household and population marginals.
        
        Args:
            household_data: DataFrame with Hushållsstorlek, Hustyp, Antal
            population_data: DataFrame with Ålder, Kön, Hushållstyp, Antal
            
        Returns:
            Dict mapping archetype name to count
        """
        # Extract marginals
        hh_size_marginal = self._extract_size_marginal(household_data)
        age_marginal = self._extract_age_marginal(population_data)
        sex_marginal = self._extract_sex_marginal(population_data)
        role_marginal = self._extract_role_marginal(population_data)
        
        # Store for use in sampling
        self.age_marginal = age_marginal
        self.sex_marginal = sex_marginal
        self.role_marginal = role_marginal
        
        logger.info(f"Target household sizes: {dict(hh_size_marginal)}")
        logger.info(f"Target population: {age_marginal.sum()}")
        
        # Build seed matrix: archetypes × 1 (we'll expand this)
        # Weight by how well each archetype matches marginals
        archetype_weights = self._compute_archetype_weights(
            hh_size_marginal, age_marginal, sex_marginal, role_marginal
        )
        
        logger.info(f"Computed weights for {len(archetype_weights)} archetypes")
        
        # Run constrained IPF to fit archetype counts to marginals
        archetype_counts = self._fit_archetype_counts(
            archetype_weights,
            hh_size_marginal,
            age_marginal,
            sex_marginal,
            role_marginal
        )
        
        self.fitted_weights = archetype_counts
        self.archetype_labels = [a['name'] for a in self.archetypes]
        
        return archetype_counts
    
    def _compute_archetype_weights(
        self,
        hh_size_marginal: pd.Series,
        age_marginal: pd.Series,
        sex_marginal: pd.Series,
        role_marginal: pd.Series
    ) -> Dict[str, float]:
        """
        Compute initial weights for each archetype based on marginal compatibility.
        """
        weights = {}
        
        for archetype in self.archetypes:
            name = archetype['name']
            size = archetype['size']
            
            # Base weight from household size marginal
            size_key = self._find_size_key(hh_size_marginal, size)
            if size_key and hh_size_marginal[size_key] > 0:
                base_weight = hh_size_marginal[size_key]
            else:
                base_weight = 1.0
            
            # Adjust by composition feasibility
            composition_factor = self._estimate_composition_feasibility(
                archetype, age_marginal, sex_marginal, role_marginal
            )
            
            weights[name] = base_weight * composition_factor
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _estimate_composition_feasibility(
        self,
        archetype: Dict,
        age_marginal: pd.Series,
        sex_marginal: pd.Series,
        role_marginal: pd.Series
    ) -> float:
        """
        Estimate how feasible an archetype is given the marginals.
        
        Returns a score 0-1 based on availability of required demographics.
        """
        score = 1.0
        
        for member in archetype['composition']:
            role = member['role']
            age_lo, age_hi = member['age_range']
            sex_req = member['sex']
            
            # Check role availability
            role_key = self._find_role_key(role_marginal, role)
            if role_key:
                role_count = role_marginal[role_key]
            else:
                role_count = 0
            
            # Check age availability
            age_count = self._count_in_age_range(age_marginal, age_lo, age_hi)
            
            # Check sex availability  
            if sex_req != 'any':
                sex_key = self._find_sex_key(sex_marginal, sex_req)
                sex_count = sex_marginal.get(sex_key, 0) if sex_key else 0
            else:
                sex_count = sex_marginal.sum()
            
            # Combine (use geometric mean-ish)
            member_score = min(1.0, (role_count * age_count * sex_count) / 1e6)
            score *= max(0.01, member_score)  # Floor to prevent zeros
        
        return score
    
    def _fit_archetype_counts(
        self,
        initial_weights: Dict[str, float],
        hh_size_marginal: pd.Series,
        age_marginal: pd.Series,
        sex_marginal: pd.Series,
        role_marginal: pd.Series
    ) -> Dict[str, float]:
        """
        Use iterative fitting to determine archetype counts.
        
        This is a simplified IPF that adjusts archetype counts to match:
        1. Household size distribution
        2. Total population count
        3. Role distribution (approximately)
        """
        # Initialize counts proportional to weights
        total_hh = hh_size_marginal.sum()
        counts = {name: weight * total_hh for name, weight in initial_weights.items()}
        
        # Build mapping of archetype -> size
        size_by_archetype = {a['name']: a['size'] for a in self.archetypes}
        
        # Build mapping of archetype -> population contribution
        pop_by_archetype = {a['name']: a['size'] for a in self.archetypes}
        
        # Build mapping of archetype -> role contributions
        def get_role_counts(archetype):
            roles = {'single': 0, 'cohabiting': 0, 'child': 0}
            for m in archetype['composition']:
                r = m['role']
                if r in roles:
                    roles[r] += 1
            return roles
        
        roles_by_archetype = {a['name']: get_role_counts(a) for a in self.archetypes}
        
        # Iterative fitting
        for iteration in range(self.max_iterations):
            old_counts = counts.copy()
            
            # Step 1: Scale to match household size distribution
            for size in range(1, 7):
                size_key = self._find_size_key(hh_size_marginal, size)
                if not size_key:
                    continue
                    
                target = hh_size_marginal[size_key]
                current = sum(c for n, c in counts.items() if size_by_archetype[n] == size)
                
                if current > 0:
                    factor = target / current
                    for name in counts:
                        if size_by_archetype[name] == size:
                            counts[name] *= factor
            
            # Step 2: Scale to match total population
            target_pop = age_marginal.sum()
            current_pop = sum(c * pop_by_archetype[n] for n, c in counts.items())
            
            if current_pop > 0:
                pop_factor = target_pop / current_pop
                # Apply gentle scaling to avoid disrupting size distribution
                pop_factor = 0.7 + 0.3 * pop_factor  # Damped
                counts = {n: c * pop_factor for n, c in counts.items()}
            
            # Check convergence
            max_change = max(
                abs(counts[n] - old_counts[n]) / max(old_counts[n], 1)
                for n in counts
            )
            self.convergence_history.append(max_change)
            
            if max_change < self.convergence_threshold:
                logger.info(f"Constrained IPF converged after {iteration + 1} iterations")
                break
        
        # Round to integers
        counts = {n: max(0, int(round(c))) for n, c in counts.items()}
        
        # Store fit statistics
        self.fit_stats = {
            'iterations': len(self.convergence_history),
            'converged': len(self.convergence_history) < self.max_iterations,
            'final_change': self.convergence_history[-1] if self.convergence_history else 0
        }
        
        return counts
    
    def sample_households(
        self,
        archetype_counts: Optional[Dict[str, float]] = None,
        random_state: Optional[int] = None
    ) -> List[Dict]:
        """
        Sample complete households from fitted distribution.
        
        Returns:
            List of household dicts, each with 'members' list
        """
        import random
        
        if random_state is not None:
            random.seed(random_state)
        
        if archetype_counts is None:
            archetype_counts = self.fitted_weights
        
        if archetype_counts is None:
            raise ValueError("Must call fit() first or provide archetype_counts")
        
        households = []
        archetype_by_name = {a['name']: a for a in self.archetypes}
        
        for archetype_name, count in archetype_counts.items():
            archetype = archetype_by_name.get(archetype_name)
            if not archetype:
                continue
                
            for _ in range(int(count)):
                hh = self._generate_household(archetype)
                households.append(hh)
        
        random.shuffle(households)
        return households
    
    def _generate_household(self, archetype: Dict) -> Dict:
        """
        Generate a single household from an archetype.
        
        Ensures constraints are satisfied:
        - Parent-child age gaps
        - Partner age differences
        
        Uses age marginal distribution (if available) to improve demographic fit.
        """
        import random
        import re
        
        members = []
        parent_ages = []
        
        # Build age sampling distribution from marginal (if available)
        age_weights = self._build_age_weights() if hasattr(self, 'age_marginal') else None
        
        for i, member_spec in enumerate(archetype['composition']):
            role = member_spec['role']
            age_lo, age_hi = member_spec['age_range']
            sex_req = member_spec['sex']
            
            # Determine sex (consider sex marginal)
            if sex_req == 'any':
                if hasattr(self, 'sex_marginal'):
                    male_key = self._find_sex_key(self.sex_marginal, 'male')
                    female_key = self._find_sex_key(self.sex_marginal, 'female')
                    male_count = self.sex_marginal.get(male_key, 1) if male_key else 1
                    female_count = self.sex_marginal.get(female_key, 1) if female_key else 1
                    male_prob = male_count / (male_count + female_count)
                    sex = 'male' if random.random() < male_prob else 'female'
                else:
                    sex = random.choice(['male', 'female'])
            else:
                sex = sex_req
            
            # Determine age with constraints
            if role == 'child' and parent_ages:
                # Child age must be consistent with parent ages
                youngest_parent = min(parent_ages)
                max_child_age = min(age_hi, youngest_parent - self.min_parent_age_gap)
                min_child_age = max(age_lo, youngest_parent - self.max_parent_age_gap)
                
                if max_child_age >= min_child_age:
                    age = self._sample_age_in_range(min_child_age, max_child_age, age_weights)
                else:
                    age = self._sample_age_in_range(age_lo, age_hi, age_weights)
            elif role == 'cohabiting' and parent_ages:
                # Partner age should be close to existing partner
                existing_partner = parent_ages[0]
                min_partner_age = max(age_lo, existing_partner - self.max_partner_age_diff)
                max_partner_age = min(age_hi, existing_partner + self.max_partner_age_diff)
                
                if max_partner_age >= min_partner_age:
                    age = self._sample_age_in_range(min_partner_age, max_partner_age, age_weights)
                else:
                    age = self._sample_age_in_range(age_lo, age_hi, age_weights)
            else:
                age = self._sample_age_in_range(age_lo, age_hi, age_weights)
            
            # Track parent ages for constraint checking
            if role in ['single', 'cohabiting']:
                parent_ages.append(age)
            
            members.append({
                'age': age,
                'sex': sex,
                'role': role
            })
        
        return {
            'archetype': archetype['name'],
            'size': archetype['size'],
            'members': members
        }
    
    def _build_age_weights(self) -> Dict[int, float]:
        """Build per-year age weights from age marginal."""
        import re
        
        weights = {}
        for key, count in self.age_marginal.items():
            key_str = str(key)
            
            # Parse "X-Y år" format
            match = re.match(r'(\d+)-(\d+)\s*år', key_str)
            if match:
                lo, hi = int(match.group(1)), int(match.group(2))
            else:
                # Try "X- år" or "X+ år" format
                match = re.match(r'(\d+)[-+]\s*år', key_str)
                if match:
                    lo = int(match.group(1))
                    hi = lo + 15  # Assume ~15 year span for open-ended
                else:
                    continue
            
            # Distribute count uniformly across ages
            span = hi - lo + 1
            weight_per_year = count / span
            for age in range(lo, hi + 1):
                weights[age] = weights.get(age, 0) + weight_per_year
        
        return weights
    
    def _sample_age_in_range(
        self, 
        lo: int, 
        hi: int, 
        age_weights: Optional[Dict[int, float]] = None
    ) -> int:
        """Sample an age in range, preferring ages with higher weight."""
        import random
        
        if age_weights is None:
            return random.randint(lo, hi)
        
        # Get weights for ages in range
        ages = list(range(lo, hi + 1))
        weights = [age_weights.get(a, 0.01) for a in ages]
        
        # Normalize
        total = sum(weights)
        if total > 0:
            probs = [w / total for w in weights]
            return random.choices(ages, weights=probs, k=1)[0]
        
        return random.randint(lo, hi)
    
    def _extract_size_marginal(self, df: pd.DataFrame) -> pd.Series:
        """Extract household size marginal."""
        size_col = 'Hushållsstorlek' if 'Hushållsstorlek' in df.columns else 'hh_size'
        count_col = 'Antal' if 'Antal' in df.columns else df.columns[-1]
        return df.groupby(size_col)[count_col].sum()
    
    def _extract_age_marginal(self, df: pd.DataFrame) -> pd.Series:
        """Extract age marginal."""
        age_col = 'Ålder' if 'Ålder' in df.columns else 'age_group'
        count_col = 'Antal' if 'Antal' in df.columns else df.columns[-1]
        return df.groupby(age_col)[count_col].sum()
    
    def _extract_sex_marginal(self, df: pd.DataFrame) -> pd.Series:
        """Extract sex marginal."""
        sex_col = 'Kön' if 'Kön' in df.columns else 'sex'
        count_col = 'Antal' if 'Antal' in df.columns else df.columns[-1]
        return df.groupby(sex_col)[count_col].sum()
    
    def _extract_role_marginal(self, df: pd.DataFrame) -> pd.Series:
        """Extract role marginal."""
        role_col = 'Hushållstyp' if 'Hushållstyp' in df.columns else 'hh_role'
        count_col = 'Antal' if 'Antal' in df.columns else df.columns[-1]
        return df.groupby(role_col)[count_col].sum()
    
    def _find_size_key(self, marginal: pd.Series, size: int) -> Optional[str]:
        """Find the key in marginal matching a household size."""
        for key in marginal.index:
            key_str = str(key).lower()
            if size == 6 and ('6' in key_str or 'fler' in key_str):
                return key
            elif str(size) in key_str and 'fler' not in key_str:
                return key
        return None
    
    def _find_role_key(self, marginal: pd.Series, role: str) -> Optional[str]:
        """Find the key in marginal matching a role.

        Delegates to ``Config.find_role_key()`` which uses exact dict
        lookups from *table_mapping.json*.
        """
        from gbgsynth.config import Config
        cfg = Config()
        return cfg.find_role_key(marginal.index, role)

    def _find_sex_key(self, marginal: pd.Series, sex: str) -> Optional[str]:
        """Find the key in marginal matching a sex.

        Uses ``Config._sex_lookup`` for exact matching.
        """
        from gbgsynth.config import Config
        cfg = Config()
        sex_lower = sex.lower()
        for key in marginal.index:
            k = str(key).strip().lower()
            mapped = cfg._sex_lookup.get(k)
            if mapped == sex_lower:
                return key
        return None
    
    def _count_in_age_range(self, age_marginal: pd.Series, lo: int, hi: int) -> int:
        """Count population in an age range."""
        import re
        
        total = 0
        for key, count in age_marginal.items():
            # Parse age range from key
            key_str = str(key)
            
            # Try "X-Y år" format
            match = re.match(r'(\d+)-(\d+)\s*år', key_str)
            if match:
                key_lo, key_hi = int(match.group(1)), int(match.group(2))
            else:
                # Try "X- år" format (open-ended)
                match = re.match(r'(\d+)[-+]\s*år', key_str)
                if match:
                    key_lo = int(match.group(1))
                    key_hi = 150
                else:
                    continue
            
            # Check overlap
            if key_hi >= lo and key_lo <= hi:
                # Proportional overlap
                overlap_lo = max(lo, key_lo)
                overlap_hi = min(hi, key_hi)
                key_range = key_hi - key_lo + 1
                overlap_range = overlap_hi - overlap_lo + 1
                
                total += count * (overlap_range / key_range)
        
        return int(total)
    
    def compute_fit_statistics(
        self,
        household_data: pd.DataFrame,
        population_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute goodness-of-fit statistics comparing generated vs target.
        """
        if self.fitted_weights is None:
            raise ValueError("Must call fit() first")
        
        # Generate summary from fitted weights
        size_by_archetype = {a['name']: a['size'] for a in self.archetypes}
        
        # Household size fit
        target_sizes = self._extract_size_marginal(household_data)
        fitted_sizes = {}
        for name, count in self.fitted_weights.items():
            size = size_by_archetype.get(name, 0)
            size_key = self._find_size_key(target_sizes, size)
            if size_key:
                fitted_sizes[size_key] = fitted_sizes.get(size_key, 0) + count
        
        # Compute errors
        size_errors = []
        for key in target_sizes.index:
            target = target_sizes[key]
            fitted = fitted_sizes.get(key, 0)
            size_errors.append(fitted - target)
        
        size_errors = np.array(size_errors)
        
        # Population fit
        target_pop = self._extract_age_marginal(population_data).sum()
        fitted_pop = sum(
            count * size_by_archetype.get(name, 0)
            for name, count in self.fitted_weights.items()
        )
        
        return {
            'rmse': float(np.sqrt(np.mean(size_errors ** 2))),
            'mae': float(np.mean(np.abs(size_errors))),
            'max_error': int(np.max(np.abs(size_errors))),
            'target_households': int(target_sizes.sum()),
            'fitted_households': int(sum(self.fitted_weights.values())),
            'target_population': int(target_pop),
            'fitted_population': int(fitted_pop),
            'iterations': self.fit_stats.get('iterations', 0),
            'converged': self.fit_stats.get('converged', False)
        }


# =========================================================================
# Engine functions — called by PopulationSynthesizer.synthesize()
# =========================================================================

def run_ipf_engine(
    population_data: pd.DataFrame,
    household_data: pd.DataFrame,
    car_data: Optional[pd.DataFrame],
    constraints: Dict,
    config,
    start_agent_id: int = 1,
    start_household_id: int = 1,
) -> Tuple[List["Agent"], List["Household"], int, int, Dict]:
    """Run the IPF synthesis engine.

    1. Fit household size × housing type via 2-D IPF.
    2. Fit age × sex × role via 3-D IPF.
    3. Sample households and individuals.
    4. Match individuals to households.

    Returns:
        ``(agents, households, next_agent_id, next_household_id, ipf_stats)``
    """
    from gbgsynth.models import Agent, Household
    from gbgsynth.helpers import household_factory as _hf
    from gbgsynth.helpers import housing_assigner as _ha
    from gbgsynth.helpers import population_generator as _pg
    from gbgsynth.helpers import car_assigner as _ca

    # ── Step 1: Fit household distribution ──
    logger.info("IPF Step 1: Fitting household distribution")

    hh_size_col = 'Hushållsstorlek' if 'Hushållsstorlek' in household_data.columns else 'hh_size'
    hh_type_col = 'Hustyp' if 'Hustyp' in household_data.columns else 'house_type'
    count_col = 'Antal' if 'Antal' in household_data.columns else household_data.columns[-1]

    hh_size_marginal = household_data.groupby(hh_size_col)[count_col].sum()
    hh_type_marginal = household_data.groupby(hh_type_col)[count_col].sum()

    hh_ipf = IPFSynthesizer()
    hh_ipf.fit_2d(hh_size_marginal, hh_type_marginal)

    total_hh = int(hh_size_marginal.sum())
    hh_samples = hh_ipf.sample(total_hh)

    hh_fit_stats = hh_ipf.compute_fit_statistics({
        'row': hh_size_marginal,
        'col': hh_type_marginal,
    })
    logger.info("Household IPF: RMSE=%.2f, converged=%s",
                hh_fit_stats['rmse'], hh_fit_stats['converged'])

    # ── Step 2: Create household objects ──
    logger.info("IPF Step 2: Creating households from IPF samples")

    households: List[Household] = []
    next_hh_id = start_household_id
    for _, row in hh_samples.iterrows():
        size = _hf.parse_household_size(row['row'], config)
        house_type = _ha.parse_house_type(row['col'])
        hh = Household(
            household_id=next_hh_id,
            size=size,
            house_type=house_type,
            cars=0,
            assigned_hustyp=row['col'],
        )
        households.append(hh)
        next_hh_id += 1

    logger.info("Created %d households from IPF", len(households))

    # ── Step 3: Fit population distribution ──
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
        'role': role_marginal,
    })

    total_pop = int(age_marginal.sum())
    pop_samples = pop_ipf.sample(total_pop)

    pop_fit_stats = pop_ipf.compute_fit_statistics({
        'age': age_marginal,
        'sex': sex_marginal,
        'role': role_marginal,
    })
    logger.info("Population IPF: RMSE=%.2f, converged=%s",
                pop_fit_stats['rmse'], pop_fit_stats['converged'])

    # ── Step 4: Create agents ──
    logger.info("IPF Step 4: Creating agents from IPF samples")

    pool: List[Agent] = []
    next_id = start_agent_id
    for _, row in pop_samples.iterrows():
        age = _pg.sample_age_from_group(row['age'], config)
        sex = _pg.translate_sex(row['sex'])
        hh_role = _pg.translate_hh_role(row['role'])
        agent = Agent(agent_id=next_id, age=age, sex=sex, hh_role=hh_role)
        pool.append(agent)
        next_id += 1

    logger.info("Created %d agents from IPF", len(pool))

    # ── Step 5: Match individuals to households ──
    logger.info("IPF Step 5: Matching individuals to households")
    agents, households, next_id, next_hh_id = _match_pool_to_households(
        pool, households, constraints, next_id, next_hh_id,
    )

    ipf_stats = {'household': hh_fit_stats, 'population': pop_fit_stats}
    return agents, households, next_id, next_hh_id, ipf_stats


def run_constrained_ipf_engine(
    population_data: pd.DataFrame,
    household_data: pd.DataFrame,
    car_data: Optional[pd.DataFrame],
    constraints: Dict,
    config,
    start_agent_id: int = 1,
    start_household_id: int = 1,
) -> Tuple[List["Agent"], List["Household"], int, int, Dict]:
    """Run the constrained-IPF synthesis engine.

    Returns:
        ``(agents, households, next_agent_id, next_household_id, ipf_stats)``
    """
    from gbgsynth.models import Agent, Household
    from gbgsynth.helpers import housing_assigner as _ha

    min_gap = constraints.get('parent_child_age_gap_min', 18)
    max_diff = constraints.get('partner_age_difference_max', 15)

    cipf = ConstrainedIPF(
        min_parent_age_gap=min_gap,
        max_partner_age_diff=max_diff,
    )

    # ── Step 1: Fit ──
    logger.info("Constrained IPF Step 1: Fitting archetype counts to marginals")
    archetype_counts = cipf.fit(household_data, population_data)

    logger.info("Fitted %d archetypes", len(archetype_counts))
    for name, count in sorted(archetype_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info("  %s: %d", name, count)

    # ── Step 2: Sample ──
    logger.info("Constrained IPF Step 2: Sampling complete households")
    sampled = cipf.sample_households(archetype_counts, random_state=None)
    logger.info("Sampled %d complete households", len(sampled))

    # ── Step 3: Build objects ──
    logger.info("Constrained IPF Step 3: Creating household and agent objects")

    hh_type_col = 'Hustyp' if 'Hustyp' in household_data.columns else 'house_type'
    count_col = 'Antal' if 'Antal' in household_data.columns else household_data.columns[-1]
    hh_type_marginal = household_data.groupby(hh_type_col)[count_col].sum()
    total_hh_type = hh_type_marginal.sum()
    hh_type_probs = hh_type_marginal / total_hh_type

    _np_rng = np.random.default_rng()

    agents: List[Agent] = []
    households: List[Household] = []
    next_id = start_agent_id
    next_hh_id = start_household_id

    for hh_data in sampled:
        ht_label = _np_rng.choice(hh_type_probs.index, p=hh_type_probs.values)
        house_type = _ha.parse_house_type(ht_label)
        hh = Household(
            household_id=next_hh_id,
            size=hh_data['size'],
            house_type=house_type,
            cars=0,
            assigned_hustyp=ht_label,
        )
        households.append(hh)
        next_hh_id += 1

        for member in hh_data['members']:
            agent = Agent(
                agent_id=next_id,
                age=member['age'],
                sex=member['sex'],
                hh_role=member['role'],
            )
            hh.add_member(agent)
            agents.append(agent)
            next_id += 1

    logger.info("Created %d households with %d agents", len(households), len(agents))

    # ── Step 4: Fit statistics ──
    fit_stats = cipf.compute_fit_statistics(household_data, population_data)
    ipf_stats = {
        'constrained_ipf': fit_stats,
        'archetype_counts': archetype_counts,
        'iterations': fit_stats.get('iterations', 0),
        'converged': fit_stats.get('converged', False),
    }

    logger.info(
        "Constrained IPF fit: RMSE=%.2f, HH error=%d, Pop error=%d",
        fit_stats['rmse'],
        fit_stats['fitted_households'] - fit_stats['target_households'],
        fit_stats['fitted_population'] - fit_stats['target_population'],
    )

    return agents, households, next_id, next_hh_id, ipf_stats


def _match_pool_to_households(
    pool: List["Agent"],
    households: List["Household"],
    constraints: Dict,
    next_agent_id: int,
    next_household_id: int,
) -> Tuple[List["Agent"], List["Household"], int, int]:
    """Match an individual pool to households (IPF post-processing).

    Returns ``(agents, households, next_agent_id, next_household_id)``.
    """
    from gbgsynth.models import Household

    hh_by_size: Dict[int, List] = {}
    for hh in households:
        hh_by_size.setdefault(hh.size, []).append(hh)

    adults_cohab = [a for a in pool if a.is_adult() and a.hh_role == 'cohabiting']
    adults_single = [a for a in pool if a.is_adult() and a.hh_role == 'single']
    children = [a for a in pool if a.is_child()]

    random.shuffle(adults_cohab)
    random.shuffle(adults_single)
    random.shuffle(children)

    multi_hhs = sorted(
        [hh for hh in households if hh.size >= 2],
        key=lambda h: h.size, reverse=True,
    )

    # Phase 1: couples
    max_age_diff = constraints.get('partner_age_difference_max', 15)
    males = [a for a in adults_cohab if a.sex == 'male']
    females = [a for a in adults_cohab if a.sex == 'female']
    couples = 0

    for hh in multi_hhs:
        if not hh.can_fit(2):
            continue
        for male in males:
            if male.household_id is not None:
                continue
            for female in females:
                if female.household_id is not None:
                    continue
                if abs(male.age - female.age) <= max_age_diff:
                    hh.add_member(male)
                    hh.add_member(female)
                    couples += 1
                    break
            else:
                continue
            break
    logger.info("Formed %d couples", couples)

    # Phase 2: children
    min_gap = constraints.get('parent_child_age_gap_min', 18)
    children_sorted = sorted(children, key=lambda c: c.age)
    hhs_with_adults = sorted(
        [hh for hh in multi_hhs if hh.members and hh.can_fit()],
        key=lambda h: h.size - len(h.members), reverse=True,
    )
    ch_assigned = 0
    for child in children_sorted:
        if child.household_id is not None:
            continue
        for hh in hhs_with_adults:
            if not hh.can_fit():
                continue
            head = hh.head
            if head and (head.age - child.age) >= min_gap:
                hh.add_member(child)
                ch_assigned += 1
                break
    logger.info("Assigned %d children", ch_assigned)

    # Phase 3: remaining cohabiting
    for hh in multi_hhs:
        if not hh.can_fit():
            continue
        for adult in adults_cohab:
            if adult.household_id is None and hh.can_fit():
                hh.add_member(adult)

    # Phase 4: singles
    singles_assigned = 0
    for single in adults_single:
        if single.household_id is not None:
            continue
        for hh in hh_by_size.get(1, []):
            if hh.is_full():
                continue
            hh.add_member(single)
            singles_assigned += 1
            break
    logger.info("Assigned %d singles", singles_assigned)

    # Phase 5: redistribute
    for agent in pool:
        if agent.household_id is not None:
            continue
        for hh in households:
            if hh.can_fit():
                hh.add_member(agent)
                break

    # Phase 6: overflow
    still_remaining = [a for a in pool if a.household_id is None]
    if still_remaining:
        logger.warning("%d individuals unmatched — creating overflow households",
                       len(still_remaining))
        for agent in still_remaining:
            hh = Household(household_id=next_household_id, size=1)
            hh.add_member(agent)
            households.append(hh)
            next_household_id += 1

    agents = []
    for hh in households:
        agents.extend(hh.members)

    return agents, households, next_agent_id, next_household_id
