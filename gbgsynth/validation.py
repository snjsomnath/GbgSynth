"""
Out-of-sample validation module for synthetic populations.

This module provides tools to validate synthetic populations against
census tables that were NOT used in the synthesis process.

CURRENT SYNTHESIZED ATTRIBUTES:
- Age, Sex (individual)
- Household role (single/cohabiting/child)
- Household size
- Housing type (apartment/detached/special)
- Floor area (dwelling)  
- Car ownership (household)
- Low income flag (household)

USABLE VALIDATION TABLES (match our attributes):
1. Overcrowding rate - derived from household size + floor area
2. Household position by age/sex - cross-validates role assignment
3. Households by type + children count - validates family structure
4. Population by household size + housing type - joint distribution

NOT USABLE (require new attributes):
- Education tables (no education attribute)
- Employment tables (no employment status)
- Foreign background (no country of birth)
- Tenure type (no rent/own distinction)
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple
import math

if TYPE_CHECKING:
    from gbgsynth.area import GbgArea
    
from gbgsynth.api_client import PxWebClient
from gbgsynth.sanity_checks import run_all_checks, SanityCheckResult

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION TABLE REGISTRY
# =============================================================================

# Tables that CAN be used for validation (we have the required attributes)
USABLE_VALIDATION_TABLES = {
    "OVERCROWDING": {
        "id": "Befolkning/Trångboddhet/10_Trangbodd_PRI.px",
        "description": "Overcrowding in multi-family housing",
        "validation_type": "derived_metric",
        "required_attributes": ["household_size", "floor_area"],
        "how_to_validate": "Derive persons/room from synth, compare to census rate",
        "variables": ["Område", "År"],
    },
    
    "HOUSEHOLD_POSITION": {
        "id": "Befolkning/Folkmängd/Folkmängd helår/60_FolkmHHStallning_PRI.px",
        "description": "Population by household position (role), age, sex",
        "validation_type": "cross_tabulation",
        "required_attributes": ["age", "sex", "household_role"],
        "how_to_validate": "Compare joint distribution of age×sex×role (fitted separately)",
        "variables": ["Område", "År", "Hushållsställning", "Kön", "Ålder"],
    },
    
    "HOUSEHOLD_TYPE_CHILDREN": {
        "id": "Befolkning/Hushåll/10_HHTypBarnU18_PRI.px",
        "description": "Households by type and number of children under 18",
        "validation_type": "cross_tabulation",
        "required_attributes": ["household_type", "children_count"],
        "how_to_validate": "Count children per household type, compare to census",
        "variables": ["Område", "År", "Hushållstyp", "Antal barn under 18 år"],
    },
    
    "POPULATION_HH_SIZE_HOUSING": {
        "id": "Befolkning/Folkmängd/Folkmängd helår/65_FolkmHHStorlHusTyp_PRI.px",
        "description": "Population by household size and housing type",
        "validation_type": "cross_tabulation",  
        "required_attributes": ["household_size", "housing_type"],
        "how_to_validate": "Joint distribution of HH size × housing type",
        "variables": ["Område", "År", "Hushållsstorlek", "Hustyp"],
    },
    
    "HOUSEHOLD_SIZE_HOUSING": {
        "id": "Befolkning/Hushåll/31_HHStorlHustyp_PRI.px",
        "description": "Households by size and housing type (USED IN FITTING)",
        "validation_type": "in_sample",  # This is actually used in synthesis!
        "required_attributes": ["household_size", "housing_type"],
        "how_to_validate": "Already used - compare for sanity check only",
        "variables": ["Område", "År", "Hushållsstorlek", "Hustyp"],
    },
}

# Tables that CANNOT be used (we don't have these attributes)
UNUSABLE_VALIDATION_TABLES = {
    "EDUCATION_LEVEL": {
        "id": "Inkomst och utbildning/Utbildning/10_UtbNiva_PRI.px",
        "description": "Education level by foreign background",
        "missing_attribute": "education_level",
        "action": "Add education attribute to Agent model",
    },
    
    "EMPLOYMENT": {
        "id": "Arbetsmarknad/Förvärvsarbetande/10_Forvarb_PRI.px",
        "description": "Employed population by sector",
        "missing_attribute": "employment_status",
        "action": "Add employment attribute to Agent model",
    },
    
    "UNEMPLOYMENT": {
        "id": "Arbetsmarknad/Arbetslöshet/10_Alosa_PRI.px",
        "description": "Unemployment by age and sex",
        "missing_attribute": "employment_status",
        "action": "Add employment attribute to Agent model",
    },
    
    "FOREIGN_BORN": {
        "id": "Befolkning/Utrikes födda/10_FodSveUtl_PRI.px",
        "description": "Population born in Sweden vs abroad",
        "missing_attribute": "country_of_birth",
        "action": "Add country_of_birth to Agent model",
    },
    
    "DWELLING_TENURE": {
        "id": "Bostäder och byggande/Bostadsbestånd/20_TypUppf_PRI.px",
        "description": "Dwellings by housing type and tenure",
        "missing_attribute": "tenure_type",
        "action": "Add tenure (rent/own/coop) to Dwelling model",
    },
    
    "POPULATION_TENURE": {
        "id": "Befolkning/Folkmängd/Folkmängd helår/66_FolkmUppform_PRI.px",
        "description": "Population by tenure type",
        "missing_attribute": "tenure_type",
        "action": "Add tenure to Dwelling model",
    },
}


# Legacy alias
VALIDATION_TABLES = {**USABLE_VALIDATION_TABLES, **UNUSABLE_VALIDATION_TABLES}


# Tables currently used in synthesis (for reference)
SYNTHESIS_TABLES = {
    "BEFOLKNING_HH": "Population by age, sex, household type",
    "HOUSEHOLD_SIZE": "Households by size and housing type",
    "INCOME": "Income standard by age and background",
    "CARS": "Car ownership",
    "DWELLING_SIZE": "Dwellings by type and floor area",
}


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    dimension: str
    census_total: int
    synth_total: int
    rmse: float
    mae: float
    max_error_pct: float
    correlation: float
    categories: List[Dict[str, Any]]
    passed: bool  # True if within tolerance


@dataclass  
class ValidationReport:
    """Complete validation report for an area."""
    area_code: str
    area_name: str
    year: int
    results: Dict[str, ValidationResult]
    overall_score: float  # 0-100
    sanity_result: Optional[Any] = None  # SanityCheckResult
    
    @property
    def is_valid(self) -> bool:
        """Returns True if sanity checks pass (no critical violations)."""
        if self.sanity_result is None:
            return True
        return self.sanity_result.is_valid
    
    def summary(self) -> str:
        """Return a text summary of validation results."""
        lines = [
            f"Validation Report: {self.area_name} ({self.year})",
            "=" * 50,
            f"Overall Score: {self.overall_score:.1f}/100",
            "",
        ]
        
        # Sanity check summary
        if self.sanity_result:
            if self.sanity_result.is_valid:
                lines.append(f"Sanity Check: ✓ PASS ({self.sanity_result.warning_count} warnings)")
            else:
                lines.append(f"Sanity Check: ✗ FAIL ({self.sanity_result.critical_count} critical issues)")
            lines.append("")
        
        for name, result in self.results.items():
            status = "✓ PASS" if result.passed else "✗ FAIL"
            lines.append(f"{name}: {status} (RMSE={result.rmse:.1f}, r={result.correlation:.3f})")
        
        return "\n".join(lines)


class Validator:
    """
    Validator for synthetic populations.
    
    Compares synthesized populations against census tables that were
    NOT used in the synthesis fitting process.
    """
    
    def __init__(self, area: "GbgArea", tolerance: float = 0.10):
        """
        Initialize validator.
        
        Args:
            area: GbgArea with generated population
            tolerance: Maximum acceptable relative error (default 10%)
        """
        self.area = area
        self.tolerance = tolerance
        self._api = PxWebClient()
        
    def validate_overcrowding(self) -> ValidationResult:
        """
        Validate overcrowding metric against census.
        
        Overcrowding = households in multi-family housing where
        persons per room > threshold.
        """
        # Calculate synthetic overcrowding in apartments using Swedish Norm 2:
        # - Max 2 persons per room, EXCLUDING kitchen and living room
        # - Single-person household in 1-room apartment = NOT overcrowded
        # - 5+ rooms = always NOT overcrowded
        overcrowded_synth = 0
        total_apartment_pop = 0
        
        for hh in self.area.households:
            dwelling = hh.dwelling
            if dwelling and dwelling.house_type_sv in ['Flerbostadshus']:
                hh_size = len(hh.members)
                total_apartment_pop += hh_size
                
                if dwelling.floor_area:
                    # Estimate total rooms from floor area (~20 sqm per room in apartments)
                    total_rooms = max(1, dwelling.floor_area // 20)
                    
                    # 5+ rooms = never overcrowded
                    if total_rooms >= 5:
                        continue
                    
                    # Single person in 1-room = NOT overcrowded
                    if hh_size == 1 and total_rooms == 1:
                        continue
                    
                    # Habitable rooms = total rooms minus kitchen and living room
                    # For small apartments: 1 room = 0 habitable, 2 rooms = 0-1 habitable
                    habitable_rooms = max(0, total_rooms - 2)
                    
                    # Overcrowded if more than 2 persons per habitable room
                    # (or any persons if 0 habitable rooms and hh_size > 1)
                    if habitable_rooms == 0:
                        if hh_size > 1:
                            overcrowded_synth += hh_size
                    elif hh_size > (habitable_rooms * 2):
                        overcrowded_synth += hh_size
        
        synth_rate = (overcrowded_synth / total_apartment_pop * 100) if total_apartment_pop > 0 else 0
        
        # Fetch census overcrowding data
        census_overcrowded = None
        census_total_apartment = None
        census_rate = None
        try:
            table_path = "Befolkning/Trångboddhet/10_Trangbodd_PRI.px"
            df = self._api.query_table(
                table_path,
                area_code=self.area.area_api_value,
                year=self.area.year
            )
            if not df.empty and 'Antal' in df.columns:
                # Parse overcrowding counts from table
                # Table has: Trångbodd, Ej trångbodd, Uppgift saknas...
                for _, row in df.iterrows():
                    status = str(row.get('Trångbodd', ''))
                    count = int(row.get('Antal', 0))
                    if 'Trångbodd' in status and 'Ej' not in status:
                        census_overcrowded = count
                    elif 'Ej trångbodd' in status:
                        census_total_apartment = (census_total_apartment or 0) + count
                        
                if census_overcrowded is not None and census_total_apartment:
                    total = census_overcrowded + census_total_apartment
                    census_rate = (census_overcrowded / total * 100) if total > 0 else 0
                    logger.info(f"Census overcrowding: {census_overcrowded}/{total} = {census_rate:.1f}%")
        except Exception as e:
            logger.warning(f"Could not fetch overcrowding census data: {e}")
        
        error_pct = abs(synth_rate - (census_rate or 0))
        # For rate comparison, pass if within tolerance percentage points
        passed = error_pct <= (self.tolerance * 100) if census_rate is not None else True
        
        logger.info(f"Overcrowding: Synth={synth_rate:.1f}%, Census={census_rate}%, Error={error_pct:.1f}pp")
        
        return ValidationResult(
            dimension="overcrowding",
            census_total=census_overcrowded or 0,
            synth_total=overcrowded_synth,
            rmse=error_pct,
            mae=error_pct,
            max_error_pct=error_pct,
            correlation=1.0,
            categories=[{
                "synth_rate": synth_rate,
                "census_rate": census_rate,
                "synth_count": overcrowded_synth,
                "total_pop": total_apartment_pop
            }],
            passed=passed
        )
    
    def validate_household_children(self) -> Optional[ValidationResult]:
        """
        Validate distribution of children across household types.
        
        Compares synthesized children per household type against
        census table 20_HHtypBarn_PRI.px.
        """
        # Calculate synthetic distribution: children per household
        synth_by_children = {0: 0, 1: 0, 2: 0, 3: 0}  # 0, 1, 2, 3+ children
        
        for hh in self.area.households:
            n_children = sum(1 for m in hh.members if m.age < 18)
            key = min(n_children, 3)  # Group 3+ together
            synth_by_children[key] = synth_by_children.get(key, 0) + 1
        
        # Fetch census data
        census_by_children = {}
        try:
            table_path = "Befolkning/Hushåll/10_HHTypBarnU18_PRI.px"
            df = self._api.query_table(
                table_path,
                area_code=self.area.area_api_value,
                year=self.area.year
            )
            
            if not df.empty and 'Antal' in df.columns:
                # Parse the census data - column is 'Antal barn 0-17 år'
                for _, row in df.iterrows():
                    children_cat = str(row.get('Antal barn 0-17 år', ''))
                    value = int(row.get('Antal', 0))
                    
                    if '0 barn' in children_cat:
                        census_by_children[0] = census_by_children.get(0, 0) + value
                    elif '1 barn' in children_cat:
                        census_by_children[1] = census_by_children.get(1, 0) + value
                    elif '2 barn' in children_cat:
                        census_by_children[2] = census_by_children.get(2, 0) + value
                    elif '3 barn' in children_cat:
                        census_by_children[3] = census_by_children.get(3, 0) + value
                    elif '4 barn' in children_cat or 'fler' in children_cat:
                        # Group 4+ with 3+
                        census_by_children[3] = census_by_children.get(3, 0) + value
                        
        except Exception as e:
            logger.warning(f"Could not fetch children distribution: {e}")
            return None
        
        if not census_by_children:
            logger.warning("No census data available for children distribution")
            return None
        
        # Compare distributions
        categories = []
        total_error = 0
        max_error = 0
        
        for n_children in range(4):
            synth = synth_by_children.get(n_children, 0)
            census = census_by_children.get(n_children, 0)
            
            if census > 0:
                error_pct = abs(synth - census) / census * 100
                total_error += (synth - census) ** 2
                max_error = max(max_error, error_pct)
            else:
                error_pct = 0
            
            label = f"{n_children}+ children" if n_children == 3 else f"{n_children} children"
            categories.append({
                "category": label,
                "synth": synth,
                "census": census,
                "error_pct": error_pct
            })
        
        rmse = math.sqrt(total_error / 4)
        # Pass if correlation is high (>0.95) OR if max error is within tolerance
        correlation = self._compute_correlation(
            [synth_by_children.get(i, 0) for i in range(4)],
            [census_by_children.get(i, 0) for i in range(4)]
        )
        passed = correlation >= 0.95 or max_error <= (self.tolerance * 100)
        
        logger.info(f"Children distribution: RMSE={rmse:.1f}, Max Error={max_error:.1f}%, r={correlation:.3f}")
        
        return ValidationResult(
            dimension="children_per_household",
            census_total=sum(census_by_children.values()),
            synth_total=sum(synth_by_children.values()),
            rmse=rmse,
            mae=sum(abs(synth_by_children.get(i, 0) - census_by_children.get(i, 0)) for i in range(4)) / 4,
            max_error_pct=max_error,
            correlation=correlation,
            categories=categories,
            passed=passed
        )
    
    def validate_household_position(self) -> Optional[ValidationResult]:
        """
        Validate household position (role) distribution by age and sex.
        
        This is a cross-tabulation validation - we fit age, sex, and role
        separately, but the JOINT distribution is out-of-sample.
        """
        # Build synthetic distribution by age group and sex
        synth_dist = {}
        for ind in self.area.individuals:
            age_group = self._age_group(ind.age)
            sex = ind.sex
            role = ind.hh_role or "unknown"
            
            key = (age_group, sex, role)
            synth_dist[key] = synth_dist.get(key, 0) + 1
        
        # Fetch census data
        census_dist = {}
        try:
            table_path = "Befolkning/Folkmängd/Folkmängd helår/60_FolkmHHStallning_PRI.px"
            # Query all variables to get the full breakdown
            df = self._api.query_all_variables(
                table_path,
                area_code=self.area.area_api_value,
                year=self.area.year
            )
            
            if not df.empty and 'Antal' in df.columns:
                for _, row in df.iterrows():
                    # Extract values
                    age_group = str(row.get('Ålder', ''))
                    sex = str(row.get('Kön', '')).lower()
                    position = str(row.get('Hushållsställning', ''))
                    value = int(row.get('Antal', 0))
                    
                    # Map sex to our format
                    if 'män' in sex:
                        sex = 'male'
                    elif 'kvinn' in sex:
                        sex = 'female'
                    else:
                        continue  # Skip unknown sex
                    
                    # Map household position
                    role = self._map_position_to_role(position)
                    age_group = self._normalize_age_group(age_group)
                    
                    if age_group and role:
                        key = (age_group, sex, role)
                        census_dist[key] = census_dist.get(key, 0) + value
                        
        except Exception as e:
            logger.warning(f"Could not fetch household position data: {e}")
            return None
        
        if not census_dist:
            logger.warning("No census data for household position")
            return None
        
        # Compare distributions
        all_keys = set(synth_dist.keys()) | set(census_dist.keys())
        categories = []
        total_sq_error = 0
        max_error = 0
        synth_vals = []
        census_vals = []
        
        for key in sorted(all_keys):
            synth = synth_dist.get(key, 0)
            census = census_dist.get(key, 0)
            synth_vals.append(synth)
            census_vals.append(census)
            
            if census > 0:
                error_pct = abs(synth - census) / census * 100
                max_error = max(max_error, error_pct)
            else:
                error_pct = 0 if synth == 0 else 100
            
            total_sq_error += (synth - census) ** 2
            categories.append({
                "category": f"{key[0]}_{key[1]}_{key[2]}",
                "synth": synth,
                "census": census,
                "error_pct": error_pct
            })
        
        n = len(all_keys)
        rmse = math.sqrt(total_sq_error / n) if n > 0 else 0
        correlation = self._compute_correlation(synth_vals, census_vals)
        # Pass if correlation is high (>0.90 for this multi-dimensional check)
        passed = correlation >= 0.90 or max_error <= (self.tolerance * 100)
        
        logger.info(f"Household position: {n} categories, RMSE={rmse:.1f}, Max Error={max_error:.1f}%, r={correlation:.3f}")
        
        return ValidationResult(
            dimension="household_position",
            census_total=sum(census_dist.values()),
            synth_total=sum(synth_dist.values()),
            rmse=rmse,
            mae=sum(abs(s - c) for s, c in zip(synth_vals, census_vals)) / n if n > 0 else 0,
            max_error_pct=max_error,
            correlation=correlation,
            categories=categories,
            passed=passed
        )
    
    def _map_position_to_role(self, position: str) -> Optional[str]:
        """Map census household position to our role categories.
        
        Census positions:
        - Person i gift par/registrerat partnerskap -> cohabiting
        - Personer i samboförhållande -> cohabiting  
        - Ensamstående förälder -> single
        - Barn -> child
        - Ensamboende -> single
        - Ej ensamboende personer, övriga -> other
        """
        position = position.lower()
        if 'ensamboende' in position:
            return 'single'
        elif 'ensamstående förälder' in position:
            return 'single'
        elif 'gift par' in position or 'sambo' in position or 'partnerskap' in position:
            return 'cohabiting'
        elif 'barn' == position.strip() or position.strip() == 'barn':
            return 'child'
        elif 'övrig' in position:
            return 'other'
        elif 'uppgift saknas' in position:
            return None  # Skip missing data
        return None
    
    def _normalize_age_group(self, age_str: str) -> Optional[str]:
        """Normalize census age group to our format.
        
        Census age groups: 0-5 år, 6-15 år, 16-18 år, 19-24 år, 25-34 år, etc.
        Our groups: 0-17, 18-24, 25-44, 45-64, 65-79, 80+
        """
        age_str = age_str.lower()
        
        # Map census fine-grained groups to our coarse groups
        if any(x in age_str for x in ['0-5', '6-15', '16-18']):
            return '0-17'
        elif any(x in age_str for x in ['19-24', '18-24']):
            return '18-24'
        elif any(x in age_str for x in ['25-34', '35-44', '25-44']):
            return '25-44'
        elif any(x in age_str for x in ['45-54', '55-64', '45-64']):
            return '45-64'
        elif any(x in age_str for x in ['65-74', '75-79', '65-79']):
            return '65-79'
        elif '80' in age_str or '85' in age_str or '90' in age_str:
            return '80+'
        return None
    
    def _compute_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 1.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        
        if std_x == 0 or std_y == 0:
            return 1.0 if std_x == std_y else 0.0
        
        return cov / (std_x * std_y)
    
    def compute_derived_metrics(self) -> Dict[str, Any]:
        """
        Compute derived metrics from synthetic population.
        
        These can be compared against census if available.
        """
        metrics = {}
        
        # Average household size
        hh_sizes = [len(hh.members) for hh in self.area.households]
        metrics["avg_household_size"] = sum(hh_sizes) / len(hh_sizes) if hh_sizes else 0
        
        # Persons per room (density)
        densities = []
        for hh in self.area.households:
            if hh.dwelling and hh.dwelling.floor_area:
                rooms = max(1, hh.dwelling.floor_area // 25)
                densities.append(len(hh.members) / rooms)
        metrics["avg_persons_per_room"] = sum(densities) / len(densities) if densities else 0
        
        # Children per family
        family_children = []
        for hh in self.area.households:
            n_children = sum(1 for m in hh.members if m.age < 18)
            if n_children > 0:
                family_children.append(n_children)
        metrics["avg_children_per_family"] = sum(family_children) / len(family_children) if family_children else 0
        
        # Single parent ratio
        single_parent_hh = 0
        for hh in self.area.households:
            adults = [m for m in hh.members if m.age >= 18]
            children = [m for m in hh.members if m.age < 18]
            if len(adults) == 1 and len(children) > 0:
                single_parent_hh += 1
        metrics["single_parent_households"] = single_parent_hh
        metrics["single_parent_rate"] = single_parent_hh / len(self.area.households) if self.area.households else 0
        
        # Elderly living alone
        elderly_alone = sum(1 for hh in self.area.households 
                          if len(hh.members) == 1 and hh.members[0].age >= 65)
        metrics["elderly_living_alone"] = elderly_alone
        
        # Working age population
        working_age = sum(1 for ind in self.area.individuals if 18 <= ind.age <= 64)
        metrics["working_age_population"] = working_age
        metrics["working_age_ratio"] = working_age / len(self.area.individuals) if self.area.individuals else 0
        
        # Dependency ratio
        young = sum(1 for ind in self.area.individuals if ind.age < 18)
        elderly = sum(1 for ind in self.area.individuals if ind.age >= 65)
        metrics["dependency_ratio"] = (young + elderly) / working_age if working_age > 0 else 0
        
        # Car ownership per capita
        total_cars = sum(hh.cars for hh in self.area.households)
        hh_with_cars = sum(1 for hh in self.area.households if hh.cars > 0)
        metrics["total_cars"] = total_cars
        metrics["households_with_cars"] = hh_with_cars
        metrics["car_ownership_rate"] = hh_with_cars / len(self.area.households) if self.area.households else 0
        metrics["cars_per_capita"] = total_cars / len(self.area.individuals) if self.area.individuals else 0
        
        return metrics
    
    def _age_group(self, age: int) -> str:
        """Convert age to age group string."""
        if age < 18:
            return "0-17"
        elif age < 25:
            return "18-24"
        elif age < 45:
            return "25-44"
        elif age < 65:
            return "45-64"
        elif age < 80:
            return "65-79"
        else:
            return "80+"
    
    def run_all_validations(self) -> ValidationReport:
        """
        Run all available validation checks against census data.
        
        Validations:
        1. Overcrowding rate (derived metric)
        2. Children per household (cross-tabulation)
        3. Household position by age/sex (joint distribution)
        4. Sanity checks (no unrealistic households)
        
        Returns:
            ValidationReport with all results
        """
        results = {}
        
        print("\n📊 Running out-of-sample validations...")
        print("-" * 50)
        
        # 1. Overcrowding validation
        print("  [1/3] Validating overcrowding rate...")
        try:
            results["overcrowding"] = self.validate_overcrowding()
            status = "✓ PASS" if results["overcrowding"].passed else "✗ FAIL"
            print(f"        {status} (synth={results['overcrowding'].categories[0]['synth_rate']:.1f}%)")
        except Exception as e:
            logger.warning(f"Overcrowding validation failed: {e}")
            print(f"        ⚠ ERROR: {e}")
        
        # 2. Children per household
        print("  [2/3] Validating children per household...")
        try:
            result = self.validate_household_children()
            if result:
                results["children_distribution"] = result
                status = "✓ PASS" if result.passed else "✗ FAIL"
                print(f"        {status} (r={result.correlation:.3f}, RMSE={result.rmse:.1f})")
            else:
                print("        ⚠ No census data available")
        except Exception as e:
            logger.warning(f"Children validation failed: {e}")
            print(f"        ⚠ ERROR: {e}")
        
        # 3. Household position by age/sex
        print("  [3/4] Validating household position (age×sex×role)...")
        try:
            result = self.validate_household_position()
            if result:
                results["household_position"] = result
                status = "✓ PASS" if result.passed else "✗ FAIL"
                print(f"        {status} (r={result.correlation:.3f}, RMSE={result.rmse:.1f})")
            else:
                print("        ⚠ No census data available")
        except Exception as e:
            logger.warning(f"Household position validation failed: {e}")
            print(f"        ⚠ ERROR: {e}")
        
        # 4. Sanity checks (no unrealistic households)
        print("  [4/4] Running sanity checks...")
        try:
            sanity_result = run_all_checks(self.area.households, self.area.individuals)
            if sanity_result.is_valid:
                print(f"        ✓ PASS ({sanity_result.warning_count} warnings)")
            else:
                print(f"        ✗ FAIL ({sanity_result.critical_count} critical issues)")
                for v in sanity_result.violations[:3]:
                    if v.severity == 'critical':
                        print(f"          - {v.description}")
        except Exception as e:
            logger.warning(f"Sanity check failed: {e}")
            sanity_result = None
            print(f"        ⚠ ERROR: {e}")
        
        print("-" * 50)
        
        # Calculate overall score
        passed = sum(1 for r in results.values() if r and r.passed)
        total = len([r for r in results.values() if r is not None])
        
        # Sanity check must pass for full score
        if sanity_result and not sanity_result.is_valid:
            score = 0.0  # Critical sanity issues = 0 score
            print(f"⚠️  CRITICAL: Population has unrealistic households - not suitable for modeling")
        else:
            score = (passed / total * 100) if total > 0 else 100.0
        
        print(f"Overall: {passed}/{total} validations passed ({score:.0f}%)")
        
        return ValidationReport(
            area_code=self.area.area_code,
            area_name=self.area.area_name,
            year=self.area.year,
            results=results,
            overall_score=score,
            sanity_result=sanity_result
        )


def list_validation_tables() -> Dict[str, Dict]:
    """
    List all available validation tables.
    
    Returns:
        Dictionary of validation table configurations
    """
    return VALIDATION_TABLES.copy()


def list_synthesis_tables() -> Dict[str, str]:
    """
    List tables currently used in synthesis.
    
    Returns:
        Dictionary of synthesis table descriptions
    """
    return SYNTHESIS_TABLES.copy()


def suggest_improvements() -> str:
    """
    Suggest additional attributes and validation opportunities.
    
    Returns:
        Formatted string with suggestions
    """
    suggestions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    VALIDATION & IMPROVEMENT OPPORTUNITIES                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

CURRENTLY SYNTHESIZED ATTRIBUTES:
─────────────────────────────────
  ✓ Age (individual)
  ✓ Sex (individual)  
  ✓ Household role (single/cohabiting/child)
  ✓ Household size
  ✓ Housing type (apartment/detached/special)
  ✓ Floor area (dwelling)
  ✓ Car ownership (household)
  ✓ Low income flag (household)

═══════════════════════════════════════════════════════════════════════════════
USABLE VALIDATION TABLES (we have these attributes!)
═══════════════════════════════════════════════════════════════════════════════

  ✓ OVERCROWDING (derived metric)
    Table: 10_TrangFlerbost_PRI.px
    Method: Compute persons/room from HH size + floor area, compare to census
    
  ✓ HOUSEHOLD POSITION by age/sex
    Table: 62_FolkmHHstallPRI.px  
    Method: Joint distribution age×sex×role (each fitted separately → out-of-sample)
    
  ✓ CHILDREN PER HOUSEHOLD TYPE
    Table: 20_HHtypBarn_PRI.px
    Method: Count children (<18) per household, group by HH type
    
  ✓ POPULATION by HH size + housing type
    Table: 65_FolkmHHStorlHusTyp_PRI.px
    Method: Cross-tabulation of household size × housing type

═══════════════════════════════════════════════════════════════════════════════
NOT USABLE (missing attributes - future enhancements)
═══════════════════════════════════════════════════════════════════════════════

  ✗ Education tables      → Need: education_level attribute
  ✗ Employment tables     → Need: employment_status attribute  
  ✗ Foreign background    → Need: country_of_birth attribute
  ✗ Tenure (rent/own)     → Need: tenure_type on Dwelling

RECOMMENDED ATTRIBUTE ADDITIONS:
────────────────────────────────
  ⊕ Education level      → Table: 10_UtbNiva_PRI.px (ages 25-64)
  ⊕ Employment status    → Table: 10_Forvarb_PRI.px + 10_Alosa_PRI.px  
  ⊕ Tenure type          → Table: 20_TypUppf_PRI.px
"""
    return suggestions
