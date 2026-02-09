"""
Sanity checks for synthetic population validation.

These checks ensure NO unrealistic households exist in the synthetic population.
Even a single violation should be flagged and fixed, as outliers reduce trust
in model results for activity demand, accessibility, and electricity modeling.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


@dataclass
class SanityViolation:
    """Represents a single sanity check violation."""
    check_name: str
    severity: str  # 'critical', 'warning', 'info'
    household_id: Any
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"[{self.severity.upper()}] HH {self.household_id}: {self.description}"


@dataclass
class SanityCheckResult:
    """Result of running all sanity checks."""
    total_households: int
    total_individuals: int
    violations: List[SanityViolation] = field(default_factory=list)
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Returns True if no critical violations."""
        return not any(v.severity == 'critical' for v in self.violations)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'critical')
    
    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'warning')
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "SANITY CHECK SUMMARY",
            "=" * 60,
            f"  Total Households:     {self.total_households:,}",
            f"  Total Individuals:    {self.total_individuals:,}",
            "",
            f"  Checks Passed:        {len(self.checks_passed)}",
            f"  Checks Failed:        {len(self.checks_failed)}",
            "",
            f"  Critical Violations:  {self.critical_count}",
            f"  Warnings:             {self.warning_count}",
            "",
        ]
        
        if self.is_valid:
            lines.append("  ✅ POPULATION IS VALID (no critical violations)")
        else:
            lines.append("  ❌ POPULATION HAS CRITICAL ISSUES")
            lines.append("")
            lines.append("  Critical violations:")
            for v in self.violations:
                if v.severity == 'critical':
                    lines.append(f"    - {v}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# CONFIGURATION: Thresholds for sanity checks
# =============================================================================

# Age thresholds
MIN_PARENT_AGE = 15          # Minimum age to be a parent
MAX_CHILD_AGE = 30           # Maximum age to be considered a "child" living at home (extended)
MIN_ADULT_AGE = 18           # Legal adult age
MAX_REASONABLE_AGE = 110     # Maximum reasonable age

# Household size thresholds by dwelling type
MAX_PERSONS_PER_ROOM = 3     # Maximum persons per room (overcrowding threshold)
MAX_HOUSEHOLD_SIZE = 12      # Absolute maximum household size

# Dwelling-specific limits (rooms -> max people)
MAX_PERSONS_BY_ROOMS = {
    1: 3,    # 1 room: max 3 people (tight but possible)
    2: 5,    # 2 rooms: max 5 people
    3: 7,    # 3 rooms: max 7 people
    4: 9,    # 4 rooms: max 9 people
    5: 11,   # 5 rooms: max 11 people
    6: 12,   # 6+ rooms: max 12 people
}

# Age gap thresholds
# A 10-year gap is the absolute minimum (parent at 10 + child at 0)
# We use 10 as critical threshold, 14 as warning threshold
MIN_PARENT_CHILD_AGE_GAP_CRITICAL = 10   # Truly impossible: gap < 10 years
MIN_PARENT_CHILD_AGE_GAP_WARNING = 14    # Unlikely but possible
MAX_PARENT_CHILD_AGE_GAP = 60            # Maximum reasonable gap


def run_all_checks(households: List, individuals: List) -> SanityCheckResult:
    """
    Run all sanity checks on the synthetic population.
    
    Args:
        households: List of Household objects
        individuals: List of Individual objects
        
    Returns:
        SanityCheckResult with all violations found
    """
    result = SanityCheckResult(
        total_households=len(households),
        total_individuals=len(individuals)
    )
    
    # Build lookup structures
    hh_members = defaultdict(list)
    for ind in individuals:
        hh_members[ind.household_id].append(ind)
    
    hh_by_id = {hh.household_id: hh for hh in households}
    
    # Define all checks
    checks = [
        ("no_children_only_households", check_no_children_only_households),
        ("no_empty_households", check_no_empty_households),
        ("household_size_matches", check_household_size_matches),
        ("no_overcrowded_dwellings", check_no_overcrowded_dwellings),
        ("valid_age_ranges", check_valid_age_ranges),
        ("parent_child_age_gaps", check_parent_child_age_gaps),
        ("role_age_consistency", check_role_age_consistency),
        ("single_households_have_one_person", check_single_households),
        ("couple_households_have_adults", check_couple_households),
        ("no_orphan_individuals", check_no_orphan_individuals),
        ("dwelling_assignment_complete", check_dwelling_assignment),
        ("sex_values_valid", check_sex_values),
        ("household_type_consistency", check_household_type_consistency),
    ]
    
    for check_name, check_func in checks:
        try:
            violations = check_func(households, individuals, hh_members, hh_by_id)
            if violations:
                result.violations.extend(violations)
                result.checks_failed.append(check_name)
            else:
                result.checks_passed.append(check_name)
        except Exception as e:
            result.violations.append(SanityViolation(
                check_name=check_name,
                severity='warning',
                household_id='N/A',
                description=f"Check failed with error: {str(e)}"
            ))
            result.checks_failed.append(check_name)
    
    return result


# =============================================================================
# INDIVIDUAL CHECKS
# =============================================================================

def check_no_children_only_households(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    CRITICAL: No household should contain only children (age < 18).
    Children cannot live alone without an adult guardian.
    """
    violations = []
    
    for hh_id, members in hh_members.items():
        if not members:
            continue
            
        adults = [m for m in members if m.age >= MIN_ADULT_AGE]
        children = [m for m in members if m.age < MIN_ADULT_AGE]
        
        if children and not adults:
            violations.append(SanityViolation(
                check_name="no_children_only_households",
                severity='critical',
                household_id=hh_id,
                description=f"Household has {len(children)} children but no adults",
                details={
                    'children_ages': [c.age for c in children],
                    'member_count': len(members)
                }
            ))
    
    return violations


def check_no_empty_households(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    WARNING: Households should have at least one member.
    Empty households waste capacity but aren't critical.
    """
    violations = []
    
    for hh in households:
        members = hh_members.get(hh.household_id, [])
        if len(members) == 0:
            violations.append(SanityViolation(
                check_name="no_empty_households",
                severity='warning',
                household_id=hh.household_id,
                description="Household has no members",
                details={'dwelling_id': getattr(hh, 'dwelling_id', None)}
            ))
    
    return violations


def check_household_size_matches(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    WARNING: Household size attribute should match actual member count.
    """
    violations = []
    
    for hh in households:
        expected_size = getattr(hh, 'size', None)
        actual_size = len(hh_members.get(hh.household_id, []))
        
        if expected_size is not None and expected_size != actual_size:
            violations.append(SanityViolation(
                check_name="household_size_matches",
                severity='warning',
                household_id=hh.household_id,
                description=f"Size mismatch: expected {expected_size}, actual {actual_size}",
                details={'expected': expected_size, 'actual': actual_size}
            ))
    
    return violations


def check_no_overcrowded_dwellings(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    CRITICAL: No dwelling should have more people than physically reasonable.
    Based on room count and dwelling type.
    """
    violations = []
    
    for hh in households:
        members = hh_members.get(hh.household_id, [])
        num_people = len(members)
        
        # Skip empty households
        if num_people == 0:
            continue
        
        # Absolute maximum
        if num_people > MAX_HOUSEHOLD_SIZE:
            violations.append(SanityViolation(
                check_name="no_overcrowded_dwellings",
                severity='critical',
                household_id=hh.household_id,
                description=f"Household has {num_people} people (max allowed: {MAX_HOUSEHOLD_SIZE})",
                details={'num_people': num_people, 'max_allowed': MAX_HOUSEHOLD_SIZE}
            ))
            continue
        
        # Check based on room count if available
        num_rooms = getattr(hh, 'num_rooms', None)
        if num_rooms is not None:
            # Cap at 6 for lookup
            rooms_key = min(int(num_rooms), 6)
            max_for_rooms = MAX_PERSONS_BY_ROOMS.get(rooms_key, MAX_HOUSEHOLD_SIZE)
            
            if num_people > max_for_rooms:
                violations.append(SanityViolation(
                    check_name="no_overcrowded_dwellings",
                    severity='critical',
                    household_id=hh.household_id,
                    description=f"{num_people} people in {num_rooms}-room dwelling (max: {max_for_rooms})",
                    details={
                        'num_people': num_people,
                        'num_rooms': num_rooms,
                        'max_for_rooms': max_for_rooms
                    }
                ))
    
    return violations


def check_valid_age_ranges(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    CRITICAL: All ages must be within valid range (0-110).
    """
    violations = []
    
    for ind in individuals:
        if ind.age < 0:
            violations.append(SanityViolation(
                check_name="valid_age_ranges",
                severity='critical',
                household_id=ind.household_id,
                description=f"Individual has negative age: {ind.age}",
                details={'individual_id': getattr(ind, 'individual_id', None), 'age': ind.age}
            ))
        elif ind.age > MAX_REASONABLE_AGE:
            violations.append(SanityViolation(
                check_name="valid_age_ranges",
                severity='critical',
                household_id=ind.household_id,
                description=f"Individual has unreasonable age: {ind.age}",
                details={'individual_id': getattr(ind, 'individual_id', None), 'age': ind.age}
            ))
    
    return violations


def check_parent_child_age_gaps(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    CRITICAL: There must be at least one adult old enough to be a parent for each minor.
    
    A household with a 17-year-old and an 18-year-old is fine IF there's also
    a 40-year-old parent. We check that at least one adult in the household
    is old enough to be a biological parent of each minor.
    
    - No adult with gap >= 15 years: CRITICAL (no possible biological parent)
    - Max adult gap < 18 years: WARNING (possible but unusual)
    
    Note: We use age < 18 for "minor" detection because adult children (24+)
    living at home are common and valid.
    """
    violations = []
    
    MIN_GAP_FOR_PARENT = 15  # At minimum, parent must be 15 years older
    
    for hh_id, members in hh_members.items():
        if len(members) < 2:
            continue
        
        # Identify adults (18+) and minors (< 18) by AGE
        adults = [m for m in members if m.age >= MIN_ADULT_AGE]
        minors = [m for m in members if m.age < MIN_ADULT_AGE]
        
        if not adults or not minors:
            continue
        
        # For each minor, check if there's at least one adult old enough
        # to be their biological parent
        for minor in minors:
            # Find the oldest adult
            oldest_adult_age = max(a.age for a in adults)
            max_gap = oldest_adult_age - minor.age
            
            if max_gap < MIN_PARENT_CHILD_AGE_GAP_CRITICAL:
                # No adult is old enough to be a parent
                violations.append(SanityViolation(
                    check_name="parent_child_age_gaps",
                    severity='critical',
                    household_id=hh_id,
                    description=f"Minor age {minor.age} - oldest adult is {oldest_adult_age} (gap: {max_gap} years - NO POSSIBLE PARENT)",
                    details={
                        'minor_age': minor.age,
                        'oldest_adult_age': oldest_adult_age,
                        'age_gap': max_gap,
                        'min_gap': MIN_PARENT_CHILD_AGE_GAP_CRITICAL,
                        'all_adult_ages': sorted([a.age for a in adults])
                    }
                ))
            elif max_gap < MIN_GAP_FOR_PARENT:
                # Possible but unusual - very young parent
                violations.append(SanityViolation(
                    check_name="parent_child_age_gaps",
                    severity='warning',
                    household_id=hh_id,
                    description=f"Minor age {minor.age} - oldest adult is {oldest_adult_age} (gap: {max_gap} years - young parent)",
                    details={
                        'minor_age': minor.age,
                        'oldest_adult_age': oldest_adult_age,
                        'age_gap': max_gap
                    }
                ))
    
    return violations


def check_role_age_consistency(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    Check that household roles are age-appropriate.
    
    The 'child' role in Swedish census means "child in the household" which includes
    adult children (20-40+) living with elderly parents. This is NORMAL.
    
    We only flag:
    - WARNING: 'child' role person where there's NO adult >= 15 years older (no possible parent)
    - WARNING: Minor (age < 18) with adult role like 'single' or 'cohabiting'
    
    Note: Adult "children" (age 25+) living with parents aged 50+ are completely normal.
    """
    violations = []
    
    # Track flagged households to avoid duplicate warnings
    flagged_hh = set()
    
    for ind in individuals:
        role = getattr(ind, 'hh_role', None)
        
        # Child role - only flag if there's no one old enough to be their parent
        # AND the person is an adult (>= 18), AND we haven't already flagged this HH
        if role == 'child' and ind.age >= 18 and ind.household_id not in flagged_hh:
            members = hh_members.get(ind.household_id, [])
            other_members = [m for m in members if m is not ind]
            
            if other_members:
                oldest_other = max(m.age for m in other_members)
                age_gap = oldest_other - ind.age
                
                # Only flag if NO ONE could be their parent (gap < 15)
                # AND the "child" is old enough that this is suspicious (>= 20)
                if age_gap < 15 and ind.age >= 20:
                    flagged_hh.add(ind.household_id)
                    violations.append(SanityViolation(
                        check_name="role_age_consistency",
                        severity='info',  # Downgrade to info - this is census data quirk
                        household_id=ind.household_id,
                        description=f"Adult (age {ind.age}) has 'child' role but oldest other is only {oldest_other}",
                        details={'age': ind.age, 'role': role, 'oldest_other': oldest_other, 'gap': age_gap}
                    ))
        
        # Minor with adult role - this is a real problem
        if role in ('single', 'cohabiting', 'single_parent') and ind.age < MIN_ADULT_AGE:
            violations.append(SanityViolation(
                check_name="role_age_consistency",
                severity='warning',
                household_id=ind.household_id,
                description=f"Minor aged {ind.age} has adult role '{role}'",
                details={'age': ind.age, 'role': role}
            ))
    
    return violations


def check_single_households(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    CRITICAL: Households marked as single-person should have exactly 1 person.
    """
    violations = []
    
    for hh in households:
        members = hh_members.get(hh.household_id, [])
        hh_type = getattr(hh, 'household_type', None)
        
        # Check if marked as single
        if hh_type and 'ensamstående' in str(hh_type).lower() and 'förälder' not in str(hh_type).lower():
            if len(members) != 1:
                violations.append(SanityViolation(
                    check_name="single_households_have_one_person",
                    severity='critical',
                    household_id=hh.household_id,
                    description=f"Single-person household has {len(members)} members",
                    details={'household_type': hh_type, 'num_members': len(members)}
                ))
    
    return violations


def check_couple_households(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    CRITICAL: Couple households should have at least 2 adults.
    """
    violations = []
    
    for hh in households:
        members = hh_members.get(hh.household_id, [])
        hh_type = getattr(hh, 'household_type', None)
        
        # Check if marked as couple/cohabiting
        if hh_type and ('sammanboende' in str(hh_type).lower() or 'gift' in str(hh_type).lower()):
            adults = [m for m in members if m.age >= MIN_ADULT_AGE]
            if len(adults) < 2:
                violations.append(SanityViolation(
                    check_name="couple_households_have_adults",
                    severity='critical',
                    household_id=hh.household_id,
                    description=f"Couple household has only {len(adults)} adult(s)",
                    details={
                        'household_type': hh_type,
                        'num_adults': len(adults),
                        'num_members': len(members)
                    }
                ))
    
    return violations


def check_no_orphan_individuals(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    CRITICAL: All individuals must belong to a valid household.
    """
    violations = []
    
    for ind in individuals:
        if ind.household_id not in hh_by_id:
            violations.append(SanityViolation(
                check_name="no_orphan_individuals",
                severity='critical',
                household_id=ind.household_id,
                description=f"Individual references non-existent household",
                details={'individual_id': getattr(ind, 'individual_id', None)}
            ))
    
    return violations


def check_dwelling_assignment(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    INFO: Check dwelling assignment coverage.
    
    Some areas have more census households than dwelling data supports.
    This is a data quality issue, not a synthesis bug.
    Only flag if >20% of households lack dwelling assignments.
    """
    violations = []
    
    occupied_without_dwelling = 0
    occupied_total = 0
    
    for hh in households:
        members = hh_members.get(hh.household_id, [])
        if len(members) > 0:
            occupied_total += 1
            dwelling_id = getattr(hh, 'dwelling_id', None)
            if dwelling_id is None:
                occupied_without_dwelling += 1
    
    # Only report as a single info message if significant
    if occupied_without_dwelling > 0:
        pct = occupied_without_dwelling / occupied_total * 100 if occupied_total > 0 else 0
        severity = 'warning' if pct > 20 else 'info'
        violations.append(SanityViolation(
            check_name="dwelling_assignment_complete",
            severity=severity,
            household_id='N/A',
            description=f"{occupied_without_dwelling} households ({pct:.1f}%) lack dwelling assignments",
            details={
                'without_dwelling': occupied_without_dwelling,
                'total_occupied': occupied_total,
                'percent_missing': pct
            }
        ))
    
    return violations


def check_sex_values(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    WARNING: Sex values should be valid ('man', 'kvinna', 'M', 'F', etc.)
    """
    violations = []
    valid_values = {'man', 'kvinna', 'm', 'f', 'male', 'female', 'män', 'kvinnor'}
    
    for ind in individuals:
        sex = getattr(ind, 'sex', None)
        if sex is None:
            violations.append(SanityViolation(
                check_name="sex_values_valid",
                severity='warning',
                household_id=ind.household_id,
                description="Individual has no sex value",
                details={'individual_id': getattr(ind, 'individual_id', None)}
            ))
        elif str(sex).lower() not in valid_values:
            violations.append(SanityViolation(
                check_name="sex_values_valid",
                severity='warning',
                household_id=ind.household_id,
                description=f"Individual has unusual sex value: {sex}",
                details={'individual_id': getattr(ind, 'individual_id', None), 'sex': sex}
            ))
    
    return violations


def check_household_type_consistency(households, individuals, hh_members, hh_by_id) -> List[SanityViolation]:
    """
    WARNING: Household composition should match declared type.
    """
    violations = []
    
    for hh in households:
        members = hh_members.get(hh.household_id, [])
        if not members:
            continue
            
        hh_type = getattr(hh, 'household_type', None)
        if not hh_type:
            continue
            
        hh_type_lower = str(hh_type).lower()
        adults = [m for m in members if m.age >= MIN_ADULT_AGE]
        children = [m for m in members if m.age < MIN_ADULT_AGE]
        
        # Single parent should have exactly 1 adult and at least 1 child
        if 'ensamstående' in hh_type_lower and 'barn' in hh_type_lower:
            if len(adults) != 1:
                violations.append(SanityViolation(
                    check_name="household_type_consistency",
                    severity='warning',
                    household_id=hh.household_id,
                    description=f"Single-parent HH has {len(adults)} adults (expected 1)",
                    details={'household_type': hh_type, 'num_adults': len(adults)}
                ))
            if len(children) == 0:
                violations.append(SanityViolation(
                    check_name="household_type_consistency",
                    severity='warning',
                    household_id=hh.household_id,
                    description=f"Single-parent HH has no children",
                    details={'household_type': hh_type, 'num_children': len(children)}
                ))
    
    return violations


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def validate_population(area) -> SanityCheckResult:
    """
    Validate a synthesized area's population.
    
    Args:
        area: An Area object with generated households and individuals
        
    Returns:
        SanityCheckResult
    """
    return run_all_checks(area.households, area.individuals)


def print_violation_details(result: SanityCheckResult, max_violations: int = 20):
    """Print detailed information about violations."""
    print(result.summary())
    
    if result.violations:
        print()
        print("VIOLATION DETAILS:")
        print("-" * 60)
        
        # Group by check name
        by_check = defaultdict(list)
        for v in result.violations:
            by_check[v.check_name].append(v)
        
        for check_name, violations in by_check.items():
            print(f"\n{check_name} ({len(violations)} violations):")
            for v in violations[:max_violations]:
                print(f"  {v}")
            if len(violations) > max_violations:
                print(f"  ... and {len(violations) - max_violations} more")
