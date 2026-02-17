"""
Tests for sanity_checks module.

Tests the sanity check framework to ensure it correctly identifies
invalid and unrealistic synthetic population configurations.
"""

import pytest
from gbgsynth.models import Agent, Household
from gbgsynth.sanity_checks import (
    run_all_checks,
    SanityViolation,
    SanityCheckResult,
    check_no_children_only_households,
    check_no_empty_households,
    check_household_size_matches,
    check_no_overcrowded_dwellings,
    check_valid_age_ranges,
    check_parent_child_age_gaps,
    check_role_age_consistency,
    check_single_households,
    check_couple_households,
    MIN_ADULT_AGE,
    MAX_HOUSEHOLD_SIZE,
)
from collections import defaultdict


class TestSanityViolation:
    """Test the SanityViolation dataclass."""
    
    def test_violation_creation(self):
        """Test creating a violation."""
        v = SanityViolation(
            check_name="test_check",
            severity="critical",
            household_id=123,
            description="Test violation"
        )
        assert v.check_name == "test_check"
        assert v.severity == "critical"
        assert v.household_id == 123
    
    def test_violation_string_representation(self):
        """Test string representation of violation."""
        v = SanityViolation(
            check_name="test",
            severity="warning",
            household_id=456,
            description="Test issue"
        )
        s = str(v)
        assert "WARNING" in s
        assert "456" in s
        assert "Test issue" in s


class TestSanityCheckResult:
    """Test the SanityCheckResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a result."""
        result = SanityCheckResult(
            total_households=100,
            total_individuals=250
        )
        assert result.total_households == 100
        assert result.total_individuals == 250
        assert result.violations == []
    
    def test_is_valid_no_violations(self):
        """Test is_valid returns True when no critical violations."""
        result = SanityCheckResult(
            total_households=10,
            total_individuals=25
        )
        assert result.is_valid is True
    
    def test_is_valid_only_warnings(self):
        """Test is_valid returns True when only warnings."""
        result = SanityCheckResult(
            total_households=10,
            total_individuals=25,
            violations=[
                SanityViolation("test", "warning", 1, "Warning issue")
            ]
        )
        assert result.is_valid is True
    
    def test_is_valid_with_critical(self):
        """Test is_valid returns False when critical violations exist."""
        result = SanityCheckResult(
            total_households=10,
            total_individuals=25,
            violations=[
                SanityViolation("test", "critical", 1, "Critical issue")
            ]
        )
        assert result.is_valid is False
    
    def test_critical_count(self):
        """Test counting critical violations."""
        result = SanityCheckResult(
            total_households=10,
            total_individuals=25,
            violations=[
                SanityViolation("test1", "critical", 1, "Issue 1"),
                SanityViolation("test2", "warning", 2, "Issue 2"),
                SanityViolation("test3", "critical", 3, "Issue 3"),
            ]
        )
        assert result.critical_count == 2
    
    def test_warning_count(self):
        """Test counting warnings."""
        result = SanityCheckResult(
            total_households=10,
            total_individuals=25,
            violations=[
                SanityViolation("test1", "critical", 1, "Issue 1"),
                SanityViolation("test2", "warning", 2, "Issue 2"),
                SanityViolation("test3", "warning", 3, "Issue 3"),
            ]
        )
        assert result.warning_count == 2
    
    def test_summary_generation(self):
        """Test summary string generation."""
        result = SanityCheckResult(
            total_households=100,
            total_individuals=250
        )
        summary = result.summary()
        assert "100" in summary
        assert "250" in summary


class TestCheckNoChildrenOnlyHouseholds:
    """Test check_no_children_only_households function."""
    
    def test_valid_household_with_adults_and_children(self):
        """Test household with both adults and children passes."""
        hh = Household(household_id=1, size=3)
        adult = Agent(agent_id=1, age=35, sex='male')
        adult.household_id = 1
        child1 = Agent(agent_id=2, age=10, sex='female')
        child1.household_id = 1
        child2 = Agent(agent_id=3, age=15, sex='male')
        child2.household_id = 1
        
        hh_members = {1: [adult, child1, child2]}
        violations = check_no_children_only_households([hh], [adult, child1, child2], hh_members, {1: hh})
        
        assert len(violations) == 0
    
    def test_children_only_household_fails(self):
        """Test household with only children fails critically."""
        hh = Household(household_id=1, size=2)
        child1 = Agent(agent_id=1, age=10, sex='male')
        child1.household_id = 1
        child2 = Agent(agent_id=2, age=15, sex='female')
        child2.household_id = 1
        
        hh_members = {1: [child1, child2]}
        violations = check_no_children_only_households([hh], [child1, child2], hh_members, {1: hh})
        
        assert len(violations) == 1
        assert violations[0].severity == 'critical'
        assert violations[0].household_id == 1
        assert '2 children' in violations[0].description
    
    def test_adult_only_household_passes(self):
        """Test household with only adults passes."""
        hh = Household(household_id=1, size=2)
        adult1 = Agent(agent_id=1, age=30, sex='male')
        adult1.household_id = 1
        adult2 = Agent(agent_id=2, age=28, sex='female')
        adult2.household_id = 1
        
        hh_members = {1: [adult1, adult2]}
        violations = check_no_children_only_households([hh], [adult1, adult2], hh_members, {1: hh})
        
        assert len(violations) == 0


class TestCheckNoEmptyHouseholds:
    """Test check_no_empty_households function."""
    
    def test_household_with_members_passes(self):
        """Test household with members passes."""
        hh = Household(household_id=1, size=1)
        agent = Agent(agent_id=1, age=30, sex='male')
        agent.household_id = 1
        
        hh_members = {1: [agent]}
        violations = check_no_empty_households([hh], [agent], hh_members, {1: hh})
        
        assert len(violations) == 0
    
    def test_empty_household_warns(self):
        """Test empty household generates warning."""
        # Note: Household model enforces size >= 1, but a household can still have no members
        # if they weren't assigned yet. This tests the case where size=1 but no members assigned.
        hh = Household(household_id=1, size=1)
        
        hh_members = {1: []}
        violations = check_no_empty_households([hh], [], hh_members, {1: hh})
        
        assert len(violations) == 1
        assert violations[0].severity == 'warning'
        assert violations[0].household_id == 1


class TestCheckHouseholdSizeMatches:
    """Test check_household_size_matches function."""
    
    def test_size_matches_actual_members(self):
        """Test household where size matches actual members."""
        hh = Household(household_id=1, size=3)
        agent1 = Agent(agent_id=1, age=40, sex='male')
        agent1.household_id = 1
        agent2 = Agent(agent_id=2, age=38, sex='female')
        agent2.household_id = 1
        agent3 = Agent(agent_id=3, age=10, sex='male')
        agent3.household_id = 1
        
        hh_members = {1: [agent1, agent2, agent3]}
        violations = check_household_size_matches([hh], [agent1, agent2, agent3], hh_members, {1: hh})
        
        assert len(violations) == 0
    
    def test_size_mismatch_warns(self):
        """Test household with size mismatch generates warning."""
        hh = Household(household_id=1, size=3)
        agent1 = Agent(agent_id=1, age=40, sex='male')
        agent1.household_id = 1
        agent2 = Agent(agent_id=2, age=38, sex='female')
        agent2.household_id = 1
        
        hh_members = {1: [agent1, agent2]}
        violations = check_household_size_matches([hh], [agent1, agent2], hh_members, {1: hh})
        
        assert len(violations) == 1
        assert violations[0].severity == 'warning'
        assert 'expected 3, actual 2' in violations[0].description


class TestCheckOvercrowdedDwellings:
    """Test check_no_overcrowded_dwellings function."""
    
    def test_reasonable_occupancy_passes(self):
        """Test household with reasonable occupancy passes."""
        hh = Household(household_id=1, size=4)
        hh.num_rooms = 3
        agents = [Agent(agent_id=i, age=20+i*5, sex='male' if i % 2 == 0 else 'female') for i in range(4)]
        for agent in agents:
            agent.household_id = 1
        
        hh_members = {1: agents}
        violations = check_no_overcrowded_dwellings([hh], agents, hh_members, {1: hh})
        
        assert len(violations) == 0
    
    def test_absolute_max_exceeded_fails(self):
        """Test household exceeding absolute maximum fails."""
        hh = Household(household_id=1, size=15)
        agents = [Agent(agent_id=i, age=20+i, sex='male' if i % 2 == 0 else 'female') for i in range(15)]
        for agent in agents:
            agent.household_id = 1
        
        hh_members = {1: agents}
        violations = check_no_overcrowded_dwellings([hh], agents, hh_members, {1: hh})
        
        assert len(violations) == 1
        assert violations[0].severity == 'critical'
        assert violations[0].household_id == 1
    
    def test_room_based_overcrowding_fails(self):
        """Test overcrowding based on room count fails."""
        hh = Household(household_id=1, size=8)
        hh.num_rooms = 2  # Max 5 people for 2 rooms
        agents = [Agent(agent_id=i, age=20+i*3, sex='male' if i % 2 == 0 else 'female') for i in range(8)]
        for agent in agents:
            agent.household_id = 1
        
        hh_members = {1: agents}
        violations = check_no_overcrowded_dwellings([hh], agents, hh_members, {1: hh})
        
        assert len(violations) == 1
        assert violations[0].severity == 'critical'


class TestCheckValidAgeRanges:
    """Test check_valid_age_ranges function."""
    
    def test_valid_ages_pass(self):
        """Test individuals with valid ages pass."""
        agents = [
            Agent(agent_id=1, age=0, sex='male'),
            Agent(agent_id=2, age=25, sex='female'),
            Agent(agent_id=3, age=65, sex='male'),
            Agent(agent_id=4, age=95, sex='female'),
        ]
        for agent in agents:
            agent.household_id = 1
        
        violations = check_valid_age_ranges([], agents, {}, {})
        
        assert len(violations) == 0
    
    def test_negative_age_fails(self):
        """Test negative age fails critically.
        
        Note: The Agent model now validates ages in __post_init__, so we can't create
        invalid agents. This test verifies that the model prevents invalid ages.
        """
        with pytest.raises(ValueError, match="Invalid age"):
            Agent(agent_id=1, age=-5, sex='male')
    
    def test_unreasonable_age_fails(self):
        """Test unreasonably high age fails critically.
        
        Note: The Agent model now validates ages in __post_init__, so we can't create
        invalid agents. This test verifies that the model prevents invalid ages.
        """
        with pytest.raises(ValueError, match="Invalid age"):
            Agent(agent_id=1, age=150, sex='female')


class TestCheckParentChildAgeGaps:
    """Test check_parent_child_age_gaps function."""
    
    def test_valid_parent_child_gap_passes(self):
        """Test household with valid parent-child age gap passes."""
        hh = Household(household_id=1, size=3)
        parent1 = Agent(agent_id=1, age=40, sex='male')
        parent1.household_id = 1
        parent2 = Agent(agent_id=2, age=38, sex='female')
        parent2.household_id = 1
        child = Agent(agent_id=3, age=10, sex='male')
        child.household_id = 1
        
        hh_members = {1: [parent1, parent2, child]}
        violations = check_parent_child_age_gaps([hh], [parent1, parent2, child], hh_members, {1: hh})
        
        assert len(violations) == 0
    
    def test_impossible_parent_age_fails(self):
        """Test household with impossible parent age fails critically."""
        hh = Household(household_id=1, size=2)
        young_adult = Agent(agent_id=1, age=18, sex='male')
        young_adult.household_id = 1
        teen = Agent(agent_id=2, age=16, sex='male')
        teen.household_id = 1
        
        hh_members = {1: [young_adult, teen]}
        violations = check_parent_child_age_gaps([hh], [young_adult, teen], hh_members, {1: hh})
        
        # Gap is 2 years, which is < 10, so should fail critically
        assert len(violations) >= 1
        critical_violations = [v for v in violations if v.severity == 'critical']
        assert len(critical_violations) >= 1
    
    def test_no_minors_no_violation(self):
        """Test household with no minors has no violations."""
        hh = Household(household_id=1, size=2)
        adult1 = Agent(agent_id=1, age=30, sex='male')
        adult1.household_id = 1
        adult2 = Agent(agent_id=2, age=28, sex='female')
        adult2.household_id = 1
        
        hh_members = {1: [adult1, adult2]}
        violations = check_parent_child_age_gaps([hh], [adult1, adult2], hh_members, {1: hh})
        
        assert len(violations) == 0


class TestCheckRoleAgeConsistency:
    """Test check_role_age_consistency function."""
    
    def test_minor_with_adult_role_warns(self):
        """Test minor with adult role generates warning.
        
        Note: Agent model auto-assigns 'child' role to anyone under 18 in __post_init__,
        so we need to manually override the role after creation.
        """
        agent = Agent(agent_id=1, age=15, sex='male')
        agent.household_id = 1
        # Manually override the auto-assigned role to test the check
        agent.hh_role = 'single'
        
        hh_members = {1: [agent]}
        violations = check_role_age_consistency([], [agent], hh_members, {})
        
        assert len(violations) >= 1
        # Should be warning severity
        warnings = [v for v in violations if v.severity == 'warning']
        assert len(warnings) >= 1
    
    def test_adult_child_living_at_home_acceptable(self):
        """Test adult child living with parents is acceptable."""
        parent = Agent(agent_id=1, age=55, sex='male', hh_role='cohabiting')
        parent.household_id = 1
        adult_child = Agent(agent_id=2, age=25, sex='male', hh_role='child')
        adult_child.household_id = 1
        
        hh_members = {1: [parent, adult_child]}
        violations = check_role_age_consistency([], [parent, adult_child], hh_members, {})
        
        # Adult child with parent 30 years older should not generate violations
        # (or at most info level)
        critical_or_warning = [v for v in violations if v.severity in ('critical', 'warning')]
        assert len(critical_or_warning) == 0


class TestRunAllChecks:
    """Test run_all_checks main function."""
    
    def test_valid_population_passes(self):
        """Test a valid population passes all checks."""
        # Create valid household
        hh = Household(household_id=1, size=3)
        parent1 = Agent(agent_id=1, age=40, sex='male', hh_role='cohabiting')
        parent1.household_id = 1
        parent2 = Agent(agent_id=2, age=38, sex='female', hh_role='cohabiting')
        parent2.household_id = 1
        child = Agent(agent_id=3, age=10, sex='male', hh_role='child')
        child.household_id = 1
        
        result = run_all_checks([hh], [parent1, parent2, child])
        
        assert result.total_households == 1
        assert result.total_individuals == 3
        assert result.is_valid is True
        assert len(result.checks_passed) > 0
    
    def test_invalid_population_reports_violations(self):
        """Test an invalid population reports violations."""
        # Create household with only children (invalid)
        hh = Household(household_id=1, size=2)
        child1 = Agent(agent_id=1, age=10, sex='male', hh_role='child')
        child1.household_id = 1
        child2 = Agent(agent_id=2, age=12, sex='female', hh_role='child')
        child2.household_id = 1
        
        result = run_all_checks([hh], [child1, child2])
        
        assert result.total_households == 1
        assert result.total_individuals == 2
        assert result.is_valid is False
        assert result.critical_count > 0
        assert len(result.checks_failed) > 0
    
    def test_multiple_violations_detected(self):
        """Test multiple violations are all detected."""
        # Create multiple problematic households
        hh1 = Household(household_id=1, size=2)
        child1 = Agent(agent_id=1, age=10, sex='male')  # Children only
        child1.household_id = 1
        child2 = Agent(agent_id=2, age=12, sex='female')
        child2.household_id = 1
        
        # Create household with size mismatch (another type of violation)
        hh2 = Household(household_id=2, size=3)
        adult = Agent(agent_id=3, age=30, sex='male')
        adult.household_id = 2
        
        result = run_all_checks([hh1, hh2], [child1, child2, adult])
        
        assert result.total_households == 2
        assert result.total_individuals == 3
        assert result.is_valid is False
        # Should have at least 1 critical violation (children-only household)
        assert result.critical_count >= 1
        # Should have violations of different types
        assert len(result.checks_failed) >= 2
    
    def test_empty_population(self):
        """Test empty population doesn't crash."""
        result = run_all_checks([], [])
        
        assert result.total_households == 0
        assert result.total_individuals == 0
        # Should pass checks or have no violations
        assert len(result.violations) == 0 or result.is_valid


class TestSanityCheckIntegration:
    """Integration tests for the entire sanity check system."""
    
    def test_realistic_population_scenario(self):
        """Test a realistic multi-household population."""
        households = []
        individuals = []
        
        # Household 1: Nuclear family
        hh1 = Household(household_id=1, size=4)
        households.append(hh1)
        for i, age in enumerate([45, 43, 15, 12]):
            agent = Agent(agent_id=i+1, age=age, sex='male' if i % 2 == 0 else 'female')
            agent.household_id = 1
            individuals.append(agent)
        
        # Household 2: Single person
        hh2 = Household(household_id=2, size=1)
        households.append(hh2)
        agent = Agent(agent_id=5, age=30, sex='female', hh_role='single')
        agent.household_id = 2
        individuals.append(agent)
        
        # Household 3: Elderly couple
        hh3 = Household(household_id=3, size=2)
        households.append(hh3)
        for i, age in enumerate([72, 69]):
            agent = Agent(agent_id=6+i, age=age, sex='male' if i == 0 else 'female')
            agent.household_id = 3
            individuals.append(agent)
        
        result = run_all_checks(households, individuals)
        
        assert result.total_households == 3
        assert result.total_individuals == 7
        assert result.is_valid is True
        assert result.critical_count == 0
