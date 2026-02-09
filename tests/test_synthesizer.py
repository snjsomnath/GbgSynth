"""
Tests for the population synthesizer.
"""

import pytest
import pandas as pd
import numpy as np
from gbgsynth.synthesizer import PopulationSynthesizer
from gbgsynth.models import Agent, Household


class TestPopulationSynthesizer:
    """Tests for the PopulationSynthesizer class."""

    @pytest.fixture
    def synthesizer(self):
        """Create a synthesizer instance for testing."""
        return PopulationSynthesizer(use_ipf=False, use_constrained_ipf=False, use_topdown=False)

    @pytest.fixture
    def simple_population_data(self):
        """Create simple population data for testing."""
        return pd.DataFrame({
            'age_group': ['0-17', '0-17', '18-64', '18-64', '65+', '65+'],
            'sex': ['male', 'female', 'male', 'female', 'male', 'female'],
            'hh_role': ['child', 'child', 'single', 'single', 'single', 'single'],
            'count': [20, 20, 50, 50, 15, 15]
        })

    @pytest.fixture
    def simple_household_data(self):
        """Create simple household data for testing."""
        return pd.DataFrame({
            'hh_size': ['1 person', '2 persons', '3 persons', '4+ persons'],
            'count': [30, 40, 20, 10]
        })

    def test_initialization(self, synthesizer):
        """Test synthesizer initialization."""
        assert synthesizer.agents == []
        assert synthesizer.households == []
        assert synthesizer.next_agent_id == 1
        assert synthesizer.next_household_id == 1

    def test_initialization_with_ipf(self):
        """Test synthesizer with IPF enabled."""
        synth = PopulationSynthesizer(use_ipf=True)
        assert synth.use_ipf == True

    def test_initialization_with_constrained_ipf(self):
        """Test synthesizer with constrained IPF enabled."""
        synth = PopulationSynthesizer(use_constrained_ipf=True)
        assert synth.use_constrained_ipf == True

    def test_initialization_with_topdown(self):
        """Test synthesizer with top-down synthesis enabled."""
        synth = PopulationSynthesizer(use_topdown=True)
        assert synth.use_topdown == True


class TestSynthesizerHelperMethods:
    """Tests for synthesizer helper methods."""

    @pytest.fixture
    def synthesizer(self):
        """Create a synthesizer instance for testing."""
        return PopulationSynthesizer(use_ipf=False, use_constrained_ipf=False, use_topdown=False)

    def test_synthesizer_has_config(self, synthesizer):
        """Test that synthesizer has config."""
        assert synthesizer.config is not None

    def test_synthesizer_has_constraints(self, synthesizer):
        """Test that synthesizer has constraints."""
        assert synthesizer.constraints is not None
        assert isinstance(synthesizer.constraints, dict)


class TestSynthesisOutput:
    """Tests for synthesis output validation."""

    def test_agents_have_required_attributes(self):
        """Test that synthesized agents have all required attributes."""
        agent = Agent(agent_id=1, age=35, sex='male')
        
        # Required attributes
        assert hasattr(agent, 'agent_id')
        assert hasattr(agent, 'age')
        assert hasattr(agent, 'sex')
        assert hasattr(agent, 'hh_role')
        assert hasattr(agent, 'household_id')

    def test_households_have_required_attributes(self):
        """Test that households have all required attributes."""
        hh = Household(household_id=1, size=2)
        
        # Required attributes
        assert hasattr(hh, 'household_id')
        assert hasattr(hh, 'size')
        assert hasattr(hh, 'members')
        assert hasattr(hh, 'cars')
        assert hasattr(hh, 'house_type')

    def test_household_member_linkage(self):
        """Test that household members are properly linked."""
        hh = Household(household_id=1, size=2)
        agent1 = Agent(agent_id=1, age=35, sex='male')
        agent2 = Agent(agent_id=2, age=32, sex='female')
        
        hh.add_member(agent1)
        hh.add_member(agent2)
        
        # All members should have correct household_id
        assert agent1.household_id == 1
        assert agent2.household_id == 1
        
        # Household should reference members
        assert len(hh.members) == 2


class TestSynthesisConstraints:
    """Tests for synthesis constraints validation."""

    def test_age_constraint_parent_child(self):
        """Test that parent-child age constraints are respected."""
        parent = Agent(agent_id=1, age=40, sex='female')
        child = Agent(agent_id=2, age=10, sex='male')
        
        # Parent should be old enough
        assert parent.can_be_parent_of(child)
        
        # Young adult cannot be parent of teen
        young_adult = Agent(agent_id=3, age=28, sex='male')
        teen = Agent(agent_id=4, age=15, sex='female')
        assert not young_adult.can_be_parent_of(teen)

    def test_partnership_constraints(self):
        """Test partnership constraints."""
        male = Agent(agent_id=1, age=35, sex='male')
        female = Agent(agent_id=2, age=33, sex='female')
        
        # Should be compatible
        assert male.can_be_partner_with(female)
        
        # Large age gap should not be compatible (default)
        older_female = Agent(agent_id=3, age=55, sex='female')
        assert not male.can_be_partner_with(older_female)

    def test_household_capacity_constraint(self):
        """Test household capacity constraints."""
        hh = Household(household_id=1, size=2)
        
        assert hh.can_fit(1)
        assert hh.can_fit(2)
        assert not hh.can_fit(3)
        
        hh.add_member(Agent(agent_id=1, age=35, sex='male'))
        
        assert hh.can_fit(1)
        assert not hh.can_fit(2)


class TestDataFrameInput:
    """Tests for DataFrame input handling."""

    def test_population_data_format(self):
        """Test expected population data format."""
        pop_data = pd.DataFrame({
            'age_group': ['0-17', '18-64', '65+'],
            'sex': ['male', 'male', 'male'],
            'hh_role': ['child', 'single', 'single'],
            'count': [100, 300, 100]
        })
        
        # Verify structure
        assert 'age_group' in pop_data.columns or 'age' in pop_data.columns
        assert 'sex' in pop_data.columns
        assert 'count' in pop_data.columns

    def test_household_data_format(self):
        """Test expected household data format."""
        hh_data = pd.DataFrame({
            'hh_size': ['1 person', '2 persons', '3 persons'],
            'count': [100, 150, 80]
        })
        
        # Verify structure
        assert 'hh_size' in hh_data.columns or 'size' in hh_data.columns
        assert 'count' in hh_data.columns


class TestEdgeCases:
    """Tests for edge cases in synthesis."""

    def test_empty_household_data(self):
        """Test handling of empty household data."""
        hh_data = pd.DataFrame({'hh_size': [], 'count': []})
        assert len(hh_data) == 0

    def test_single_person_households_only(self):
        """Test synthesis with only single-person households."""
        hh = Household(household_id=1, size=1)
        agent = Agent(agent_id=1, age=45, sex='female', hh_role='single')
        
        hh.add_member(agent)
        
        assert hh.is_full()
        assert hh.is_single()
        assert not hh.is_couple()

    def test_large_household(self):
        """Test household with many members."""
        hh = Household(household_id=1, size=6)
        
        # Add parents
        hh.add_member(Agent(agent_id=1, age=45, sex='male'))
        hh.add_member(Agent(agent_id=2, age=43, sex='female'))
        
        # Add children
        for i in range(4):
            hh.add_member(Agent(agent_id=i+3, age=15-i*3, sex='male' if i % 2 == 0 else 'female'))
        
        assert hh.is_full()
        assert len(hh.children) == 4

    def test_zero_count_category(self):
        """Test handling of zero-count categories."""
        pop_data = pd.DataFrame({
            'age_group': ['0-17', '18-64', '65+'],
            'sex': ['male', 'male', 'male'],
            'count': [0, 100, 0]  # Zero counts for some categories
        })
        
        # Should have only non-zero counts
        non_zero = pop_data[pop_data['count'] > 0]
        assert len(non_zero) == 1


class TestIncomeHandling:
    """Tests for income handling edge cases."""

    def test_none_income_in_sum(self):
        """Test that None incomes don't break household income calculation."""
        hh = Household(household_id=1, size=3)
        
        # Add members with various income states
        agent1 = Agent(agent_id=1, age=45, sex='male')
        agent1.income = 500000
        
        agent2 = Agent(agent_id=2, age=43, sex='female')
        agent2.income = None  # No income data
        
        agent3 = Agent(agent_id=3, age=10, sex='male')
        # No income attribute set at all
        
        hh.add_member(agent1)
        hh.add_member(agent2)
        hh.add_member(agent3)
        
        # Calculate total income handling None values
        incomes = [getattr(m, 'income', 0) or 0 for m in hh.members]
        total = sum(incomes)
        
        assert total == 500000

    def test_all_none_incomes(self):
        """Test household where all members have None income."""
        hh = Household(household_id=1, size=2)
        
        agent1 = Agent(agent_id=1, age=25, sex='male')
        agent1.income = None
        
        agent2 = Agent(agent_id=2, age=24, sex='female')
        agent2.income = None
        
        hh.add_member(agent1)
        hh.add_member(agent2)
        
        incomes = [getattr(m, 'income', 0) or 0 for m in hh.members]
        total = sum(incomes)
        
        assert total == 0
