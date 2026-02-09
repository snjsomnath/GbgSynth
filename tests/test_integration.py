"""
Integration tests for the GbgSynth package.

These tests verify that components work together correctly.
"""

import pytest
import pandas as pd
import numpy as np
from gbgsynth.models import Agent, Household
from gbgsynth.synthesizer import PopulationSynthesizer
from gbgsynth.ipf import IPFSynthesizer
from gbgsynth.config import Config
from gbgsynth import GbgSynth, GbgArea


class TestEndToEndSynthesis:
    """End-to-end tests for full synthesis workflow."""

    @pytest.fixture
    def city(self):
        """Create a GbgSynth instance."""
        return GbgSynth(year=2023)

    def test_synthesize_single_area(self, city):
        """Test synthesizing a single area end-to-end."""
        # Use a small area for faster testing
        area = city.synthesize("Haga")
        
        # Verify basic outputs
        assert area._is_generated
        assert len(area.individuals) > 0
        assert len(area.households) > 0
        
        # Verify relationships
        for ind in area.individuals:
            assert ind.household_id is not None
        
        # Verify all agents are in households
        agent_ids = {a.agent_id for a in area.individuals}
        for hh in area.households:
            for member in hh.members:
                assert member.agent_id in agent_ids

    def test_synthesis_statistics(self, city):
        """Test that synthesis produces valid statistics."""
        area = city.synthesize("Haga")
        stats = area.get_summary_statistics()
        
        # Verify statistics are computed correctly
        assert stats['total_population'] == len(area.individuals)
        assert stats['total_households'] == len(area.households)
        assert stats['avg_household_size'] > 0
        assert stats['num_children'] >= 0
        assert stats['num_adults'] >= 0
        assert stats['num_children'] + stats['num_adults'] == stats['total_population']

    def test_marginal_comparison(self, city):
        """Test that marginal comparison works correctly."""
        area = city.synthesize("Haga")
        comparison = area.compare_to_marginals(print_report=False)
        
        # Verify comparison structure
        assert 'overall' in comparison
        assert 'sex' in comparison
        assert 'age' in comparison
        
        # Verify overall statistics
        overall = comparison['overall']
        assert 'correlation' in overall
        assert 'rmse' in overall
        assert 'mae' in overall
        
        # Correlation should be high for good synthesis
        assert overall['correlation'] > 0.9

    def test_dataframe_export(self, city):
        """Test that DataFrame export works correctly."""
        area = city.synthesize("Haga")
        
        # Get DataFrames
        ind_df = area.individuals_df
        hh_df = area.households_df
        
        # Verify DataFrame structure
        assert len(ind_df) == len(area.individuals)
        assert len(hh_df) == len(area.households)
        
        # Verify required columns
        assert 'age' in ind_df.columns
        assert 'sex' in ind_df.columns
        assert 'household_id' in ind_df.columns
        
        assert 'household_id' in hh_df.columns
        assert 'size' in hh_df.columns

    def test_list_areas(self, city):
        """Test that listing areas works."""
        areas = city.list_areas()
        assert len(areas) > 0
        assert 'Haga' in areas

    def test_get_area_by_code(self, city):
        """Test getting area by code."""
        area = city.get_area("107")
        assert area.area_code == "107"
        assert "Haga" in area.area_name

    def test_get_area_by_name(self, city):
        """Test getting area by name."""
        area = city.get_area("Haga")
        assert area.area_code == "107"


class TestModelsIntegration:
    """Integration tests for model classes."""

    def test_complete_family_workflow(self):
        """Test creating a complete family through the standard workflow."""
        # Create household
        hh = Household(household_id=1, size=4)
        
        # Create family members
        father = Agent(agent_id=1, age=42, sex='male', status='employed', income=700000)
        mother = Agent(agent_id=2, age=40, sex='female', status='employed', income=550000)
        teen = Agent(agent_id=3, age=16, sex='male')
        child = Agent(agent_id=4, age=10, sex='female')
        
        # Add to household
        hh.add_member(father)
        hh.add_member(mother)
        hh.add_member(teen)
        hh.add_member(child)
        
        # Verify household structure
        assert hh.is_full()
        assert hh.is_couple()
        assert len(hh.children) == 2
        
        # Verify linkages
        assert all(m.household_id == 1 for m in hh.members)
        
        # Verify relationships
        assert father.can_be_partner_with(mother)
        assert father.can_be_parent_of(teen)
        assert mother.can_be_parent_of(child)

    def test_multiple_households(self):
        """Test creating and managing multiple households."""
        households = []
        agents = []
        agent_id = 1
        
        # Create 5 households of varying sizes
        for hh_id in range(1, 6):
            size = min(hh_id, 4)  # Sizes 1-4
            hh = Household(household_id=hh_id, size=size)
            
            # Add head
            head = Agent(agent_id=agent_id, age=35 + hh_id, sex='male')
            hh.add_member(head)
            agents.append(head)
            agent_id += 1
            
            # Add additional members
            for i in range(1, size):
                if i == 1 and size > 1:
                    # Partner
                    member = Agent(agent_id=agent_id, age=33 + hh_id, sex='female')
                else:
                    # Child
                    member = Agent(agent_id=agent_id, age=15 - i*3, sex='male' if i % 2 == 0 else 'female')
                hh.add_member(member)
                agents.append(member)
                agent_id += 1
            
            households.append(hh)
        
        # Verify
        assert len(households) == 5
        assert all(hh.is_full() for hh in households)
        
        # Count total individuals
        total_individuals = sum(len(hh.members) for hh in households)
        assert total_individuals == len(agents)


class TestIPFIntegration:
    """Integration tests for IPF with synthesis."""

    def test_ipf_produces_valid_distribution(self):
        """Test that IPF produces a valid probability distribution."""
        ipf = IPFSynthesizer()
        
        marginals = {
            'age': pd.Series([100, 300, 100], index=['young', 'middle', 'old']),
            'sex': pd.Series([250, 250], index=['male', 'female'])
        }
        
        weights = ipf.fit(marginals)
        
        # Should be valid weights
        assert np.all(weights >= 0)
        assert np.isclose(weights.sum(), 500, rtol=0.01)
        
        # Marginals should match
        assert np.allclose(weights.sum(axis=1), [100, 300, 100], rtol=0.01)
        assert np.allclose(weights.sum(axis=0), [250, 250], rtol=0.01)

    def test_ipf_with_3d_marginals(self):
        """Test IPF with 3-dimensional marginals."""
        ipf = IPFSynthesizer()
        
        marginals = {
            'age': pd.Series([100, 200, 100], index=['young', 'middle', 'old']),
            'sex': pd.Series([200, 200], index=['male', 'female']),
            'role': pd.Series([150, 200, 50], index=['single', 'couple', 'child'])
        }
        
        weights = ipf.fit(marginals)
        
        # Should have correct shape
        assert weights.shape == (3, 2, 3)


class TestConfigIntegration:
    """Integration tests for configuration."""

    def test_config_loads_for_synthesizer(self):
        """Test that config works with synthesizer."""
        synth = PopulationSynthesizer()
        
        assert synth.config is not None
        assert synth.constraints is not None

    def test_config_constraints_used(self):
        """Test that config constraints are accessible."""
        config = Config()
        constraints = config.constraints
        
        # Constraints should be usable
        assert isinstance(constraints, dict)


class TestDataFlowIntegration:
    """Integration tests for data flow through the system."""

    def test_agent_to_household_linkage(self):
        """Test proper agent-to-household data linkage."""
        hh = Household(household_id=42, size=2)
        agent1 = Agent(agent_id=1, age=30, sex='male')
        agent2 = Agent(agent_id=2, age=28, sex='female')
        
        hh.add_member(agent1)
        hh.add_member(agent2)
        
        # Verify bidirectional linkage
        assert agent1.household_id == hh.household_id
        assert agent2.household_id == hh.household_id
        assert agent1 in hh.members
        assert agent2 in hh.members

    def test_household_serialization_roundtrip(self):
        """Test that household data can be serialized and contains correct info."""
        hh = Household(household_id=1, size=3, house_type='apartment', cars=1)
        
        father = Agent(agent_id=1, age=40, sex='male', income=600000)
        mother = Agent(agent_id=2, age=38, sex='female', income=400000)
        child = Agent(agent_id=3, age=10, sex='male')
        
        hh.add_member(father)
        hh.add_member(mother)
        hh.add_member(child)
        
        # Serialize
        data = hh.to_dict()
        
        # Verify data completeness
        assert data['household_id'] == 1
        assert data['size'] == 3
        assert data['actual_size'] == 3
        assert data['is_couple'] == True
        assert data['total_income'] == 1000000

    def test_agent_serialization_roundtrip(self):
        """Test that agent data can be serialized correctly."""
        agent = Agent(
            agent_id=1,
            age=35,
            sex='male',
            hh_role='cohabiting',
            status='employed',
            income=550000,
            income_decile=7,
            education='university',
            household_id=42
        )
        
        data = agent.to_dict()
        
        # Verify all fields present
        assert data['agent_id'] == 1
        assert data['age'] == 35
        assert data['sex'] == 'male'
        assert data['income'] == 550000
        assert data['household_id'] == 42


class TestHouseholdTypeIntegration:
    """Integration tests for different household types."""

    def test_single_person_household_complete(self, single_person_household):
        """Test single person household is complete and valid."""
        hh = single_person_household
        
        assert hh.is_full()
        assert hh.is_single()
        assert not hh.is_couple()
        assert not hh.is_single_parent()
        assert len(hh.children) == 0

    def test_couple_household_complete(self, couple_household):
        """Test couple household is complete and valid."""
        hh = couple_household
        
        assert hh.is_full()
        assert hh.is_couple()
        assert not hh.is_single()
        assert not hh.is_single_parent()
        assert len(hh.children) == 0

    def test_single_parent_household_complete(self, single_parent_household):
        """Test single parent household is complete and valid."""
        hh = single_parent_household
        
        assert hh.is_full()
        assert hh.is_single_parent()
        assert not hh.is_couple()
        assert not hh.is_single()
        assert len(hh.children) == 2


class TestEdgeCasesIntegration:
    """Integration tests for edge cases."""

    def test_elderly_single_household(self):
        """Test elderly person living alone."""
        hh = Household(household_id=1, size=1)
        agent = Agent(agent_id=1, age=85, sex='female')
        
        hh.add_member(agent)
        
        assert hh.is_full()
        assert agent.status == 'retired'
        assert hh.is_single()

    def test_young_adult_single(self):
        """Test young adult living alone."""
        hh = Household(household_id=1, size=1)
        agent = Agent(agent_id=1, age=22, sex='male', status='student')
        
        hh.add_member(agent)
        
        assert hh.is_full()
        assert agent.status == 'student'

    def test_large_family(self):
        """Test large family household."""
        hh = Household(household_id=1, size=6)
        
        # Parents
        hh.add_member(Agent(agent_id=1, age=50, sex='male', income=800000))
        hh.add_member(Agent(agent_id=2, age=48, sex='female', income=600000))
        
        # Children of varying ages
        hh.add_member(Agent(agent_id=3, age=17, sex='male'))
        hh.add_member(Agent(agent_id=4, age=14, sex='female'))
        hh.add_member(Agent(agent_id=5, age=11, sex='male'))
        hh.add_member(Agent(agent_id=6, age=8, sex='female'))
        
        assert hh.is_full()
        assert hh.is_couple()
        assert len(hh.children) == 4
        assert hh.income == 1400000
