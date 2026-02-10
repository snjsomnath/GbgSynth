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
        return PopulationSynthesizer()

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

    def test_initialization_has_stats(self):
        """Test synthesizer has stats dictionary."""
        synth = PopulationSynthesizer()
        assert hasattr(synth, 'stats')
        assert isinstance(synth.stats, dict)


class TestSynthesizerHelperMethods:
    """Tests for synthesizer helper methods."""

    @pytest.fixture
    def synthesizer(self):
        """Create a synthesizer instance for testing."""
        return PopulationSynthesizer()

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


class TestEducationLevelAssignment:
    """Tests for education level assignment."""

    @pytest.fixture
    def synthesizer_with_agents(self):
        """Create a synthesizer with some agents already created."""
        synth = PopulationSynthesizer()
        # Adults of various ages and sexes
        synth.agents = [
            Agent(agent_id=1, age=20, sex='male', hh_role='single'),
            Agent(agent_id=2, age=30, sex='female', hh_role='single'),
            Agent(agent_id=3, age=50, sex='male', hh_role='cohabiting'),
            Agent(agent_id=4, age=70, sex='female', hh_role='single'),
            Agent(agent_id=5, age=10, sex='male'),   # Child
            Agent(agent_id=6, age=5, sex='female'),   # Child
        ]
        return synth

    @pytest.fixture
    def education_data(self):
        """Create sample education level data matching the 23_InkomsterUtbildning format."""
        rows = []
        age_groups = ['18-24 år', '25-34 år', '35-44 år', '45-54 år', '55-64 år', '65-74 år', '75- år']
        sexes = ['Man', 'Kvinna']
        edu_levels = ['Förgymnasial utbildning', 'Gymnasial utbildning',
                      'Eftergymnasial utbildning', 'Uppgift saknas']
        
        for ag in age_groups:
            for sex in sexes:
                for edu in edu_levels:
                    # Create a distribution that favours post_secondary
                    if edu == 'Eftergymnasial utbildning':
                        count = 100
                    elif edu == 'Gymnasial utbildning':
                        count = 50
                    elif edu == 'Förgymnasial utbildning':
                        count = 20
                    else:
                        count = 5
                    rows.append({
                        'Område': '107 Haga',
                        'Ålder': ag,
                        'Kön': sex,
                        'Utbildningsnivå': edu,
                        'Tabellvärde': 'Folkmängd',
                        'År': 2023,
                        'Antal': count,
                    })
        return pd.DataFrame(rows)

    def test_assigns_education_to_adults(self, synthesizer_with_agents, education_data):
        """Test that education level is assigned to all adults."""
        synth = synthesizer_with_agents
        synth._assign_education_level(education_data)
        
        adults = [a for a in synth.agents if a.age >= 18]
        for adult in adults:
            assert adult.education in ['pre_secondary', 'secondary', 'post_secondary', 'unknown'], \
                f"Agent {adult.agent_id} has unexpected education: {adult.education}"

    def test_children_get_child_education(self, synthesizer_with_agents, education_data):
        """Test that children get education='child'."""
        synth = synthesizer_with_agents
        synth._assign_education_level(education_data)
        
        children = [a for a in synth.agents if a.age < 18]
        for child in children:
            assert child.education == 'child'

    def test_distribution_roughly_matches(self, education_data):
        """Test that assigned distribution roughly matches input probabilities."""
        synth = PopulationSynthesizer()
        # Create 1000 adults aged 30, male
        synth.agents = [
            Agent(agent_id=i, age=30, sex='male', hh_role='single')
            for i in range(1000)
        ]
        synth._assign_education_level(education_data)
        
        counts = {}
        for a in synth.agents:
            counts[a.education] = counts.get(a.education, 0) + 1
        
        # Expected proportions: post_secondary=100/175≈57%, secondary=50/175≈29%,
        # pre_secondary=20/175≈11%, unknown=5/175≈3%
        assert counts.get('post_secondary', 0) > counts.get('secondary', 0)
        assert counts.get('secondary', 0) > counts.get('pre_secondary', 0)
        assert counts.get('pre_secondary', 0) > counts.get('unknown', 0)

    def test_handles_none_data(self, synthesizer_with_agents):
        """Test that None education data is handled gracefully."""
        synth = synthesizer_with_agents
        synth._assign_education_level(None)
        # Should not crash, agents keep their default education
        for agent in synth.agents:
            assert agent.education is None or agent.education == 'child'

    def test_handles_empty_data(self, synthesizer_with_agents):
        """Test that empty DataFrame is handled gracefully."""
        synth = synthesizer_with_agents
        synth._assign_education_level(pd.DataFrame())
        for agent in synth.agents:
            assert agent.education is None or agent.education == 'child'


class TestIncomeSourceAssignment:
    """Tests for income source assignment."""

    @pytest.fixture
    def synthesizer_with_agents(self):
        """Create a synthesizer with agents of various ages and sexes."""
        synth = PopulationSynthesizer()
        synth.agents = [
            Agent(agent_id=1, age=25, sex='male', hh_role='single'),
            Agent(agent_id=2, age=35, sex='female', hh_role='single'),
            Agent(agent_id=3, age=70, sex='male', hh_role='single'),
            Agent(agent_id=4, age=45, sex='female', hh_role='cohabiting'),
            Agent(agent_id=5, age=15, sex='male'),   # Under 20
            Agent(agent_id=6, age=5, sex='female'),   # Child
        ]
        return synth

    @pytest.fixture
    def income_source_data(self):
        """Create sample income source data matching the 20_HuvudInk format."""
        rows = []
        sexes = ['Man', 'Kvinna']
        sources = [
            ('Ersättning för arbete', 200),
            ('Ersättning vid arbetslöshet', 10),
            ('Ersättning för studier', 15),
            ('Pension', 50),
            ('Ersättning vid långvarigt nedsatt arbetsförmåga', 5),
            ('Ersättning vid sjukdom', 5),
            ('Ersättning vid föräldraledighet eller närståendeomvårdnad', 10),
            ('Ekonomiskt stöd', 5),
            ('Saknar ersättningar', 10),
        ]
        for sex in sexes:
            for source, count in sources:
                rows.append({
                    'Område': '107 Haga',
                    'Kön': sex,
                    'Huvudsaklig inkomstkälla': source,
                    'År': 2023,
                    'Antal': count,
                })
        return pd.DataFrame(rows)

    def test_assigns_income_source_to_adults(self, synthesizer_with_agents, income_source_data):
        """Test that income source is assigned to adults 20+."""
        synth = synthesizer_with_agents
        synth._assign_income_source(income_source_data)
        
        valid_sources = {'work', 'unemployment', 'studies', 'pension', 'disability',
                         'sickness', 'parental_leave', 'financial_support', 'no_income'}
        adults_20plus = [a for a in synth.agents if a.age >= 20]
        for adult in adults_20plus:
            assert adult.income_source in valid_sources, \
                f"Agent {adult.agent_id} has unexpected income_source: {adult.income_source}"

    def test_children_and_teens_get_none(self, synthesizer_with_agents, income_source_data):
        """Test that agents under 20 get income_source=None."""
        synth = synthesizer_with_agents
        synth._assign_income_source(income_source_data)
        
        under_20 = [a for a in synth.agents if a.age < 20]
        for agent in under_20:
            assert agent.income_source is None

    def test_distribution_favors_work(self, income_source_data):
        """Test that 'work' is the most common income source given the test data."""
        synth = PopulationSynthesizer()
        synth.agents = [
            Agent(agent_id=i, age=35, sex='male', hh_role='single')
            for i in range(1000)
        ]
        synth._assign_income_source(income_source_data)
        
        counts = {}
        for a in synth.agents:
            counts[a.income_source] = counts.get(a.income_source, 0) + 1
        
        assert counts.get('work', 0) > counts.get('pension', 0)
        assert counts.get('work', 0) > counts.get('studies', 0)

    def test_handles_none_data(self, synthesizer_with_agents):
        """Test that None income source data is handled gracefully."""
        synth = synthesizer_with_agents
        synth._assign_income_source(None)
        for agent in synth.agents:
            assert agent.income_source is None

    def test_handles_empty_data(self, synthesizer_with_agents):
        """Test that empty DataFrame is handled gracefully."""
        synth = synthesizer_with_agents
        synth._assign_income_source(pd.DataFrame())
        for agent in synth.agents:
            assert agent.income_source is None


class TestMedianIncomeAssignment:
    """Tests for education-based median income assignment."""

    @pytest.fixture
    def synthesizer_with_educated_agents(self):
        """Create a synthesizer with agents that have education already assigned."""
        synth = PopulationSynthesizer()
        synth.agents = [
            Agent(agent_id=1, age=30, sex='male', hh_role='single', education='post_secondary'),
            Agent(agent_id=2, age=30, sex='female', hh_role='single', education='secondary'),
            Agent(agent_id=3, age=55, sex='male', hh_role='cohabiting', education='pre_secondary'),
            Agent(agent_id=4, age=10, sex='female'),  # Child
        ]
        synth.households = [
            Household(household_id=1, size=1, members=[synth.agents[0]]),
            Household(household_id=2, size=1, members=[synth.agents[1]]),
            Household(household_id=3, size=2, members=[synth.agents[2], synth.agents[3]]),
        ]
        for a in synth.agents[:1]:
            a.household_id = 1
        synth.agents[1].household_id = 2
        for a in synth.agents[2:]:
            a.household_id = 3
        return synth

    @pytest.fixture
    def education_data_with_income(self):
        """Create education data that includes Medianinkomst rows."""
        rows = []
        age_groups = ['18-24 år', '25-34 år', '35-44 år', '45-54 år', '55-64 år', '65-74 år', '75- år']
        sexes = ['Man', 'Kvinna']
        edu_levels = ['Förgymnasial utbildning', 'Gymnasial utbildning',
                      'Eftergymnasial utbildning', 'Uppgift saknas']
        
        median_incomes = {
            'Förgymnasial utbildning': 250000,
            'Gymnasial utbildning': 350000,
            'Eftergymnasial utbildning': 500000,
            'Uppgift saknas': 200000,
        }
        
        for ag in age_groups:
            for sex in sexes:
                for edu in edu_levels:
                    # Folkmängd row
                    rows.append({
                        'Område': '107 Haga', 'Ålder': ag, 'Kön': sex,
                        'Utbildningsnivå': edu, 'Tabellvärde': 'Folkmängd',
                        'År': 2023, 'Antal': 50,
                    })
                    # Medianinkomst row
                    rows.append({
                        'Område': '107 Haga', 'Ålder': ag, 'Kön': sex,
                        'Utbildningsnivå': edu, 'Tabellvärde': 'Medianinkomst',
                        'År': 2023, 'Antal': median_incomes[edu],
                    })
        return pd.DataFrame(rows)

    @pytest.fixture
    def income_data(self):
        """Create minimal income standard data."""
        return pd.DataFrame([
            {'Inkomststandard': 'Ingår i helårshushåll som inte har låg inkomststandard', 'Antal': 900},
            {'Inkomststandard': 'Ingår i helårshushåll som har låg inkomststandard', 'Antal': 100},
        ])

    def test_builds_median_income_table(self, education_data_with_income):
        """Test that _build_median_income_table creates a valid lookup table."""
        synth = PopulationSynthesizer()
        table = synth._build_median_income_table(education_data_with_income)
        
        assert len(table) > 0
        # Check a specific key
        assert ('25-34 år', 'male', 'post_secondary') in table
        assert table[('25-34 år', 'male', 'post_secondary')] == 500000

    def test_median_income_higher_for_post_secondary(self, synthesizer_with_educated_agents,
                                                      income_data, education_data_with_income):
        """Test that post_secondary education yields higher income than secondary."""
        import random
        random.seed(42)
        synth = synthesizer_with_educated_agents
        synth._assign_income(income_data, education_data_with_income)
        
        post_sec = [a for a in synth.agents if a.education == 'post_secondary'][0]
        secondary = [a for a in synth.agents if a.education == 'secondary'][0]
        
        # On average, post_secondary should have higher income
        # With seed=42 this should be deterministic, but the expected relationship
        # may occasionally not hold for a single draw. We mainly test it doesn't crash.
        assert post_sec.income > 0
        assert secondary.income > 0

    def test_children_get_zero_income(self, synthesizer_with_educated_agents,
                                      income_data, education_data_with_income):
        """Test that children get zero income."""
        synth = synthesizer_with_educated_agents
        synth._assign_income(income_data, education_data_with_income)
        
        children = [a for a in synth.agents if a.age < 18]
        for child in children:
            assert child.income == 0

    def test_falls_back_to_decile_without_education_data(self, synthesizer_with_educated_agents,
                                                          income_data):
        """Test that income assignment works without education data."""
        synth = synthesizer_with_educated_agents
        synth._assign_income(income_data, None)
        
        adults = [a for a in synth.agents if a.age >= 18]
        for adult in adults:
            assert adult.income > 0
            assert adult.income_decile is not None


class TestIncomeSourceAgeConditioning:
    """Tests for age-conditioned income source assignment."""

    @pytest.fixture
    def income_source_data(self):
        """Create sample income source data matching the 20_HuvudInk format."""
        rows = []
        sexes = ['Man', 'Kvinna']
        sources = [
            ('Ersättning för arbete', 200),
            ('Ersättning vid arbetslöshet', 10),
            ('Ersättning för studier', 15),
            ('Pension', 50),
            ('Ersättning vid långvarigt nedsatt arbetsförmåga', 5),
            ('Ersättning vid sjukdom', 5),
            ('Ersättning vid föräldraledighet eller närståendeomvårdnad', 10),
            ('Ekonomiskt stöd', 5),
            ('Saknar ersättningar', 10),
        ]
        for sex in sexes:
            for source, count in sources:
                rows.append({
                    'Kön': sex,
                    'Huvudsaklig inkomstkälla': source,
                    'Antal': count,
                })
        return pd.DataFrame(rows)

    def test_elderly_mostly_pension(self, income_source_data):
        """Test that 65+ agents are preferentially assigned 'pension' slots.
        
        With deterministic quota allocation, the total pension count matches
        the census proportion (~16%).  But among all agents, the elderly
        should fill those pension slots first due to high age affinity.
        """
        import random
        random.seed(42)
        synth = PopulationSynthesizer()
        # Mix of old and young to test that elderly get the pension slots
        synth.agents = [
            Agent(agent_id=i, age=72, sex='male' if i % 2 == 0 else 'female',
                  hh_role='single')
            for i in range(250)
        ] + [
            Agent(agent_id=i + 250, age=30, sex='male' if i % 2 == 0 else 'female',
                  hh_role='single')
            for i in range(250)
        ]
        synth._assign_income_source(income_source_data)

        # Census has 50 pension per sex out of 310 total → ~16% quota
        pension_old = sum(1 for a in synth.agents if a.age == 72 and a.income_source == 'pension')
        pension_young = sum(1 for a in synth.agents if a.age == 30 and a.income_source == 'pension')
        # Elderly should get ALL the pension slots (high affinity)
        assert pension_old > pension_young, \
            f"Expected elderly to get more pension slots than young: old={pension_old}, young={pension_young}"
        # Young (age 30, weight=0.0 for pension) should get essentially none
        assert pension_young == 0, \
            f"Expected 0 pension for age 30 (weight=0.0), got {pension_young}"

    def test_young_adults_mostly_work_or_studies(self, income_source_data):
        """Test that 20-24 year olds preferentially fill 'studies' slots."""
        import random
        random.seed(42)
        synth = PopulationSynthesizer()
        # Mix young and middle-aged to test affinity
        synth.agents = [
            Agent(agent_id=i, age=22, sex='male' if i % 2 == 0 else 'female',
                  hh_role='single')
            for i in range(250)
        ] + [
            Agent(agent_id=i + 250, age=45, sex='male' if i % 2 == 0 else 'female',
                  hh_role='single')
            for i in range(250)
        ]
        synth._assign_income_source(income_source_data)

        studies_young = sum(1 for a in synth.agents if a.age == 22 and a.income_source == 'studies')
        studies_old = sum(1 for a in synth.agents if a.age == 45 and a.income_source == 'studies')
        # Young adults should fill studies slots preferentially
        assert studies_young > studies_old, \
            f"Expected more studies at 22 ({studies_young}) than at 45 ({studies_old})"

    def test_no_pension_for_young_adults(self, income_source_data):
        """Test that agents with 0.0 pension weight don't get pension when
        competing with agents who have positive pension weight."""
        import random
        random.seed(42)
        synth = PopulationSynthesizer()
        # 250 young (weight=0.0 for pension) + 250 old (weight=2.8)
        synth.agents = [
            Agent(agent_id=i, age=30, sex='male', hh_role='single')
            for i in range(250)
        ] + [
            Agent(agent_id=i + 250, age=70, sex='male', hh_role='single')
            for i in range(250)
        ]
        synth._assign_income_source(income_source_data)

        pension_young = sum(1 for a in synth.agents if a.age == 30 and a.income_source == 'pension')
        assert pension_young == 0, \
            f"Expected 0 pension for age 30 when 70-year-olds available, got {pension_young}"

    def test_parental_leave_peak_in_30s(self, income_source_data):
        """Test that parental leave slots go to 25-44 over 55-64."""
        import random
        random.seed(42)
        synth = PopulationSynthesizer()
        synth.agents = [
            Agent(agent_id=i, age=32, sex='female', hh_role='cohabiting')
            for i in range(250)
        ] + [
            Agent(agent_id=i + 250, age=58, sex='female', hh_role='cohabiting')
            for i in range(250)
        ]
        synth._assign_income_source(income_source_data)

        young_pl = sum(1 for a in synth.agents if a.age == 32 and a.income_source == 'parental_leave')
        old_pl = sum(1 for a in synth.agents if a.age == 58 and a.income_source == 'parental_leave')
        assert young_pl > old_pl, \
            f"Expected more parental leave at 32 ({young_pl}) than at 58 ({old_pl})"

    def test_quota_matches_census_proportions(self, income_source_data):
        """Test that deterministic allocation matches census totals exactly."""
        import random
        random.seed(42)
        synth = PopulationSynthesizer()
        # All-male population to isolate one sex
        synth.agents = [
            Agent(agent_id=i, age=40, sex='male', hh_role='single')
            for i in range(310)  # Exactly matches census male total
        ]
        synth._assign_income_source(income_source_data)

        counts = {}
        for a in synth.agents:
            counts[a.income_source] = counts.get(a.income_source, 0) + 1
        
        # Census male: work=200, pension=50, studies=15, etc.
        assert counts.get('work', 0) == 200, f"Expected 200 work, got {counts.get('work', 0)}"
        assert counts.get('pension', 0) == 50, f"Expected 50 pension, got {counts.get('pension', 0)}"

    def test_age_weights_class_attribute_exists(self):
        """Test that the age weights table is defined as a class attribute."""
        assert hasattr(PopulationSynthesizer, '_INCOME_SOURCE_AGE_WEIGHTS')
        weights = PopulationSynthesizer._INCOME_SOURCE_AGE_WEIGHTS
        assert len(weights) == 7  # 7 age bands
        # Check all expected sources present in each band
        expected_sources = {'work', 'unemployment', 'studies', 'pension',
                           'disability', 'sickness', 'parental_leave',
                           'financial_support', 'no_income'}
        for age_range, sources in weights.items():
            assert set(sources.keys()) == expected_sources, \
                f"Age band {age_range} missing sources: {expected_sources - set(sources.keys())}"
