"""
Tests for data models (Agent, Household, Dwelling).
"""

import pytest
from gbgsynth.models import Agent, Household, Dwelling


class TestDwelling:
    """Tests for the Dwelling class."""

    def test_create_small_apartment(self):
        """Test creating a small studio apartment."""
        dwelling = Dwelling(dwelling_id=1, floor_area=30.0)
        assert dwelling.dwelling_id == 1
        assert dwelling.floor_area == 30.0
        assert dwelling.recommended_occupants == 1
        assert dwelling.max_occupants == 2  # 30/15 = 2
        assert dwelling.is_vacant()

    def test_create_family_apartment(self):
        """Test creating a family-sized apartment."""
        dwelling = Dwelling(dwelling_id=2, floor_area=95.0, house_type='apartment')
        assert dwelling.recommended_occupants == 4  # 95m² -> 4 persons (95-119 range)
        assert dwelling.max_occupants == 6  # 95/15 = 6
        assert dwelling.can_fit(3)
        assert dwelling.can_fit(4)
        assert not dwelling.can_fit(7)

    def test_create_house(self):
        """Test creating a detached house."""
        dwelling = Dwelling(
            dwelling_id=3, 
            floor_area=150.0, 
            house_type='detached_house',
            house_type_sv='Småhus'
        )
        assert dwelling.recommended_occupants == 6  # 150m²+ -> 6 persons
        assert dwelling.house_type == 'detached_house'
        assert dwelling.house_type_sv == 'Småhus'

    def test_dwelling_vacancy(self):
        """Test dwelling vacancy tracking."""
        dwelling = Dwelling(dwelling_id=4, floor_area=60.0)
        assert dwelling.is_vacant()
        
        dwelling.household_id = 100
        assert not dwelling.is_vacant()

    def test_dwelling_to_dict(self):
        """Test dwelling serialization."""
        dwelling = Dwelling(
            dwelling_id=5,
            floor_area=75.5,
            floor_area_range='71-80',
            house_type='apartment'
        )
        d = dwelling.to_dict()
        assert d['dwelling_id'] == 5
        assert d['floor_area'] == 75.5
        assert d['floor_area_range'] == '71-80'
        assert d['is_vacant'] == True

    def test_household_dwelling_assignment(self):
        """Test assigning a dwelling to a household."""
        dwelling = Dwelling(dwelling_id=10, floor_area=80.0)
        hh = Household(household_id=20, size=3)
        
        # Assign dwelling to household
        hh.assign_dwelling(dwelling)
        
        # Check household has dwelling reference
        assert hh.dwelling_id == 10
        assert hh.dwelling is dwelling
        assert hh.floor_area == 80.0
        
        # Check dwelling is no longer vacant
        assert dwelling.household_id == 20
        assert not dwelling.is_vacant()


class TestAgent:
    """Tests for the Agent class."""

    def test_create_valid_adult_male(self):
        """Test creating a valid adult male agent."""
        agent = Agent(agent_id=1, age=35, sex='male')
        assert agent.agent_id == 1
        assert agent.age == 35
        assert agent.sex == 'male'
        assert agent.is_adult()
        assert not agent.is_child()

    def test_create_valid_adult_female(self):
        """Test creating a valid adult female agent."""
        agent = Agent(agent_id=2, age=28, sex='female', hh_role='cohabiting')
        assert agent.agent_id == 2
        assert agent.age == 28
        assert agent.sex == 'female'
        assert agent.hh_role == 'cohabiting'

    def test_create_child(self):
        """Test creating a child agent - role should be auto-assigned."""
        agent = Agent(agent_id=3, age=10, sex='male')
        assert agent.age == 10
        assert agent.hh_role == 'child'
        assert agent.status == 'child'
        assert agent.income == 0
        assert agent.is_child()
        assert not agent.is_adult()

    def test_create_teenager(self):
        """Test creating a teenage agent - should be student."""
        agent = Agent(agent_id=4, age=16, sex='female')
        assert agent.hh_role == 'child'
        assert agent.status == 'student'
        assert agent.is_child()

    def test_create_elderly(self):
        """Test creating an elderly agent - should be retired."""
        agent = Agent(agent_id=5, age=70, sex='male')
        assert agent.status == 'retired'
        assert agent.is_adult()

    def test_invalid_sex(self):
        """Test that invalid sex raises ValueError."""
        with pytest.raises(ValueError, match="Invalid sex"):
            Agent(agent_id=1, age=30, sex='unknown')

    def test_invalid_age_negative(self):
        """Test that negative age raises ValueError."""
        with pytest.raises(ValueError, match="Invalid age"):
            Agent(agent_id=1, age=-5, sex='male')

    def test_invalid_age_too_old(self):
        """Test that age > 120 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid age"):
            Agent(agent_id=1, age=150, sex='female')

    def test_can_be_parent(self):
        """Test can_be_parent method."""
        adult = Agent(agent_id=1, age=30, sex='female')
        child = Agent(agent_id=2, age=10, sex='male')
        
        assert adult.can_be_parent()
        assert not child.can_be_parent()

    def test_can_be_partner_with_compatible(self):
        """Test partnership compatibility between compatible agents."""
        male = Agent(agent_id=1, age=35, sex='male')
        female = Agent(agent_id=2, age=32, sex='female')
        
        assert male.can_be_partner_with(female)
        assert female.can_be_partner_with(male)

    def test_can_be_partner_with_same_sex(self):
        """Test partnership with same sex agents (default behavior)."""
        male1 = Agent(agent_id=1, age=35, sex='male')
        male2 = Agent(agent_id=2, age=32, sex='male')
        
        # Default behavior doesn't allow same-sex partnerships
        assert not male1.can_be_partner_with(male2)

    def test_can_be_partner_with_same_sex_allowed(self):
        """Test partnership with same-sex agents when explicitly allowed."""
        male1 = Agent(agent_id=1, age=35, sex='male')
        male2 = Agent(agent_id=2, age=32, sex='male')
        female1 = Agent(agent_id=3, age=30, sex='female')
        female2 = Agent(agent_id=4, age=28, sex='female')

        assert male1.can_be_partner_with(male2, allow_same_sex=True)
        assert female1.can_be_partner_with(female2, allow_same_sex=True)

    def test_can_be_partner_with_age_difference_limit(self):
        """Test partnership with large age difference."""
        young = Agent(agent_id=1, age=25, sex='male')
        older = Agent(agent_id=2, age=50, sex='female')
        
        # Default max_age_diff is 10
        assert not young.can_be_partner_with(older)
        # But with higher limit, should work
        assert young.can_be_partner_with(older, max_age_diff=30)

    def test_can_be_partner_with_child(self):
        """Test that children cannot be partners."""
        adult = Agent(agent_id=1, age=30, sex='male')
        child = Agent(agent_id=2, age=15, sex='female')
        
        assert not adult.can_be_partner_with(child)

    def test_can_be_parent_of(self):
        """Test parent-child relationship validation."""
        parent = Agent(agent_id=1, age=35, sex='female')
        child = Agent(agent_id=2, age=10, sex='male')
        
        assert parent.can_be_parent_of(child)

    def test_can_be_parent_of_age_gap_too_small(self):
        """Test parent-child with insufficient age gap."""
        young_adult = Agent(agent_id=1, age=25, sex='female')
        teen = Agent(agent_id=2, age=15, sex='male')
        
        # 10 year gap is less than min_age_gap of 18
        assert not young_adult.can_be_parent_of(teen)

    def test_to_dict(self):
        """Test agent serialization to dictionary."""
        agent = Agent(
            agent_id=1, age=35, sex='male', hh_role='single',
            status='employed', income=500000, income_decile=7
        )
        d = agent.to_dict()
        
        assert d['agent_id'] == 1
        assert d['age'] == 35
        assert d['sex'] == 'male'
        assert d['income'] == 500000
        assert d['income_decile'] == 7


class TestHousehold:
    """Tests for the Household class."""

    def test_create_empty_household(self):
        """Test creating an empty household."""
        hh = Household(household_id=1, size=3)
        
        assert hh.household_id == 1
        assert hh.size == 3
        assert len(hh.members) == 0
        assert hh.cars == 0
        assert not hh.is_full()

    def test_add_member(self):
        """Test adding a member to household."""
        hh = Household(household_id=1, size=2)
        agent = Agent(agent_id=1, age=35, sex='male')
        
        hh.add_member(agent)
        
        assert len(hh.members) == 1
        assert agent.household_id == 1
        assert hh.head_id == 1  # First adult becomes head

    def test_add_couple(self):
        """Test adding a couple to household."""
        hh = Household(household_id=1, size=2)
        male = Agent(agent_id=1, age=35, sex='male')
        female = Agent(agent_id=2, age=32, sex='female')
        
        hh.add_member(male)
        hh.add_member(female)
        
        assert hh.head_id == 1
        assert hh.partner_id == 2
        assert hh.is_full()
        assert hh.is_couple()

    def test_add_child(self):
        """Test adding a child to household."""
        hh = Household(household_id=1, size=3)
        parent = Agent(agent_id=1, age=35, sex='female')
        child = Agent(agent_id=2, age=10, sex='male')
        
        hh.add_member(parent)
        hh.add_member(child)
        
        assert hh.head_id == 1
        assert 2 in hh.child_ids
        assert hh.is_single_parent()

    def test_add_member_overflow(self):
        """Test that adding member to full household raises error."""
        hh = Household(household_id=1, size=1)
        hh.add_member(Agent(agent_id=1, age=35, sex='male'))
        
        with pytest.raises(ValueError, match="already full"):
            hh.add_member(Agent(agent_id=2, age=30, sex='female'))

    def test_is_full(self):
        """Test is_full method."""
        hh = Household(household_id=1, size=2)
        assert not hh.is_full()
        
        hh.add_member(Agent(agent_id=1, age=35, sex='male'))
        assert not hh.is_full()
        
        hh.add_member(Agent(agent_id=2, age=32, sex='female'))
        assert hh.is_full()

    def test_has_capacity(self):
        """Test has_capacity method."""
        hh = Household(household_id=1, size=3)
        
        assert hh.can_fit(1)
        assert hh.can_fit(3)
        assert not hh.can_fit(4)
        
        hh.add_member(Agent(agent_id=1, age=35, sex='male'))
        assert hh.can_fit(2)
        assert not hh.can_fit(3)

    def test_get_head(self):
        """Test getting household head."""
        hh = Household(household_id=1, size=2)
        agent = Agent(agent_id=1, age=35, sex='male')
        hh.add_member(agent)
        
        head = hh.head
        assert head is not None
        assert head.agent_id == 1

    def test_get_head_empty_household(self):
        """Test getting head from empty household."""
        hh = Household(household_id=1, size=2)
        assert hh.head is None

    def test_get_partner(self):
        """Test getting partner."""
        hh = Household(household_id=1, size=2)
        hh.add_member(Agent(agent_id=1, age=35, sex='male'))
        hh.add_member(Agent(agent_id=2, age=32, sex='female'))
        
        partner = hh.partner
        assert partner is not None
        assert partner.agent_id == 2

    def test_get_children(self):
        """Test getting children from household."""
        hh = Household(household_id=1, size=4)
        hh.add_member(Agent(agent_id=1, age=40, sex='male'))
        hh.add_member(Agent(agent_id=2, age=38, sex='female'))
        hh.add_member(Agent(agent_id=3, age=12, sex='male'))
        hh.add_member(Agent(agent_id=4, age=8, sex='female'))
        
        children = hh.children
        assert len(children) == 2
        assert all(c.is_child() for c in children)

    def test_is_single_person(self):
        """Test single person household detection."""
        single_hh = Household(household_id=1, size=1)
        multi_hh = Household(household_id=2, size=3)
        
        assert single_hh.is_single()
        assert not multi_hh.is_single()

    def test_get_household_income(self):
        """Test household income calculation."""
        hh = Household(household_id=1, size=3)
        hh.add_member(Agent(agent_id=1, age=40, sex='male', income=600000))
        hh.add_member(Agent(agent_id=2, age=38, sex='female', income=400000))
        hh.add_member(Agent(agent_id=3, age=12, sex='male'))  # Child, income=0
        
        assert hh.income == 1000000

    def test_to_dict(self):
        """Test household serialization."""
        hh = Household(household_id=1, size=2, house_type='apartment', cars=1)
        hh.add_member(Agent(agent_id=1, age=35, sex='male', income=500000))
        hh.add_member(Agent(agent_id=2, age=32, sex='female', income=450000))
        
        d = hh.to_dict()
        
        assert d['household_id'] == 1
        assert d['size'] == 2
        assert d['actual_size'] == 2
        assert d['house_type'] == 'apartment'
        assert d['cars'] == 1
        assert d['is_couple'] == True
        assert d['total_income'] == 950000


class TestAgentHouseholdIntegration:
    """Integration tests for Agent and Household interaction."""

    def test_family_household(self):
        """Test creating a complete family household."""
        hh = Household(household_id=1, size=4)
        
        father = Agent(agent_id=1, age=42, sex='male', status='employed', income=650000)
        mother = Agent(agent_id=2, age=40, sex='female', status='employed', income=550000)
        son = Agent(agent_id=3, age=15, sex='male')
        daughter = Agent(agent_id=4, age=12, sex='female')
        
        hh.add_member(father)
        hh.add_member(mother)
        hh.add_member(son)
        hh.add_member(daughter)
        
        assert hh.is_full()
        assert hh.is_couple()
        assert not hh.is_single_parent()
        assert len(hh.children) == 2
        assert hh.income == 1200000
        
        # Verify parent can be parent of children
        assert father.can_be_parent_of(son)
        assert mother.can_be_parent_of(daughter)
        
        # Verify couple compatibility
        assert father.can_be_partner_with(mother)
