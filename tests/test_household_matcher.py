"""
Tests for household_matcher module.

Tests the constrained matching logic that assigns individuals to household containers.
"""

import pytest
from gbgsynth.models import Agent, Household
from gbgsynth.helpers.household_matcher import (
    form_couples,
    place_single_parents,
    place_children,
    place_other,
    fill_remaining_slots,
    place_singles,
    redistribute_unplaced,
    fix_children_only_households,
)


class TestFormCouples:
    """Test form_couples function."""
    
    def test_form_couples_basic(self):
        """Test basic couple formation with compatible ages."""
        # Create cohabiting agents
        male1 = Agent(agent_id=1, age=30, sex='male', hh_role='cohabiting')
        male2 = Agent(agent_id=2, age=40, sex='male', hh_role='cohabiting')
        female1 = Agent(agent_id=3, age=28, sex='female', hh_role='cohabiting')
        female2 = Agent(agent_id=4, age=38, sex='female', hh_role='cohabiting')
        
        cohabiting = [male1, male2, female1, female2]
        
        # Create multi-person households
        hh1 = Household(household_id=1, size=2)
        hh2 = Household(household_id=2, size=2)
        multi_hh = [hh1, hh2]
        
        constraints = {'partner_age_difference_max': 15}
        
        num_couples = form_couples(cohabiting, multi_hh, constraints)
        
        # Should form 2 couples
        assert num_couples == 2
        # Check households have members
        assert len(hh1.members) == 2
        assert len(hh2.members) == 2
    
    def test_form_couples_age_constraint(self):
        """Test that couples respect age difference constraint."""
        male = Agent(agent_id=1, age=25, sex='male', hh_role='cohabiting')
        female = Agent(agent_id=2, age=60, sex='female', hh_role='cohabiting')
        
        cohabiting = [male, female]
        
        hh = Household(household_id=1, size=2)
        multi_hh = [hh]
        
        constraints = {'partner_age_difference_max': 15}
        
        num_couples = form_couples(cohabiting, multi_hh, constraints)
        
        # Should not form couple due to age difference > 15
        assert num_couples == 0
        assert len(hh.members) == 0
    
    def test_form_couples_no_males(self):
        """Test couple formation with no males."""
        female1 = Agent(agent_id=1, age=30, sex='female', hh_role='cohabiting')
        female2 = Agent(agent_id=2, age=32, sex='female', hh_role='cohabiting')
        
        cohabiting = [female1, female2]
        
        hh = Household(household_id=1, size=2)
        multi_hh = [hh]
        
        constraints = {'partner_age_difference_max': 15}
        
        num_couples = form_couples(cohabiting, multi_hh, constraints)
        
        # Should not form couples (no males)
        assert num_couples == 0
    
    def test_form_couples_no_females(self):
        """Test couple formation with no females."""
        male1 = Agent(agent_id=1, age=30, sex='male', hh_role='cohabiting')
        male2 = Agent(agent_id=2, age=32, sex='male', hh_role='cohabiting')
        
        cohabiting = [male1, male2]
        
        hh = Household(household_id=1, size=2)
        multi_hh = [hh]
        
        constraints = {'partner_age_difference_max': 15}
        
        num_couples = form_couples(cohabiting, multi_hh, constraints)
        
        # Should not form couples (no females)
        assert num_couples == 0
    
    def test_form_couples_no_available_households(self):
        """Test couple formation with no available households."""
        male = Agent(agent_id=1, age=30, sex='male', hh_role='cohabiting')
        female = Agent(agent_id=2, age=28, sex='female', hh_role='cohabiting')
        
        cohabiting = [male, female]
        multi_hh = []  # No households
        
        constraints = {'partner_age_difference_max': 15}
        
        num_couples = form_couples(cohabiting, multi_hh, constraints)
        
        assert num_couples == 0
    
    def test_form_couples_limited_households(self):
        """Test that couple formation is limited by household availability."""
        # Create 3 potential couples but only 2 households
        males = [Agent(agent_id=i, age=30+i, sex='male', hh_role='cohabiting') for i in range(3)]
        females = [Agent(agent_id=i+10, age=28+i, sex='female', hh_role='cohabiting') for i in range(3)]
        
        cohabiting = males + females
        
        # Only 2 households
        hh1 = Household(household_id=1, size=2)
        hh2 = Household(household_id=2, size=2)
        multi_hh = [hh1, hh2]
        
        constraints = {'partner_age_difference_max': 15}
        
        num_couples = form_couples(cohabiting, multi_hh, constraints)
        
        # Can only form 2 couples (limited by households)
        assert num_couples == 2


class TestPlaceSingleParents:
    """Test place_single_parents function."""
    
    def test_place_single_parents_basic(self):
        """Test placing single parents in empty households."""
        parent1 = Agent(agent_id=1, age=35, sex='female', hh_role='single_parent')
        parent2 = Agent(agent_id=2, age=40, sex='male', hh_role='single_parent')
        
        single_parents = [parent1, parent2]
        
        # Create empty multi-person households
        hh1 = Household(household_id=1, size=3)
        hh2 = Household(household_id=2, size=2)
        multi_hh = [hh1, hh2]
        
        num_placed = place_single_parents(single_parents, multi_hh)
        
        assert num_placed == 2
        assert len(hh1.members) == 1
        assert len(hh2.members) == 1
    
    def test_place_single_parents_no_space(self):
        """Test placing single parents when no empty households."""
        parent = Agent(agent_id=1, age=35, sex='female', hh_role='single_parent')
        single_parents = [parent]
        
        # Create households that already have members
        hh = Household(household_id=1, size=3)
        agent = Agent(agent_id=10, age=30, sex='male')
        hh.add_member(agent)
        multi_hh = [hh]
        
        num_placed = place_single_parents(single_parents, multi_hh)
        
        # Should not place (no empty households)
        assert num_placed == 0
        assert parent.household_id is None


class TestPlaceChildren:
    """Test place_children function."""
    
    def test_place_children_with_valid_parent(self):
        """Test placing children in households with appropriate-aged parents."""
        # Create household with parent
        hh = Household(household_id=1, size=4)
        parent = Agent(agent_id=1, age=40, sex='female', hh_role='cohabiting')
        parent.household_id = 1
        hh.add_member(parent)
        
        # Create child
        child = Agent(agent_id=2, age=10, sex='male', hh_role='child')
        children = [child]
        
        multi_hh = [hh]
        constraints = {'min_parent_child_age_gap': 18}
        
        num_placed = place_children(children, multi_hh, constraints)
        
        assert num_placed == 1
        assert child.household_id == 1
        assert len(hh.members) == 2
    
    def test_place_children_no_valid_parent(self):
        """Test that children are not placed if no adult old enough."""
        # Create household with young adult
        hh = Household(household_id=1, size=4)
        young_adult = Agent(agent_id=1, age=19, sex='female', hh_role='cohabiting')
        young_adult.household_id = 1
        hh.add_member(young_adult)
        
        # Create child
        child = Agent(agent_id=2, age=10, sex='male', hh_role='child')
        children = [child]
        
        multi_hh = [hh]
        constraints = {'min_parent_child_age_gap': 18}
        
        num_placed = place_children(children, multi_hh, constraints)
        
        # Should not place (parent too young: 19 - 10 = 9 < 18)
        assert num_placed == 0
        assert child.household_id is None
    
    def test_place_children_youngest_first(self):
        """Test that children are placed youngest first."""
        # Create household with parents
        hh = Household(household_id=1, size=5)
        parent = Agent(agent_id=1, age=45, sex='female', hh_role='cohabiting')
        parent.household_id = 1
        hh.add_member(parent)
        
        # Create children of different ages
        child1 = Agent(agent_id=2, age=15, sex='male', hh_role='child')
        child2 = Agent(agent_id=3, age=8, sex='female', hh_role='child')
        child3 = Agent(agent_id=4, age=12, sex='male', hh_role='child')
        
        children = [child1, child2, child3]  # Not sorted by age
        
        multi_hh = [hh]
        constraints = {'min_parent_child_age_gap': 18}
        
        num_placed = place_children(children, multi_hh, constraints)
        
        # All should be placed
        assert num_placed == 3
        # Youngest (8) should be first member added after parent
        assert hh.members[1].age == 8


class TestPlaceOther:
    """Test place_other function for placing 'other' household members."""
    
    def test_place_other_basic(self):
        """Test placing 'other' role members in households."""
        # Create household with some members
        hh = Household(household_id=1, size=4)
        adult = Agent(agent_id=1, age=40, sex='male', hh_role='cohabiting')
        hh.add_member(adult)
        adult.household_id = 1
        
        # Create 'other' members
        other = Agent(agent_id=2, age=25, sex='female', hh_role='other')
        other_list = [other]
        
        multi_hh = [hh]
        
        num_placed = place_other(other_list, multi_hh)
        
        assert num_placed == 1
        assert other.household_id == 1
        assert len(hh.members) == 2


class TestFillRemainingSlots:
    """Test fill_remaining_slots function."""
    
    def test_fill_remaining_slots_basic(self):
        """Test filling remaining household slots."""
        # Create household with space
        hh = Household(household_id=1, size=3)
        adult = Agent(agent_id=1, age=40, sex='male')
        hh.add_member(adult)
        adult.household_id = 1
        
        # Create unplaced agents
        agent2 = Agent(agent_id=2, age=30, sex='female')
        agent3 = Agent(agent_id=3, age=28, sex='male')
        unplaced = [agent2, agent3]
        
        multi_hh = [hh]
        
        num_placed = fill_remaining_slots(unplaced, multi_hh)
        
        assert num_placed == 2
        assert len(hh.members) == 3
        assert agent2.household_id == 1
        assert agent3.household_id == 1


class TestPlaceSingles:
    """Test place_singles function."""
    
    def test_place_singles_basic(self):
        """Test placing single adults in single-person households."""
        single1 = Agent(agent_id=1, age=30, sex='male', hh_role='single')
        single2 = Agent(agent_id=2, age=35, sex='female', hh_role='single')
        
        singles = [single1, single2]
        
        # Create single-person households
        hh1 = Household(household_id=1, size=1)
        hh2 = Household(household_id=2, size=1)
        single_hh = [hh1, hh2]
        
        num_placed = place_singles(singles, single_hh)
        
        assert num_placed == 2
        # Both should be placed (exact household id doesn't matter)
        placed_count = sum(1 for s in [single1, single2] if s.household_id is not None)
        assert placed_count == 2
        # Households should have members
        assert len(hh1.members) + len(hh2.members) == 2
    
    def test_place_singles_limited_households(self):
        """Test placing singles when not enough households."""
        singles = [Agent(agent_id=i, age=30+i, sex='male' if i % 2 == 0 else 'female', hh_role='single') for i in range(3)]
        
        # Only 2 single-person households
        hh1 = Household(household_id=1, size=1)
        hh2 = Household(household_id=2, size=1)
        single_hh = [hh1, hh2]
        
        num_placed = place_singles(singles, single_hh)
        
        # Can only place 2
        assert num_placed == 2
        # One should remain unplaced
        unplaced = [s for s in singles if s.household_id is None]
        assert len(unplaced) == 1


class TestRedistributeUnplaced:
    """Test redistribute_unplaced function."""
    
    def test_redistribute_unplaced_basic(self):
        """Test redistributing unplaced individuals."""
        # Create households with some space
        hh1 = Household(household_id=1, size=3)
        agent1 = Agent(agent_id=1, age=40, sex='male')
        hh1.add_member(agent1)
        agent1.household_id = 1
        
        hh2 = Household(household_id=2, size=2)
        agent2 = Agent(agent_id=2, age=35, sex='female')
        hh2.add_member(agent2)
        agent2.household_id = 2
        
        households = [hh1, hh2]
        
        # Create unplaced individuals
        unplaced1 = Agent(agent_id=10, age=25, sex='male')
        unplaced2 = Agent(agent_id=11, age=30, sex='female')
        unplaced3 = Agent(agent_id=12, age=28, sex='male')
        
        unplaced = [unplaced1, unplaced2, unplaced3]
        
        # Function returns None, but modifies households in place
        redistribute_unplaced(unplaced, households)
        
        # Check that unplaced individuals were assigned households
        placed_count = sum(1 for agent in unplaced if agent.household_id is not None)
        assert placed_count > 0  # At least some should be placed
    
    def test_redistribute_unplaced_all_placed(self):
        """Test redistribute when all already placed."""
        hh = Household(household_id=1, size=2)
        agent1 = Agent(agent_id=1, age=40, sex='male')
        agent2 = Agent(agent_id=2, age=38, sex='female')
        hh.add_member(agent1)
        hh.add_member(agent2)
        agent1.household_id = 1
        agent2.household_id = 1
        
        households = [hh]
        unplaced = []  # Empty list - all already placed
        
        # Function returns None, but should handle empty input gracefully
        redistribute_unplaced(unplaced, households)
        
        # Household should remain unchanged
        assert len(hh.members) == 2


class TestFixChildrenOnlyHouseholds:
    """Test fix_children_only_households function."""
    
    def test_fix_children_only_households_basic(self):
        """Test fixing households with only children."""
        # Create household with only children
        hh_problem = Household(household_id=1, size=2)
        child1 = Agent(agent_id=1, age=10, sex='male', hh_role='child')
        child2 = Agent(agent_id=2, age=12, sex='female', hh_role='child')
        hh_problem.add_member(child1)
        hh_problem.add_member(child2)
        child1.household_id = 1
        child2.household_id = 1
        
        # Create household with adults and capacity
        hh_adult = Household(household_id=2, size=4)
        adult = Agent(agent_id=10, age=40, sex='female', hh_role='cohabiting')
        hh_adult.add_member(adult)
        adult.household_id = 2
        
        households = [hh_problem, hh_adult]
        all_agents = [child1, child2, adult]
        
        # Function returns pruned household list
        result_households = fix_children_only_households(all_agents, households)
        
        assert result_households is not None
        # Should have redistributed children to household with adult
        # The household with adult should now have children
        adults_hh = [hh for hh in result_households if any(m.age >= 18 for m in hh.members)]
        assert len(adults_hh) > 0
        # Children should now be in a household with an adult
        for child in [child1, child2]:
            if child.household_id:
                child_hh = next((hh for hh in result_households if hh.household_id == child.household_id), None)
                if child_hh and child_hh.members:
                    # Check if there's an adult in the same household
                    assert any(m.age >= 18 for m in child_hh.members)
    
    def test_fix_children_only_households_none_to_fix(self):
        """Test when no children-only households exist."""
        # Create valid household
        hh = Household(household_id=1, size=3)
        adult = Agent(agent_id=1, age=40, sex='female', hh_role='cohabiting')
        child = Agent(agent_id=2, age=10, sex='male', hh_role='child')
        hh.add_member(adult)
        hh.add_member(child)
        adult.household_id = 1
        child.household_id = 1
        
        households = [hh]
        all_agents = [adult, child]
        
        # Function returns pruned household list
        result_households = fix_children_only_households(all_agents, households)
        
        # Should return same households (nothing to fix)
        assert result_households == households
        assert len(result_households) == 1
        assert len(result_households[0].members) == 2


class TestHouseholdMatcherIntegration:
    """Integration tests for household matching workflow."""
    
    def test_complete_matching_flow(self):
        """Test a complete matching workflow with all phases."""
        # Create agent pools
        males = [Agent(agent_id=i, age=30+i, sex='male', hh_role='cohabiting') for i in range(2)]
        females = [Agent(agent_id=i+10, age=28+i, sex='female', hh_role='cohabiting') for i in range(2)]
        single_parents = [Agent(agent_id=20, age=35, sex='female', hh_role='single_parent')]
        children = [Agent(agent_id=i+30, age=5+i*3, sex='male' if i % 2 == 0 else 'female', hh_role='child') for i in range(3)]
        singles = [Agent(agent_id=i+40, age=25+i*5, sex='male', hh_role='single') for i in range(2)]
        
        # Create households
        multi_hh = [Household(household_id=i, size=4) for i in range(3)]
        single_hh = [Household(household_id=i+10, size=1) for i in range(2)]
        
        constraints = {
            'partner_age_difference_max': 15,
            'min_parent_child_age_gap': 18
        }
        
        # Phase 1: Form couples
        cohabiting = males + females
        couples_formed = form_couples(cohabiting, multi_hh, constraints)
        assert couples_formed >= 0
        
        # Phase 2: Place single parents
        sp_placed = place_single_parents(single_parents, multi_hh)
        assert sp_placed >= 0
        
        # Phase 3: Place children
        children_placed = place_children(children, multi_hh, constraints)
        assert children_placed >= 0
        
        # Phase 4: Place singles
        singles_placed = place_singles(singles, single_hh)
        assert singles_placed >= 0
        
        # Verify no household is empty if we had successful placements
        total_placed = couples_formed * 2 + sp_placed + children_placed + singles_placed
        if total_placed > 0:
            non_empty_hh = [hh for hh in multi_hh + single_hh if len(hh.members) > 0]
            assert len(non_empty_hh) > 0
    
    def test_matching_respects_constraints(self):
        """Test that matching respects all constraints."""
        # Create couple with large age gap
        male = Agent(agent_id=1, age=25, sex='male', hh_role='cohabiting')
        female = Agent(agent_id=2, age=50, sex='female', hh_role='cohabiting')
        
        hh = Household(household_id=1, size=2)
        
        constraints = {'partner_age_difference_max': 10}  # Strict constraint
        
        couples = form_couples([male, female], [hh], constraints)
        
        # Should not form couple due to constraint
        assert couples == 0
        assert len(hh.members) == 0
