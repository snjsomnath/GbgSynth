"""
Tests for the GbgArea class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from gbgsynth.area import GbgArea
from gbgsynth.models import Agent, Household


class TestGbgAreaInitialization:
    """Tests for GbgArea initialization."""

    def test_basic_initialization(self):
        """Test basic area initialization."""
        area = GbgArea(
            area_code="107",
            area_name="107 Haga",
            year=2023
        )
        
        assert area.area_code == "107"
        assert area.area_name == "107 Haga"
        assert area.year == 2023
        assert area.individuals == []
        assert area.households == []
        assert not area._is_generated

    def test_initialization_with_api_value(self):
        """Test initialization with explicit API value."""
        area = GbgArea(
            area_code="107",
            area_name="Haga",
            year=2023,
            area_api_value="107 Haga"
        )
        
        assert area.area_api_value == "107 Haga"

    def test_api_value_defaults_to_name(self):
        """Test that API value defaults to area name."""
        area = GbgArea(
            area_code="107",
            area_name="107 Haga",
            year=2023
        )
        
        assert area.area_api_value == area.area_name

    def test_initialization_creates_client(self):
        """Test that initialization creates a client if not provided."""
        area = GbgArea(
            area_code="107",
            area_name="107 Haga",
            year=2023
        )
        
        assert area.client is not None

    def test_initialization_creates_config(self):
        """Test that initialization creates a config if not provided."""
        area = GbgArea(
            area_code="107",
            area_name="107 Haga",
            year=2023
        )
        
        assert area.config is not None


class TestGbgAreaAttributes:
    """Tests for GbgArea attributes and state."""

    @pytest.fixture
    def area(self):
        """Create a GbgArea instance for testing."""
        return GbgArea(
            area_code="107",
            area_name="107 Haga",
            year=2023
        )

    def test_marginals_storage(self, area):
        """Test that area has marginals storage."""
        assert hasattr(area, '_marginals')
        assert isinstance(area._marginals, dict)

    def test_stats_storage(self, area):
        """Test that area has stats storage."""
        assert hasattr(area, 'stats')
        assert isinstance(area.stats, dict)

    def test_is_generated_flag(self, area):
        """Test the is_generated flag."""
        assert not area._is_generated


class TestGbgAreaExportMethods:
    """Tests for GbgArea export functionality."""

    @pytest.fixture
    def area_with_data(self):
        """Create an area with some synthetic data."""
        area = GbgArea(
            area_code="107",
            area_name="107 Haga",
            year=2023
        )
        
        # Add some test data
        hh = Household(household_id=1, size=2)
        agent1 = Agent(agent_id=1, age=35, sex='male', income=500000)
        agent2 = Agent(agent_id=2, age=32, sex='female', income=450000)
        
        hh.add_member(agent1)
        hh.add_member(agent2)
        
        area.individuals = [agent1, agent2]
        area.households = [hh]
        area._is_generated = True
        
        return area

    def test_has_individuals_after_setting(self, area_with_data):
        """Test that area has individuals after setting."""
        assert len(area_with_data.individuals) == 2

    def test_has_households_after_setting(self, area_with_data):
        """Test that area has households after setting."""
        assert len(area_with_data.households) == 1

    def test_individuals_are_agents(self, area_with_data):
        """Test that individuals are Agent instances."""
        for ind in area_with_data.individuals:
            assert isinstance(ind, Agent)

    def test_households_are_household_instances(self, area_with_data):
        """Test that households are Household instances."""
        for hh in area_with_data.households:
            assert isinstance(hh, Household)


class TestGbgAreaGenerateMethod:
    """Tests for the generate method (without actual API calls)."""

    @pytest.fixture
    def mock_area(self):
        """Create an area with mocked client."""
        with patch('gbgsynth.area.PxWebClient'):
            area = GbgArea(
                area_code="107",
                area_name="107 Haga",
                year=2023
            )
            return area

    def test_generate_accepts_buildings(self, mock_area):
        """Test that generate method accepts buildings parameter."""
        # The method signature should accept buildings
        import inspect
        sig = inspect.signature(mock_area.generate)
        assert 'buildings' in sig.parameters

    def test_generate_accepts_allocate_dwellings(self, mock_area):
        """Test that generate method accepts allocate_dwellings flag."""
        import inspect
        sig = inspect.signature(mock_area.generate)
        assert 'allocate_dwellings' in sig.parameters


class TestGbgAreaYearHandling:
    """Tests for year handling in GbgArea."""

    def test_accepts_different_years(self):
        """Test that different years are accepted."""
        area_2020 = GbgArea(area_code="107", area_name="Haga", year=2020)
        area_2023 = GbgArea(area_code="107", area_name="Haga", year=2023)
        
        assert area_2020.year == 2020
        assert area_2023.year == 2023

    def test_year_stored_correctly(self):
        """Test that year is stored correctly."""
        area = GbgArea(area_code="107", area_name="Haga", year=2022)
        assert area.year == 2022


class TestGbgAreaCodeHandling:
    """Tests for area code handling."""

    def test_different_area_codes(self):
        """Test handling of different area codes."""
        codes = ["101", "107", "203", "999"]
        
        for code in codes:
            area = GbgArea(area_code=code, area_name=f"{code} Test", year=2023)
            assert area.area_code == code

    def test_area_code_is_string(self):
        """Test that area code is stored as string."""
        area = GbgArea(area_code="107", area_name="Haga", year=2023)
        assert isinstance(area.area_code, str)


class TestCompareMedianIncome:
    """Tests for _compare_median_income method."""

    def _make_area_with_median(self, census_rows, individuals):
        """Helper: build a GbgArea with pre-set marginals and individuals."""
        area = GbgArea(area_code="107", area_name="Haga", year=2023)
        area._is_generated = True
        area.individuals = individuals
        area.households = []
        edu_df = pd.DataFrame(census_rows)
        area._marginals = {
            'population': pd.DataFrame(),
            'household': pd.DataFrame(),
            'household_position': None,
            'income': None,
            'education_level': edu_df,
            'income_source': None,
        }
        return area

    def test_returns_empty_dict_when_no_education_data(self):
        """Median income comparison is empty when no education data."""
        area = GbgArea(area_code="107", area_name="Haga", year=2023)
        area._is_generated = True
        area.individuals = []
        area.households = []
        area._marginals = {
            'population': pd.DataFrame(),
            'household': pd.DataFrame(),
            'household_position': None,
            'income': None,
            'education_level': None,
            'income_source': None,
        }
        result = area._compare_median_income()
        assert result == {}

    def test_basic_median_comparison(self):
        """Census median vs synth median for a single group."""
        census_rows = [
            {'Tabellvärde': 'Medianinkomst', 'Kön': 'Man', 'Ålder': '25-34 år',
             'Utbildningsnivå': 'Gymnasial utbildning', 'Antal': 300_000},
        ]
        agents = [
            Agent(agent_id=1, age=28, sex='male', education='secondary', income=310_000),
            Agent(agent_id=2, age=30, sex='male', education='secondary', income=290_000),
            Agent(agent_id=3, age=32, sex='male', education='secondary', income=305_000),
        ]
        area = self._make_area_with_median(census_rows, agents)
        result = area._compare_median_income()

        assert result != {}
        assert result['name'] == 'Median Income (SEK, informational)'
        assert len(result['comparison']) == 1

        row = result['comparison'][0]
        assert row['actual'] == 300_000
        # Synth median of [290000, 305000, 310000] = 305000
        assert row['synth'] == 305_000

    def test_multiple_groups(self):
        """Two census groups produce two comparison rows."""
        census_rows = [
            {'Tabellvärde': 'Medianinkomst', 'Kön': 'Man', 'Ålder': '25-34 år',
             'Utbildningsnivå': 'Gymnasial utbildning', 'Antal': 300_000},
            {'Tabellvärde': 'Medianinkomst', 'Kön': 'Kvinna', 'Ålder': '25-34 år',
             'Utbildningsnivå': 'Eftergymnasial utbildning', 'Antal': 350_000},
        ]
        agents = [
            Agent(agent_id=1, age=28, sex='male', education='secondary', income=310_000),
            Agent(agent_id=2, age=30, sex='female', education='post_secondary', income=360_000),
        ]
        area = self._make_area_with_median(census_rows, agents)
        result = area._compare_median_income()
        assert len(result['comparison']) == 2

    def test_children_excluded(self):
        """Children (age < 18) are not included in synth median."""
        census_rows = [
            {'Tabellvärde': 'Medianinkomst', 'Kön': 'Man', 'Ålder': '25-34 år',
             'Utbildningsnivå': 'Gymnasial utbildning', 'Antal': 300_000},
        ]
        agents = [
            Agent(agent_id=1, age=10, sex='male', education='secondary', income=0),
            Agent(agent_id=2, age=28, sex='male', education='secondary', income=310_000),
        ]
        area = self._make_area_with_median(census_rows, agents)
        result = area._compare_median_income()
        row = result['comparison'][0]
        # Only the adult (310k) should be in the synth median
        assert row['synth'] == 310_000

    def test_folkmaengd_rows_ignored(self):
        """Folkmängd rows are not treated as median income values."""
        census_rows = [
            {'Tabellvärde': 'Folkmängd', 'Kön': 'Man', 'Ålder': '25-34 år',
             'Utbildningsnivå': 'Gymnasial utbildning', 'Antal': 500},
            {'Tabellvärde': 'Medianinkomst', 'Kön': 'Man', 'Ålder': '25-34 år',
             'Utbildningsnivå': 'Gymnasial utbildning', 'Antal': 300_000},
        ]
        agents = [
            Agent(agent_id=1, age=28, sex='male', education='secondary', income=310_000),
        ]
        area = self._make_area_with_median(census_rows, agents)
        result = area._compare_median_income()
        assert len(result['comparison']) == 1
        assert result['comparison'][0]['actual'] == 300_000


class TestCompareHhTypeChildren:
    """Tests for _compare_hh_type_children joint validation."""

    def _make_area(self, census_rows, households, individuals):
        """Helper to create a GbgArea with hh_type_children marginal."""
        area = GbgArea(area_code="107", area_name="107 Haga", year=2023)
        area._is_generated = True
        area.households = households
        area.individuals = individuals
        area._marginals = {
            'population': pd.DataFrame(),
            'household': pd.DataFrame(),
            'household_position': None,
            'income': None,
            'education_level': None,
            'income_source': None,
            'hh_type_children': pd.DataFrame(census_rows) if census_rows else None,
        }
        return area

    def test_empty_data_returns_empty(self):
        """No HH type × children data returns empty dict."""
        area = self._make_area(None, [], [])
        result = area._compare_hh_type_children()
        assert result == {}

    def test_basic_comparison(self):
        """Test basic HH type × children comparison."""
        census_rows = [
            {'Hushållstyp': 'Ensamstående', 'Antal barn 0-17 år': '0 barn', 'Antal': 50},
            {'Hushållstyp': 'Ensamstående', 'Antal barn 0-17 år': '1 barn', 'Antal': 10},
            {'Hushållstyp': 'Sammanboende', 'Antal barn 0-17 år': '0 barn', 'Antal': 30},
            {'Hushållstyp': 'Sammanboende', 'Antal barn 0-17 år': '2 barn', 'Antal': 20},
        ]
        # Create synth households:
        # 2 single adults (no children) → Ensamstående | 0 barn × 2
        # 1 couple with 2 kids → Sammanboende | 2 barn × 1
        adults = [
            Agent(agent_id=1, age=30, sex='male', hh_role='single'),
            Agent(agent_id=2, age=40, sex='female', hh_role='single'),
            Agent(agent_id=3, age=35, sex='male', hh_role='cohabiting'),
            Agent(agent_id=4, age=33, sex='female', hh_role='cohabiting'),
            Agent(agent_id=5, age=8, sex='male', hh_role='child'),
            Agent(agent_id=6, age=5, sex='female', hh_role='child'),
        ]
        adults[0].household_id = 1
        adults[1].household_id = 2
        for a in adults[2:]:
            a.household_id = 3
        households = [
            Household(household_id=1, size=1, members=[adults[0]]),
            Household(household_id=2, size=1, members=[adults[1]]),
            Household(household_id=3, size=4, members=adults[2:]),
        ]
        area = self._make_area(census_rows, households, adults)
        result = area._compare_hh_type_children()

        assert result['name'] == 'HH Type × Children 0-17 (informational)'
        assert len(result['comparison']) > 0
        # Check that census key exists in comparison
        keys = [r['category'] for r in result['comparison']]
        assert 'Ensamstående | 0 barn' in keys

    def test_four_plus_children(self):
        """HH with 4+ children should map to '4 barn eller fler'."""
        census_rows = [
            {'Hushållstyp': 'Sammanboende', 'Antal barn 0-17 år': '4 barn eller fler', 'Antal': 5},
        ]
        children = [
            Agent(agent_id=i, age=i + 3, sex='male', hh_role='child')
            for i in range(1, 6)  # 5 children aged 4-8
        ]
        parents = [
            Agent(agent_id=10, age=40, sex='male', hh_role='cohabiting'),
            Agent(agent_id=11, age=38, sex='female', hh_role='cohabiting'),
        ]
        members = parents + children
        for m in members:
            m.household_id = 1
        hh = Household(household_id=1, size=7, members=members)
        area = self._make_area(census_rows, [hh], members)
        result = area._compare_hh_type_children()
        for row in result['comparison']:
            if row['category'] == 'Sammanboende | 4 barn eller fler':
                assert row['synth'] == 1
                break
        else:
            pytest.fail("Did not find 4+ children category")


class TestCompareJointRoleAgeSex:
    """Tests for _compare_joint_role_age_sex validation."""

    def _make_area(self, census_rows, individuals, households):
        """Helper to create a GbgArea with household_position marginal."""
        area = GbgArea(area_code="107", area_name="107 Haga", year=2023)
        area._is_generated = True
        area.individuals = individuals
        area.households = households
        area._marginals = {
            'population': pd.DataFrame(),
            'household': pd.DataFrame(),
            'household_position': pd.DataFrame(census_rows) if census_rows else None,
            'income': None,
            'education_level': None,
            'income_source': None,
            'hh_type_children': None,
        }
        return area

    def test_empty_data_returns_empty(self):
        """No household position data returns empty dict."""
        area = self._make_area(None, [], [])
        result = area._compare_joint_role_age_sex()
        assert result == {}

    def test_basic_seven_categories(self):
        """Test that the 7-category role comparison is produced."""
        census_rows = [
            {'Ålder': '25-34 år', 'Kön': 'Män', 'Hushållsställning': 'Person i gift par/registrerat partnerskap', 'Antal': 10},
            {'Ålder': '25-34 år', 'Kön': 'Män', 'Hushållsställning': 'Personer i samboförhållande', 'Antal': 20},
            {'Ålder': '25-34 år', 'Kön': 'Män', 'Hushållsställning': 'Ensamstående förälder', 'Antal': 5},
            {'Ålder': '25-34 år', 'Kön': 'Män', 'Hushållsställning': 'Barn', 'Antal': 15},
            {'Ålder': '25-34 år', 'Kön': 'Män', 'Hushållsställning': 'Ensamboende', 'Antal': 30},
            {'Ålder': '25-34 år', 'Kön': 'Män', 'Hushållsställning': 'Ej ensamboende personer, övriga', 'Antal': 8},
        ]
        # Create synth individuals
        ind1 = Agent(agent_id=1, age=28, sex='male', hh_role='cohabiting')
        ind1.household_id = 1
        ind2 = Agent(agent_id=2, age=30, sex='male', hh_role='single')
        ind2.household_id = 2
        hh1 = Household(household_id=1, size=2, members=[ind1])
        hh2 = Household(household_id=2, size=1, members=[ind2])

        area = self._make_area(census_rows, [ind1, ind2], [hh1, hh2])
        result = area._compare_joint_role_age_sex()

        assert 'Detailed HH Position' in result['name']
        categories = [r['category'] for r in result['comparison']]
        # Should have census categories (minus Uppgift saknas)
        assert 'Ensamboende' in categories
        assert 'Barn' in categories

    def test_single_parent_detected(self):
        """Test that single parent in multi-person HH is correctly classified."""
        census_rows = [
            {'Ålder': '35-44 år', 'Kön': 'Kvinnor', 'Hushållsställning': 'Ensamstående förälder', 'Antal': 10},
            {'Ålder': '0-5 år', 'Kön': 'Män', 'Hushållsställning': 'Barn', 'Antal': 5},
        ]
        parent = Agent(agent_id=1, age=38, sex='female', hh_role='single')
        parent.household_id = 1
        child = Agent(agent_id=2, age=4, sex='male', hh_role='child')
        child.household_id = 1
        hh = Household(household_id=1, size=2, members=[parent, child])

        area = self._make_area(census_rows, [parent, child], [hh])
        result = area._compare_joint_role_age_sex()

        for row in result['comparison']:
            if row['category'] == 'Ensam förälder':
                assert row['synth'] == 1
                break
        else:
            pytest.fail("Did not find Ensam förälder in synth")
