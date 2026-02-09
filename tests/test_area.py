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
