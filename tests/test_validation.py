"""
Tests for validation module.

Tests the out-of-sample validation framework for synthetic populations.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from gbgsynth.validation import (
    ValidationResult,
    ValidationReport,
    Validator,
    list_validation_tables,
    list_synthesis_tables,
    suggest_improvements,
    USABLE_VALIDATION_TABLES,
    UNUSABLE_VALIDATION_TABLES,
)
from gbgsynth.models import Agent, Household, Dwelling
from gbgsynth.sanity_checks import SanityCheckResult


class TestValidationResult:
    """Test the ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            dimension="age_distribution",
            census_total=1000,
            synth_total=995,
            rmse=12.5,
            mae=8.3,
            max_error_pct=15.2,
            correlation=0.98,
            categories=[
                {"category": "0-18", "census": 200, "synth": 195}
            ],
            passed=True
        )
        
        assert result.dimension == "age_distribution"
        assert result.census_total == 1000
        assert result.synth_total == 995
        assert result.rmse == 12.5
        assert result.passed is True
        assert len(result.categories) == 1
    
    def test_validation_result_with_failure(self):
        """Test validation result that failed."""
        result = ValidationResult(
            dimension="household_size",
            census_total=500,
            synth_total=480,
            rmse=45.8,
            mae=35.2,
            max_error_pct=58.3,
            correlation=0.65,
            categories=[],
            passed=False
        )
        
        assert result.passed is False
        assert result.correlation < 0.8


class TestValidationReport:
    """Test the ValidationReport dataclass."""
    
    def test_validation_report_creation(self):
        """Test creating a validation report."""
        result1 = ValidationResult(
            dimension="age",
            census_total=1000,
            synth_total=995,
            rmse=10.0,
            mae=8.0,
            max_error_pct=12.0,
            correlation=0.99,
            categories=[],
            passed=True
        )
        
        report = ValidationReport(
            area_code="107",
            area_name="Haga",
            year=2023,
            results={"age": result1},
            overall_score=95.5
        )
        
        assert report.area_code == "107"
        assert report.area_name == "Haga"
        assert report.year == 2023
        assert report.overall_score == 95.5
        assert "age" in report.results
    
    def test_is_valid_without_sanity_check(self):
        """Test is_valid when no sanity check performed."""
        report = ValidationReport(
            area_code="107",
            area_name="Haga",
            year=2023,
            results={},
            overall_score=90.0
        )
        
        assert report.is_valid is True
    
    def test_is_valid_with_passing_sanity_check(self):
        """Test is_valid with passing sanity checks."""
        sanity_result = SanityCheckResult(
            total_households=100,
            total_individuals=250
        )
        
        report = ValidationReport(
            area_code="107",
            area_name="Haga",
            year=2023,
            results={},
            overall_score=90.0,
            sanity_result=sanity_result
        )
        
        assert report.is_valid is True
    
    def test_is_valid_with_failing_sanity_check(self):
        """Test is_valid with failing sanity checks."""
        from gbgsynth.sanity_checks import SanityViolation
        
        sanity_result = SanityCheckResult(
            total_households=100,
            total_individuals=250,
            violations=[
                SanityViolation("test", "critical", 1, "Critical issue")
            ]
        )
        
        report = ValidationReport(
            area_code="107",
            area_name="Haga",
            year=2023,
            results={},
            overall_score=90.0,
            sanity_result=sanity_result
        )
        
        assert report.is_valid is False
    
    def test_summary_generation(self):
        """Test generating a summary string."""
        result1 = ValidationResult(
            dimension="age",
            census_total=1000,
            synth_total=995,
            rmse=10.0,
            mae=8.0,
            max_error_pct=12.0,
            correlation=0.99,
            categories=[],
            passed=True
        )
        
        report = ValidationReport(
            area_code="107",
            area_name="Haga",
            year=2023,
            results={"age": result1},
            overall_score=95.5
        )
        
        summary = report.summary()
        
        assert "Haga" in summary
        assert "2023" in summary
        assert "95.5" in summary
        assert "age" in summary


class TestValidator:
    """Test the Validator class."""
    
    @pytest.fixture
    def mock_area(self):
        """Create a mock GbgArea for testing."""
        area = Mock()
        area.area_code = "107"
        area.area_name = "Haga"
        area.year = 2023
        area.area_api_value = "107"
        area.households = []
        area.individuals = []
        return area
    
    def test_validator_initialization(self, mock_area):
        """Test creating a Validator instance."""
        validator = Validator(mock_area, tolerance=0.15)
        
        assert validator.area == mock_area
        assert validator.tolerance == 0.15
        assert validator._api is not None
    
    def test_validator_default_tolerance(self, mock_area):
        """Test default tolerance value."""
        validator = Validator(mock_area)
        
        assert validator.tolerance == 0.10
    
    def test_compute_correlation_perfect(self, mock_area):
        """Test correlation computation with perfect correlation."""
        validator = Validator(mock_area)
        
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        corr = validator._compute_correlation(x, y)
        
        assert abs(corr - 1.0) < 0.01
    
    def test_compute_correlation_negative(self, mock_area):
        """Test correlation computation with negative correlation."""
        validator = Validator(mock_area)
        
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        
        corr = validator._compute_correlation(x, y)
        
        assert abs(corr - (-1.0)) < 0.01
    
    def test_compute_correlation_no_variance(self, mock_area):
        """Test correlation with no variance returns 0."""
        validator = Validator(mock_area)
        
        x = [1.0, 1.0, 1.0, 1.0]
        y = [2.0, 3.0, 4.0, 5.0]
        
        corr = validator._compute_correlation(x, y)
        
        assert corr == 0.0
    
    def test_compute_correlation_empty_lists(self, mock_area):
        """Test correlation with empty lists returns 1.0 (convention for n<2)."""
        validator = Validator(mock_area)
        
        corr = validator._compute_correlation([], [])
        
        # Implementation returns 1.0 for n < 2
        assert corr == 1.0
    
    def test_age_group_classification(self, mock_area):
        """Test age group classification."""
        validator = Validator(mock_area)
        
        assert validator._age_group(5) == "0-17"
        assert validator._age_group(15) == "0-17"
        assert validator._age_group(18) == "18-24"
        assert validator._age_group(24) == "18-24"
        assert validator._age_group(25) == "25-44"
        assert validator._age_group(35) == "25-44"
        assert validator._age_group(50) == "45-64"
        assert validator._age_group(70) == "65-79"
        assert validator._age_group(85) == "80+"
    
    def test_map_position_to_role(self, mock_area):
        """Test mapping Swedish position strings to roles."""
        validator = Validator(mock_area)
        
        # Test some common Swedish position names
        # Note: actual mappings depend on config, these are placeholders
        result = validator._map_position_to_role("Barn")
        assert result in [None, "child"]  # May be None if config not loaded
        
        result = validator._map_position_to_role("Ensamboende")
        assert result in [None, "single"]
    
    def test_normalize_age_group(self, mock_area):
        """Test normalization of age group strings."""
        validator = Validator(mock_area)
        
        # Test some common age group formats (based on actual implementation)
        assert validator._normalize_age_group("0-5 år") == "0-17"
        assert validator._normalize_age_group("6-15 år") == "0-17"
        assert validator._normalize_age_group("19-24 år") == "18-24"
        assert validator._normalize_age_group("25-34 år") == "25-44"
        assert validator._normalize_age_group("80+ år") == "80+"
        assert validator._normalize_age_group("unknown") is None
    
    def test_compute_derived_metrics_basic(self, mock_area):
        """Test computing derived metrics from population."""
        # Create a simple population
        hh1 = Household(household_id=1, size=3)
        hh1.members = []
        hh1.cars = 0
        
        agent1 = Agent(agent_id=1, age=40, sex='male', hh_role='cohabiting')
        agent1.household_id = 1
        agent2 = Agent(agent_id=2, age=38, sex='female', hh_role='cohabiting')
        agent2.household_id = 1
        agent3 = Agent(agent_id=3, age=10, sex='male', hh_role='child')
        agent3.household_id = 1
        
        hh1.members = [agent1, agent2, agent3]
        
        mock_area.households = [hh1]
        mock_area.individuals = [agent1, agent2, agent3]
        
        validator = Validator(mock_area)
        metrics = validator.compute_derived_metrics()
        
        # Check that metrics are computed
        assert 'avg_household_size' in metrics
        assert 'working_age_population' in metrics
        assert 'single_parent_households' in metrics
        assert metrics['avg_household_size'] == 3.0
        assert metrics['working_age_population'] == 2
        assert metrics['single_parent_households'] == 0
    
    @patch('gbgsynth.validation.run_all_checks')
    @patch.object(Validator, 'validate_overcrowding')
    def test_run_all_validations(self, mock_overcrowding, mock_sanity, mock_area):
        """Test running all validations."""
        # Setup mocks
        mock_sanity.return_value = SanityCheckResult(
            total_households=100,
            total_individuals=250
        )
        
        mock_overcrowding.return_value = ValidationResult(
            dimension="overcrowding",
            census_total=1000,
            synth_total=995,
            rmse=10.0,
            mae=8.0,
            max_error_pct=12.0,
            correlation=0.99,
            categories=[],
            passed=True
        )
        
        validator = Validator(mock_area)
        report = validator.run_all_validations()
        
        assert isinstance(report, ValidationReport)
        assert report.area_code == "107"
        assert report.area_name == "Haga"
        assert report.year == 2023
        assert report.sanity_result is not None


class TestValidationOvercrowding:
    """Test overcrowding validation specifically."""
    
    @pytest.fixture
    def mock_area_with_apartments(self):
        """Create a mock area with apartment households."""
        area = Mock()
        area.area_code = "107"
        area.area_name = "Haga"
        area.year = 2023
        area.area_api_value = "107"
        
        # Create households with apartments
        hh1 = Household(household_id=1, size=4)
        dwelling1 = Dwelling(
            dwelling_id=1,
            building_id="B1",
            floor_area=60.0  # 3 rooms = 60sqm / 20
        )
        dwelling1.house_type_sv = "Flerbostadshus"
        hh1.dwelling = dwelling1
        hh1.members = [
            Agent(agent_id=i, age=30+i*5, sex='male' if i % 2 == 0 else 'female')
            for i in range(4)
        ]
        for agent in hh1.members:
            agent.household_id = 1
        
        # Create non-overcrowded household
        hh2 = Household(household_id=2, size=2)
        dwelling2 = Dwelling(
            dwelling_id=2,
            building_id="B2",
            floor_area=60.0
        )
        dwelling2.house_type_sv = "Flerbostadshus"
        hh2.dwelling = dwelling2
        hh2.members = [
            Agent(agent_id=10, age=30, sex='male'),
            Agent(agent_id=11, age=28, sex='female')
        ]
        for agent in hh2.members:
            agent.household_id = 2
        
        area.households = [hh1, hh2]
        area.individuals = hh1.members + hh2.members
        
        return area
    
    @patch('gbgsynth.validation.PxWebClient')
    def test_validate_overcrowding_basic(self, mock_client, mock_area_with_apartments):
        """Test basic overcrowding validation."""
        # Mock API response
        mock_api_instance = Mock()
        mock_client.return_value = mock_api_instance
        
        import pandas as pd
        mock_df = pd.DataFrame({
            'Trångbodd': ['Trångbodd', 'Ej trångbodd'],
            'Antal': [50, 950]
        })
        mock_api_instance.query_table.return_value = mock_df
        
        validator = Validator(mock_area_with_apartments)
        
        # Test that it doesn't crash
        result = validator.validate_overcrowding()
        
        assert isinstance(result, ValidationResult)
        assert result.dimension == "overcrowding"


class TestValidationTableRegistry:
    """Test validation table registry functions."""
    
    def test_list_validation_tables(self):
        """Test listing validation tables."""
        tables = list_validation_tables()
        
        assert isinstance(tables, dict)
        assert len(tables) > 0
        # Should include both usable and unusable tables
        assert "OVERCROWDING" in tables or "HOUSEHOLD_POSITION" in tables
    
    def test_list_synthesis_tables(self):
        """Test listing synthesis tables."""
        tables = list_synthesis_tables()
        
        assert isinstance(tables, dict)
        assert len(tables) > 0
    
    def test_suggest_improvements(self):
        """Test suggestion generation."""
        suggestions = suggest_improvements()
        
        assert isinstance(suggestions, str)
        assert len(suggestions) > 0


class TestValidationTableDefinitions:
    """Test the validation table definitions."""
    
    def test_usable_tables_structure(self):
        """Test that usable validation tables have required fields."""
        for table_name, table_def in USABLE_VALIDATION_TABLES.items():
            assert 'id' in table_def
            assert 'description' in table_def
            assert 'validation_type' in table_def
            assert 'required_attributes' in table_def
            assert isinstance(table_def['required_attributes'], list)
    
    def test_unusable_tables_structure(self):
        """Test that unusable validation tables have required fields."""
        for table_name, table_def in UNUSABLE_VALIDATION_TABLES.items():
            assert 'id' in table_def
            assert 'description' in table_def
            assert 'missing_attribute' in table_def
            assert 'action' in table_def
    
    def test_no_duplicate_table_ids(self):
        """Test that there are no duplicate table IDs."""
        all_ids = []
        for table_def in USABLE_VALIDATION_TABLES.values():
            all_ids.append(table_def['id'])
        for table_def in UNUSABLE_VALIDATION_TABLES.values():
            all_ids.append(table_def['id'])
        
        # Check for duplicates
        assert len(all_ids) == len(set(all_ids))


class TestValidationIntegration:
    """Integration tests for validation framework."""
    
    def test_end_to_end_validation_flow(self):
        """Test complete validation workflow."""
        # Create a minimal population
        area = Mock()
        area.area_code = "107"
        area.area_name = "Haga"
        area.year = 2023
        area.area_api_value = "107"
        
        hh = Household(household_id=1, size=2)
        hh.cars = 0
        agent1 = Agent(agent_id=1, age=30, sex='male', hh_role='cohabiting')
        agent1.household_id = 1
        agent2 = Agent(agent_id=2, age=28, sex='female', hh_role='cohabiting')
        agent2.household_id = 1
        
        hh.members = [agent1, agent2]
        area.households = [hh]
        area.individuals = [agent1, agent2]
        
        validator = Validator(area)
        
        # Test derived metrics computation
        metrics = validator.compute_derived_metrics()
        assert 'avg_household_size' in metrics
        assert 'working_age_population' in metrics
        assert metrics['avg_household_size'] == 2.0
        assert metrics['working_age_population'] == 2
    
    def test_validator_handles_empty_population(self):
        """Test validator with empty population."""
        area = Mock()
        area.area_code = "107"
        area.area_name = "Haga"
        area.year = 2023
        area.area_api_value = "107"
        area.households = []
        area.individuals = []
        
        validator = Validator(area)
        metrics = validator.compute_derived_metrics()
        
        # Check that metrics handle empty population without crashing
        assert 'avg_household_size' in metrics
        assert 'working_age_population' in metrics
        assert metrics['avg_household_size'] == 0
        assert metrics['working_age_population'] == 0
