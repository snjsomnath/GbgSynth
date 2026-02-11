"""
Tests for the plotting module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Check if matplotlib is available
pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

from gbgsynth import plotting
from gbgsynth.models import Agent, Household
from gbgsynth.exceptions import DataNotGeneratedError
from gbgsynth.area import GbgArea


@pytest.fixture
def mock_area():
    """Create a mock GbgArea with generated population for testing."""
    area = Mock(spec=GbgArea)
    area.area_code = "107"
    area.area_name = "107 Haga"
    area.year = 2023
    area._is_generated = True
    
    # Create mock individuals
    individuals = []
    ages = [5, 10, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    sexes = ['male', 'female'] * 7
    for i, (age, sex) in enumerate(zip(ages, sexes)):
        ind = Mock(spec=Agent)
        ind.age = age
        ind.sex = sex
        ind.hh_role = 'child' if age < 18 else 'cohabiting'
        individuals.append(ind)
    
    area.individuals = individuals
    
    # Create mock households
    households = []
    for i in range(5):
        hh = Mock(spec=Household)
        hh.size = i + 1
        hh.cars = i % 3
        hh.assigned_hustyp = ['Småhus', 'Flerbostadshus'][i % 2]
        households.append(hh)
    
    area.households = households
    
    # Mock marginals
    area._marginals = {
        'population': pd.DataFrame({
            'Ålder': ['0-5 år', '6-15 år', '16-18 år'],
            'Antal': [10, 20, 15]
        }),
        'household': pd.DataFrame({
            'Hushållsstorlek': ['1 person', '2 personer', '3 personer'],
            'Antal': [50, 100, 75]
        })
    }
    
    # Mock comparison methods via _comparator()
    mock_comparator = Mock()
    mock_comparator._compare_age_distribution = Mock(return_value={
        'name': 'Age Distribution',
        'comparison': [
            {'category': '0-5 år', 'actual': 10, 'synth': 12, 'diff': 2, 'error_pct': 20},
            {'category': '6-15 år', 'actual': 20, 'synth': 18, 'diff': -2, 'error_pct': -10},
        ]
    })
    
    mock_comparator._compare_household_size_distribution = Mock(return_value={
        'name': 'Household Size',
        'comparison': [
            {'category': '1 person', 'actual': 50, 'synth': 52, 'diff': 2, 'error_pct': 4},
            {'category': '2 personer', 'actual': 100, 'synth': 98, 'diff': -2, 'error_pct': -2},
        ]
    })
    area._comparator = Mock(return_value=mock_comparator)
    
    area.compare_to_marginals = Mock(return_value={
        'sex': {
            'name': 'Sex Distribution',
            'comparison': [
                {'category': 'Male', 'actual': 100, 'synth': 102, 'diff': 2, 'error_pct': 2},
                {'category': 'Female', 'actual': 100, 'synth': 98, 'diff': -2, 'error_pct': -2},
            ]
        },
        'age': {
            'name': 'Age Distribution',
            'comparison': [
                {'category': '0-17', 'actual': 50, 'synth': 48, 'diff': -2, 'error_pct': -4},
                {'category': '18-64', 'actual': 120, 'synth': 122, 'diff': 2, 'error_pct': 1.7},
            ]
        },
        'overall': {
            'rmse': 5.5,
            'mae': 4.2,
            'max_error': 10,
            'correlation': 0.95
        }
    })
    
    return area


class TestPlottingImports:
    """Test that plotting module handles imports correctly."""

    def test_has_matplotlib_flag(self):
        """Test that HAS_MATPLOTLIB is set correctly."""
        assert plotting.HAS_MATPLOTLIB is True

    def test_check_matplotlib_passes(self):
        """Test that _check_matplotlib doesn't raise when matplotlib is available."""
        # Should not raise
        plotting._check_matplotlib()


class TestSetStyle:
    """Tests for set_style function."""

    def test_set_style_ggplot(self):
        """Test setting ggplot style."""
        plotting.set_style("ggplot")

    def test_set_style_classic(self):
        """Test setting classic style."""
        plotting.set_style("classic")


class TestPlotAgeDistribution:
    """Tests for plot_age_distribution function."""

    def test_basic_plot(self, mock_area):
        """Test creating a basic age distribution plot."""
        fig = plotting.plot_age_distribution(mock_area, show_marginals=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_custom_title(self, mock_area):
        """Test plot with custom title."""
        fig = plotting.plot_age_distribution(
            mock_area, 
            show_marginals=False,
            title="Custom Title"
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_raises_if_not_generated(self, mock_area):
        """Test that error is raised if generate() wasn't called."""
        mock_area._is_generated = False
        with pytest.raises(DataNotGeneratedError):
            plotting.plot_age_distribution(mock_area)

    def test_census_overlay_correct_order(self, mock_area):
        """Regression test: census bars must match bins by label, not position.

        The census data can come back in alphabetical order, which puts
        '6-15 år' after '55-64 år'. The plot must still place each bar
        at the correct age-group position.
        """
        import matplotlib.pyplot as plt

        # Provide comparison in ALPHABETICAL order (the real bug scenario)
        mock_comparator = Mock()
        mock_comparator._compare_age_distribution = Mock(return_value={
            'name': 'Age Distribution',
            'comparison': [
                # alphabetical: 0-5, 16-18, 19-24, 25-34, 35-44, 45-54, 55-64, 6-15, 65-74, 75-84, 85-
                {'category': '0-5 år',   'actual': 139, 'synth': 139, 'diff': 0, 'error_pct': 0},
                {'category': '16-18 år', 'actual': 130, 'synth': 130, 'diff': 0, 'error_pct': 0},
                {'category': '19-24 år', 'actual': 215, 'synth': 214, 'diff':-1, 'error_pct':-0.5},
                {'category': '25-34 år', 'actual': 304, 'synth': 303, 'diff':-1, 'error_pct':-0.3},
                {'category': '35-44 år', 'actual': 454, 'synth': 454, 'diff': 0, 'error_pct': 0},
                {'category': '45-54 år', 'actual': 643, 'synth': 641, 'diff':-2, 'error_pct':-0.3},
                {'category': '55-64 år', 'actual': 675, 'synth': 672, 'diff':-3, 'error_pct':-0.4},
                {'category': '6-15 år',  'actual': 338, 'synth': 338, 'diff': 0, 'error_pct': 0},
                {'category': '65-74 år', 'actual': 617, 'synth': 617, 'diff': 0, 'error_pct': 0},
                {'category': '75-84 år', 'actual': 252, 'synth': 252, 'diff': 0, 'error_pct': 0},
                {'category': '85- år',   'actual':  48, 'synth':  48, 'diff': 0, 'error_pct': 0},
            ]
        })
        mock_area._comparator = Mock(return_value=mock_comparator)

        fig = plotting.plot_age_distribution(mock_area, show_marginals=True)
        ax = fig.axes[0]

        # With side-by-side bars, we should have two sets of bar containers
        # (synthesized + census). Extract census bar heights by container.
        containers = ax.containers
        assert len(containers) == 2, "Should have 2 bar groups (synth + census)"
        census_bars = containers[1]  # second group is census
        census_heights = [b.get_height() for b in census_bars]

        # age_labels order: 0-5, 6-15, 16-18, 19-24, 25-34, 35-44,
        #                   45-54, 55-64, 65-74, 75-84, 85+
        # Index 7 = "55-64" must be 675, NOT 338
        assert census_heights[7] == 675, (
            f"Census bar for '55-64' (index 7) should be 675, "
            f"got {census_heights[7]} (likely 6-15 due to alphabetical sort)"
        )

        # Index 1 = "6-15" must be 338
        assert census_heights[1] == 338, (
            f"Census bar for '6-15' (index 1) should be 338, got {census_heights[1]}"
        )

        plt.close(fig)


class TestPlotHouseholdSize:
    """Tests for plot_household_size function."""

    def test_basic_plot(self, mock_area):
        """Test creating a basic household size plot."""
        fig = plotting.plot_household_size(mock_area, show_marginals=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_marginals(self, mock_area):
        """Test plot with marginal comparison."""
        fig = plotting.plot_household_size(mock_area, show_marginals=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotPopulationPyramid:
    """Tests for plot_population_pyramid function."""

    def test_basic_plot(self, mock_area):
        """Test creating a population pyramid."""
        fig = plotting.plot_population_pyramid(mock_area)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_custom_figsize(self, mock_area):
        """Test plot with custom figure size."""
        fig = plotting.plot_population_pyramid(mock_area, figsize=(12, 10))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotCarOwnership:
    """Tests for plot_car_ownership function."""

    def test_plot_by_household_size(self, mock_area):
        """Test car ownership plot grouped by household size."""
        fig = plotting.plot_car_ownership(mock_area, by="household_size")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_by_housing_type(self, mock_area):
        """Test car ownership plot grouped by housing type."""
        fig = plotting.plot_car_ownership(mock_area, by="housing_type")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_invalid_grouping(self, mock_area):
        """Test that invalid grouping raises error."""
        # Should still work but use default behavior
        fig = plotting.plot_car_ownership(mock_area, by="household_size")
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotMarginalComparison:
    """Tests for plot_marginal_comparison function."""

    def test_basic_comparison(self, mock_area):
        """Test creating a marginal comparison plot."""
        fig = plotting.plot_marginal_comparison(mock_area)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_specific_dimensions(self, mock_area):
        """Test plotting specific dimensions."""
        fig = plotting.plot_marginal_comparison(mock_area, dimensions=['sex'])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestCompareAreas:
    """Tests for compare_areas function."""

    def test_compare_population(self, mock_area):
        """Test comparing areas by population."""
        areas = [mock_area, mock_area]  # Use same mock twice
        fig = plotting.compare_areas(areas, metric="population")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_compare_households(self, mock_area):
        """Test comparing areas by households."""
        areas = [mock_area, mock_area]
        fig = plotting.compare_areas(areas, metric="households")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_compare_cars_per_capita(self, mock_area):
        """Test comparing areas by cars per capita."""
        areas = [mock_area, mock_area]
        fig = plotting.compare_areas(areas, metric="cars_per_capita")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_invalid_metric(self, mock_area):
        """Test that invalid metric raises error."""
        areas = [mock_area]
        with pytest.raises(ValueError, match="Unknown metric"):
            plotting.compare_areas(areas, metric="invalid_metric")


class TestSaveAllPlots:
    """Tests for save_all_plots function."""

    def test_save_all_plots(self, mock_area, tmp_path):
        """Test saving all plots to directory."""
        output_dir = str(tmp_path / "plots")
        saved = plotting.save_all_plots(mock_area, output_dir=output_dir)
        
        # Should have saved some plots
        assert len(saved) > 0
        
        # Check files exist
        import os
        for filepath in saved:
            assert os.path.exists(filepath)

    def test_save_with_different_format(self, mock_area, tmp_path):
        """Test saving plots in different format."""
        output_dir = str(tmp_path / "plots_pdf")
        saved = plotting.save_all_plots(
            mock_area, 
            output_dir=output_dir,
            format="pdf"
        )
        
        # Check file extensions
        for filepath in saved:
            assert filepath.endswith(".pdf")


class TestErrorAnalysisPlot:
    """Tests for plot_error_analysis function."""

    def test_basic_error_analysis(self, mock_area):
        """Test basic error analysis plot creation."""
        fig = plotting.plot_error_analysis(mock_area)
        assert fig is not None
        plt.close(fig)

    def test_error_analysis_with_custom_title(self, mock_area):
        """Test error analysis with custom title."""
        fig = plotting.plot_error_analysis(mock_area, title="Custom Error Analysis")
        assert fig is not None
        plt.close(fig)

    def test_error_analysis_requires_generation(self, mock_area):
        """Test that error analysis requires generated population."""
        mock_area._is_generated = False
        with pytest.raises(DataNotGeneratedError):
            plotting.plot_error_analysis(mock_area)


class TestScatterComparisonPlot:
    """Tests for plot_scatter_comparison function."""

    def test_basic_scatter_comparison(self, mock_area):
        """Test basic scatter comparison plot creation."""
        fig = plotting.plot_scatter_comparison(mock_area)
        assert fig is not None
        plt.close(fig)

    def test_scatter_comparison_with_custom_title(self, mock_area):
        """Test scatter comparison with custom title."""
        fig = plotting.plot_scatter_comparison(mock_area, title="Custom Scatter")
        assert fig is not None
        plt.close(fig)

    def test_scatter_comparison_requires_generation(self, mock_area):
        """Test that scatter comparison requires generated population."""
        mock_area._is_generated = False
        with pytest.raises(DataNotGeneratedError):
            plotting.plot_scatter_comparison(mock_area)
