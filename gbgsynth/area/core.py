"""
Area-specific population synthesis orchestrator.

This module provides the ``GbgArea`` class that coordinates data
fetching, synthesis, dwelling allocation, validation, and export for
a specific geographic area.  Heavy lifting is delegated to
sub-modules in the ``gbgsynth.area`` package.
"""

import logging
import os
import pandas as pd
from typing import Dict, List, Optional

from gbgsynth.api_client import PxWebClient
from gbgsynth.config import Config
from gbgsynth.exceptions import DataNotGeneratedError, InvalidDataError
from gbgsynth.models import Agent, Household, Dwelling
from gbgsynth.synthesizer import PopulationSynthesizer
from gbgsynth.prognosis import PrognosisScaler, scale_population_marginals

from gbgsynth.area.data_fetcher import AreaDataFetcher
from gbgsynth.area.dwelling_allocator import DwellingAllocator
from gbgsynth.area.exporter import AreaExporter
from gbgsynth.area.marginal_comparator import MarginalComparator

logger = logging.getLogger(__name__)


class GbgArea:
    """
    Represents a specific primary area in Gothenburg.

    Handles data fetching, synthesis orchestration, and export for
    a single geographic area.
    """

    def __init__(
        self,
        area_code: str,
        area_name: str,
        year: int,
        client: Optional[PxWebClient] = None,
        config: Optional[Config] = None,
        area_api_value: Optional[str] = None
    ):
        """
        Initialize area synthesizer.

        Args:
            area_code: Primary area code (e.g., "107")
            area_name: Area name (e.g., "107 Haga")
            year: Year to synthesize
            client: PxWeb client (will create if None)
            config: Configuration (will create if None)
            area_api_value: Full API value for queries (e.g., "107 Haga")
        """
        self.area_code = area_code
        self.area_name = area_name
        self.year = year
        self.client = client or PxWebClient()
        self.config = config or Config()
        # Use API value for queries, fallback to area_name if not provided
        self.area_api_value = area_api_value or area_name

        # Synthesis results
        self.individuals: List[Agent] = []
        self.households: List[Household] = []
        self.dwellings: List[Dwelling] = []
        self._is_generated = False

        # Store original marginals for validation
        self._marginals: dict = {}

        # Synthesis statistics
        self.stats: dict = {}

        # Lazy helpers (created on first use)
        self._fetcher: Optional[AreaDataFetcher] = None
        self._exporter: Optional[AreaExporter] = None

    # ------------------------------------------------------------------
    # Helper accessors
    # ------------------------------------------------------------------

    @property
    def _data_fetcher(self) -> AreaDataFetcher:
        if self._fetcher is None:
            self._fetcher = AreaDataFetcher(
                client=self.client,
                config=self.config,
                area_api_value=self.area_api_value,
                area_code=self.area_code,
                year=self.year,
            )
        return self._fetcher

    @property
    def _area_exporter(self) -> AreaExporter:
        if self._exporter is None:
            self._exporter = AreaExporter(
                area_code=self.area_code,
                area_name=self.area_name,
                year=self.year,
            )
        return self._exporter

    def _comparator(self) -> MarginalComparator:
        return MarginalComparator(
            individuals=self.individuals,
            households=self.households,
            marginals=self._marginals,
            config=self.config,
            area_name=self.area_name,
            year=self.year,
        )

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        status = "generated" if self._is_generated else "not generated"
        if self._is_generated:
            return (
                f"GbgArea('{self.area_name}', year={self.year}, "
                f"pop={len(self.individuals)}, "
                f"hh={len(self.households)})")
        return f"GbgArea('{self.area_name}', year={self.year}, {status})"

    def generate(
        self,
        buildings: Optional[pd.DataFrame] = None,
        allocate_dwellings: bool = True,
        engine: str = 'topdown',
    ) -> 'GbgArea':
        """
        Generate synthetic population for this area.

        Steps:
        1. Fetch required census tables from API (cached)
        2. Run population synthesizer
        3. Allocate dwellings (if enabled)
        4. Store results

        Args:
            buildings: Optional GeoDataFrame with building footprints for
                      spatial linking.
            allocate_dwellings: If True (default), fetch dwelling data and
                              match households to dwelling units.
            engine: Synthesis algorithm — ``'topdown'`` (default),
                ``'ipf'``, or ``'constrained_ipf'``.

        Returns:
            self (for method chaining)

        Example:
            >>> area = city.get_area("Haga")
            >>> area.generate().save("output/")
        """
        logger.info(
            f"Generating population for {self.area_name} ({self.year})")

        fetcher = self._data_fetcher

        # Fetch data (cached by API client)
        population_data = fetcher.fetch_population_data()
        household_data = fetcher.fetch_household_data()
        household_position_data = fetcher.fetch_household_position_data()
        income_data = fetcher.fetch_income_data()
        education_level_data = fetcher.fetch_education_level_data()
        income_source_data = fetcher.fetch_income_source_data()
        hh_type_children_data = fetcher.fetch_hh_type_children_data()
        car_data = fetcher.fetch_car_data()

        # Validate we have required data
        if population_data.empty:
            raise InvalidDataError(
                f"No population data available for "
                f"{self.area_name} ({self.year})",
                field="population_data",
            )
        if household_data.empty:
            raise InvalidDataError(
                f"No household data available for "
                f"{self.area_name} ({self.year})",
                field="household_data",
            )

        # Store marginals for validation
        self._marginals = {
            'population': population_data.copy(),
            'household': household_data.copy(),
            'household_position': (
                household_position_data.copy()
                if household_position_data is not None else None),
            'income': (
                income_data.copy()
                if income_data is not None else None),
            'education_level': (
                education_level_data.copy()
                if education_level_data is not None else None),
            'income_source': (
                income_source_data.copy()
                if income_source_data is not None else None),
            'hh_type_children': (
                hh_type_children_data.copy()
                if hh_type_children_data is not None else None),
        }

        # Synthesize
        synthesizer = PopulationSynthesizer(self.config, engine=engine)
        self.individuals, self.households = synthesizer.synthesize(
            population_data=population_data,
            household_data=household_data,
            income_data=income_data,
            car_data=car_data,
            buildings=buildings,
            household_position_data=household_position_data,
            education_level_data=education_level_data,
            income_source_data=income_source_data,
        )

        # Store synthesis stats
        self.stats = synthesizer.stats

        self._is_generated = True
        logger.info(
            f"Synthesis complete: {len(self.individuals)} individuals, "
            f"{len(self.households)} households")

        # Allocate dwellings if requested
        if allocate_dwellings:
            self._allocate_dwellings()

        return self  # Allow method chaining

    # ------------------------------------------------------------------
    # Dwelling allocation (delegates to DwellingAllocator)
    # ------------------------------------------------------------------

    def _allocate_dwellings(self) -> None:
        """Fetch dwelling data and allocate households to dwelling units."""
        dwelling_data = self._data_fetcher.fetch_dwelling_data()
        allocator = DwellingAllocator(
            config=self.config,
            area_name=self.area_name,
            area_code=self.area_code,
        )
        self.dwellings = allocator.allocate(
            households=self.households,
            dwelling_data=dwelling_data,
        )

    # ------------------------------------------------------------------
    # Save / export  (delegates to AreaExporter)
    # ------------------------------------------------------------------

    def save(self, output_dir: str = ".", prefix: Optional[str] = None) -> dict:
        """
        Save individuals, households, and dwellings to CSV files.

        Creates three files:
        - {prefix}_individuals.csv
        - {prefix}_households.csv
        - {prefix}_dwellings.csv (if dwellings allocated)

        Args:
            output_dir: Directory to save files (default: current directory)
            prefix: Filename prefix (default: "{area_code}_{area_name}")

        Returns:
            Dictionary with paths to saved files:
            {'individuals': '...', 'households': '...', 'dwellings': '...' or None}

        Raises:
            RuntimeError: If generate() hasn't been called yet

        Example:
            >>> area.generate()
            >>> files = area.save()  # Saves to current dir
            >>> files = area.save("output/", prefix="haga_2024")
        """
        if not self._is_generated:
            raise DataNotGeneratedError("saving")

        return self._area_exporter.save(
            individuals=self.individuals,
            households=self.households,
            dwellings=self.dwellings,
            output_dir=output_dir,
            prefix=prefix,
        )

    def save_dwellings_to_csv(self, filepath: str) -> None:
        """Save dwelling data to CSV."""
        if not self._is_generated:
            raise DataNotGeneratedError("saving dwellings")
        self._area_exporter.save_dwellings_to_csv(
            self.dwellings, filepath)

    def save_to_csv(self, filepath: str) -> None:
        """Save the synthetic population (individuals) to CSV."""
        if not self._is_generated:
            raise DataNotGeneratedError("saving individuals")
        self._area_exporter.save_individuals_to_csv(
            self.individuals, filepath)

    def save_households_to_csv(self, filepath: str) -> None:
        """Save household data to CSV."""
        if not self._is_generated:
            raise DataNotGeneratedError("saving households")
        self._area_exporter.save_households_to_csv(
            self.households, filepath)

    def to_dataframes(self) -> dict:
        """
        Get population data as pandas DataFrames.

        Returns:
            Dictionary with 'individuals', 'households', and 'dwellings'
            DataFrames.
        """
        if not self._is_generated:
            raise DataNotGeneratedError("accessing data")
        return self._area_exporter.to_dataframes(
            individuals=self.individuals,
            households=self.households,
            dwellings=self.dwellings,
        )

    @property
    def individuals_df(self) -> pd.DataFrame:
        """Get individuals as a pandas DataFrame."""
        return self.to_dataframes()['individuals']

    @property
    def households_df(self) -> pd.DataFrame:
        """Get households as a pandas DataFrame."""
        return self.to_dataframes()['households']

    def export(self, format: str, output_path: str, **kwargs) -> str:
        """
        Export population to specified format for downstream simulation
        tools.

        Supported formats:
        - "sweloadsim": SweLoadSim household energy simulation (JSON)
        """
        if not self._is_generated:
            raise DataNotGeneratedError("exporting")
        return AreaExporter.export_format(
            self, format, output_path, **kwargs)

    # ------------------------------------------------------------------
    # Validation / comparison (delegates to MarginalComparator)
    # ------------------------------------------------------------------

    def get_summary_statistics(self) -> dict:
        """Get summary statistics for the generated population."""
        if not self._is_generated:
            raise RuntimeError(
                "Must call generate() before getting statistics")
        stats = self._comparator().get_summary_statistics()
        stats['area_code'] = self.area_code
        return stats

    def compare_to_marginals(
        self,
        print_report: bool = True,
        use_logging: bool = False,
    ) -> dict:
        """
        Compare synthesized population against original census marginals.

        Args:
            print_report: If True, outputs a formatted comparison report
            use_logging: If True, uses logging.info instead of print.

        Returns:
            Dictionary containing comparison metrics.
        """
        if not self._is_generated:
            raise RuntimeError(
                "Must call generate() before comparing to marginals")
        if not self._marginals:
            raise RuntimeError(
                "No marginals stored - regenerate with current version")
        return self._comparator().compare(
            print_report=print_report, use_logging=use_logging)

    def log_statistics(
        self, include_marginal_comparison: bool = True
    ) -> None:
        """Log summary statistics and optionally marginal comparison."""
        if not self._is_generated:
            raise RuntimeError(
                "Must call generate() before logging statistics")

        stats = self.get_summary_statistics()

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            f"POPULATION SUMMARY: {self.area_name} ({self.year})")
        logger.info("=" * 60)
        logger.info(
            f"  Total Population:       {stats['total_population']:,}")
        logger.info(
            f"  Total Households:       {stats['total_households']:,}")
        logger.info(
            f"  Avg Household Size:     "
            f"{stats['avg_household_size']:.2f}")
        logger.info(
            f"  Number of Children:     {stats['num_children']:,}")
        logger.info(
            f"  Number of Adults:       {stats['num_adults']:,}")
        logger.info(
            f"  Couple Households:      {stats['num_couples']:,}")
        logger.info(
            f"  Single Parent HH:       {stats['num_single_parent']:,}")
        logger.info(
            f"  Single Person HH:       {stats['num_single_person']:,}")
        logger.info(
            f"  Average Income:         {stats['avg_income']:,.0f} SEK")
        logger.info(
            f"  Total Cars:             {stats['total_cars']:,}")
        logger.info("")

        if self.stats:
            logger.info("Synthesis Statistics:")
            logger.info(
                f"  Method: {self.stats.get('method', 'unknown')}")
            if 'rmse' in self.stats:
                logger.info(f"  RMSE: {self.stats['rmse']:.4f}")
            if 'converged' in self.stats:
                logger.info(f"  Converged: {self.stats['converged']}")
            if 'iterations' in self.stats:
                logger.info(
                    f"  Iterations: {self.stats['iterations']}")
            if 'households_created' in self.stats:
                logger.info(
                    f"  Households: "
                    f"{self.stats['households_created']}")
            if 'individuals_placed' in self.stats:
                logger.info(
                    f"  Individuals Placed: "
                    f"{self.stats['individuals_placed']}")
            logger.info("")

        if include_marginal_comparison:
            self.compare_to_marginals(
                print_report=True, use_logging=True)

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get the marginal comparison as a pandas DataFrame."""
        return self._comparator().get_comparison_dataframe()

    # ------------------------------------------------------------------
    # Prognosis scaling
    # ------------------------------------------------------------------

    def scale_to_year(
        self,
        target_year: int,
        base_year: int = 2025,
        allocate_dwellings: bool = True,
    ) -> 'GbgArea':
        """
        Generate a future-year population scaled by official prognosis data.

        Args:
            target_year: Future year to project to (2025–2032)
            base_year: Reference year in the prognosis (default 2025)
            allocate_dwellings: Whether to allocate dwellings (default True)

        Returns:
            A **new** ``GbgArea`` instance with the scaled population.
            The original area is not modified.
        """
        logger.info(
            f"Scaling {self.area_name} from prognosis "
            f"{base_year}→{target_year}")

        fetcher = self._data_fetcher

        # 1. Fetch original census marginals
        population_data = fetcher.fetch_population_data()
        household_data = fetcher.fetch_household_data()

        # 2. Get scale factors from prognosis
        scaler = PrognosisScaler(
            base_year=base_year,
            target_year=target_year,
        )
        scaled_pop, scaled_hh = scaler.scale_marginals(
            self.area_code, population_data, household_data)

        summary = scaler.summary(self.area_code)
        logger.info(
            f"Prognosis scaling for {self.area_name}: "
            f"{summary['mel_name']} ({summary['base_population']}→"
            f"{summary['target_population']}, "
            f"{summary['overall_growth']})")

        # 3. Create a new area instance for the future year
        future_area = GbgArea(
            area_code=self.area_code,
            area_name=self.area_name,
            year=target_year,
            client=self.client,
            config=self.config,
            area_api_value=self.area_api_value,
        )

        # 4. Fetch supplementary data
        household_position_data = fetcher.fetch_household_position_data()
        income_data = fetcher.fetch_income_data()
        car_data = fetcher.fetch_car_data()

        # 4b. Scale position data
        base_df, target_df = scaler.get_prognosis(self.area_code)
        if household_position_data is not None:
            household_position_data = scale_population_marginals(
                household_position_data, base_df, target_df)

        # 5. Store marginals for validation
        future_area._marginals = {
            'population': scaled_pop.copy(),
            'household': scaled_hh.copy(),
            'household_position': (
                household_position_data.copy()
                if household_position_data is not None else None),
            'income': (
                income_data.copy()
                if income_data is not None else None),
        }

        # 6. Synthesise with scaled marginals
        synthesizer = PopulationSynthesizer(future_area.config)
        future_area.individuals, future_area.households = (
            synthesizer.synthesize(
                population_data=scaled_pop,
                household_data=scaled_hh,
                income_data=income_data,
                car_data=car_data,
                household_position_data=household_position_data,
            ))

        future_area.stats = synthesizer.stats
        future_area.stats['prognosis'] = summary
        future_area._is_generated = True

        logger.info(
            f"Future synthesis complete for {self.area_name} "
            f"({target_year}): "
            f"{len(future_area.individuals)} individuals, "
            f"{len(future_area.households)} households")

        # 7. Allocate dwellings if requested
        if allocate_dwellings:
            future_area._allocate_dwellings()

        return future_area

    def get_prognosis_summary(
        self,
        base_year: int = 2025,
        target_year: int = 2030,
    ) -> dict:
        """Preview the prognosis scaling factors without generating."""
        scaler = PrognosisScaler(
            base_year=base_year,
            target_year=target_year,
        )
        return scaler.summary(self.area_code)


