"""
Export / serialisation for synthesised area populations.

Handles CSV output, DataFrame conversion, and format-specific
exporters (e.g. SweLoadSim).
"""

import logging
import os
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from gbgsynth.exceptions import DataNotGeneratedError
from gbgsynth.models import Agent, Household, Dwelling

if TYPE_CHECKING:
    pass  # no extra imports needed yet

logger = logging.getLogger(__name__)

__all__ = ['AreaExporter']


class AreaExporter:
    """Serialisation helper for a single area's population.

    Every method checks that the population has actually been
    generated before attempting to export.

    Parameters
    ----------
    area_code : str
    area_name : str
    year : int
    """

    def __init__(self, area_code: str, area_name: str, year: int):
        self.area_code = area_code
        self.area_name = area_name
        self.year = year

    # ------------------------------------------------------------------
    # Bulk save
    # ------------------------------------------------------------------

    def save(
        self,
        individuals: List[Agent],
        households: List[Household],
        dwellings: List[Dwelling],
        output_dir: str = ".",
        prefix: Optional[str] = None,
    ) -> dict:
        """Save individuals, households, and dwellings to CSV files.

        Returns
        -------
        dict
            ``{'individuals': path, 'households': path,
              'dwellings': path | None}``
        """
        os.makedirs(output_dir, exist_ok=True)

        if prefix is None:
            name_only = (
                self.area_name.split(' ', 1)[-1]
                if ' ' in self.area_name else self.area_name)
            safe_name = name_only.replace(' ', '_')
            prefix = f"{self.area_code}_{safe_name}"

        ind_path = os.path.join(output_dir, f"{prefix}_individuals.csv")
        hh_path = os.path.join(output_dir, f"{prefix}_households.csv")

        self.save_individuals_to_csv(individuals, ind_path)
        self.save_households_to_csv(households, hh_path)

        result = {
            'individuals': ind_path,
            'households': hh_path,
            'dwellings': None,
        }

        if dwellings:
            dw_path = os.path.join(
                output_dir, f"{prefix}_dwellings.csv")
            self.save_dwellings_to_csv(dwellings, dw_path)
            result['dwellings'] = dw_path

        logger.info(f"Saved population to {output_dir}/")
        return result

    # ------------------------------------------------------------------
    # Individual CSV writers
    # ------------------------------------------------------------------

    def save_individuals_to_csv(
        self, individuals: List[Agent], filepath: str
    ) -> None:
        """Save individuals to CSV."""
        df = pd.DataFrame([ind.to_dict() for ind in individuals])
        df['area_code'] = self.area_code
        df['area_name'] = self.area_name
        df['year'] = self.year
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} individuals to {filepath}")

    def save_households_to_csv(
        self, households: List[Household], filepath: str
    ) -> None:
        """Save households to CSV."""
        df = pd.DataFrame([hh.to_dict() for hh in households])
        df['area_code'] = self.area_code
        df['area_name'] = self.area_name
        df['year'] = self.year
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} households to {filepath}")

    def save_dwellings_to_csv(
        self, dwellings: List[Dwelling], filepath: str
    ) -> None:
        """Save dwellings to CSV."""
        if not dwellings:
            logger.warning("No dwellings to save")
            return
        df = pd.DataFrame([dw.to_dict() for dw in dwellings])
        df['area_code'] = self.area_code
        df['area_name'] = self.area_name
        df['year'] = self.year
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} dwellings to {filepath}")

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def to_dataframes(
        self,
        individuals: List[Agent],
        households: List[Household],
        dwellings: List[Dwelling],
    ) -> dict:
        """Convert population to DataFrames.

        Returns
        -------
        dict
            ``{'individuals': DataFrame, 'households': DataFrame,
              'dwellings': DataFrame | None}``
        """
        ind_df = pd.DataFrame([ind.to_dict() for ind in individuals])
        ind_df['area_code'] = self.area_code
        ind_df['area_name'] = self.area_name
        ind_df['year'] = self.year

        hh_df = pd.DataFrame([hh.to_dict() for hh in households])
        hh_df['area_code'] = self.area_code
        hh_df['area_name'] = self.area_name
        hh_df['year'] = self.year

        result = {
            'individuals': ind_df,
            'households': hh_df,
            'dwellings': None,
        }

        if dwellings:
            dw_df = pd.DataFrame([dw.to_dict() for dw in dwellings])
            dw_df['area_code'] = self.area_code
            dw_df['area_name'] = self.area_name
            dw_df['year'] = self.year
            result['dwellings'] = dw_df

        return result

    # ------------------------------------------------------------------
    # Format-specific export
    # ------------------------------------------------------------------

    @staticmethod
    def export_format(area, fmt: str, output_path: str, **kwargs) -> str:
        """Export via a registered exporter (e.g. ``"sweloadsim"``).

        Parameters
        ----------
        area : GbgArea
            The area instance (passed to the exporter).
        fmt : str
            Exporter name.
        output_path : str
            Destination file path.
        **kwargs
            Forwarded to the exporter constructor.

        Returns
        -------
        str
            Path to created output file.
        """
        from gbgsynth.exporters import get_exporter
        from pathlib import Path

        exporter = get_exporter(fmt, **kwargs)
        result_path = exporter.export(area, Path(output_path))
        logger.info(
            f"Exported {len(area.households)} households to "
            f"{result_path} ({fmt} format)")
        return str(result_path)
