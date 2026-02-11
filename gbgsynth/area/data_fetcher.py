"""
Data fetching for area-level census tables.

Encapsulates all PxWeb API calls needed by GbgArea, isolating
network/table logic from synthesis orchestration.
"""

import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ['AreaDataFetcher']


class AreaDataFetcher:
    """Fetches census marginals for a single area from the PxWeb API.

    Parameters
    ----------
    client : PxWebClient
        Pre-configured API client (handles caching, retries, etc.).
    config : Config
        Table-id lookups and translation helpers.
    area_api_value : str
        Full area name as the API expects (e.g. ``"107 Haga"``).
    area_code : str
        Numeric area code (e.g. ``"107"``).
    year : int
        Census year to query.
    """

    def __init__(self, client, config, area_api_value: str,
                 area_code: str, year: int):
        self.client = client
        self.config = config
        self.area_api_value = area_api_value
        self.area_code = area_code
        self.year = year

    # ------------------------------------------------------------------
    # Public convenience
    # ------------------------------------------------------------------

    def fetch_all(self) -> dict:
        """Fetch every table needed by the synthesiser in one call.

        Returns
        -------
        dict
            Keys match the names used by ``GbgArea.generate``.
        """
        return {
            'population': self.fetch_population_data(),
            'household': self.fetch_household_data(),
            'household_position': self.fetch_household_position_data(),
            'income': self.fetch_income_data(),
            'education_level': self.fetch_education_level_data(),
            'income_source': self.fetch_income_source_data(),
            'hh_type_children': self.fetch_hh_type_children_data(),
            'car': self.fetch_car_data(),
            'dwelling': self.fetch_dwelling_data(),
        }

    # ------------------------------------------------------------------
    # Individual table fetchers
    # ------------------------------------------------------------------

    def fetch_population_data(self) -> pd.DataFrame:
        """Fetch population demographics (age/sex/household role)."""
        table_path = self.config.get_table_id('BEFOLKNING_HH')
        try:
            df = self.client.query_all_variables(
                table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} population records")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch population data: {e}")
            return pd.DataFrame({
                'Ålder': ['25-44 år'],
                'Kön': ['Män'],
                'Hushållstyp': ['Ensamstående'],
                'Antal': [100]
            })

    def fetch_household_data(self) -> pd.DataFrame:
        """Fetch household size and type statistics."""
        table_path = self.config.get_table_id('HOUSEHOLD_SIZE')
        try:
            df = self.client.query_all_variables(
                table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} household records")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch household data: {e}")
            return pd.DataFrame({
                'Hushållsstorlek': ['1 person', '2 personer'],
                'Hustyp': ['Flerbostadshus', 'Flerbostadshus'],
                'Antal': [50, 30]
            })

    def fetch_income_data(self) -> Optional[pd.DataFrame]:
        """Fetch income distribution data.

        The income table may use different area naming than population
        tables.  We try the standard name first, then discover the
        correct name from metadata if needed.
        """
        table_path = self.config.get_table_id('INCOME')
        try:
            df = self.client.query_all_variables(
                table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} income records")
            return df
        except Exception as e:
            if '400' in str(e):
                discovered_name = self._discover_income_area_name(table_path)
                if discovered_name and discovered_name != self.area_api_value:
                    try:
                        df = self.client.query_all_variables(
                            table_path, discovered_name, self.year)
                        logger.info(
                            f"Fetched {len(df)} income records "
                            f"(using discovered name '{discovered_name}')")
                        return df
                    except Exception:
                        pass

            for fallback_year in [self.year - 1, self.year - 2, 2023, 2022]:
                try:
                    df = self.client.query_all_variables(
                        table_path, self.area_api_value, fallback_year)
                    logger.info(
                        f"Fetched {len(df)} income records "
                        f"(using {fallback_year} data)")
                    return df
                except Exception:
                    continue
            logger.warning(f"Income data not available: {e}")
            return None

    def fetch_education_level_data(self) -> Optional[pd.DataFrame]:
        """Fetch education level distribution data by age and sex.

        Uses the 23_InkomsterUtbildning_PRI.px table which provides
        population counts (Folkmängd) AND income statistics by
        education level, age group, and sex for adults 18+.
        """
        table_path = self.config.get_table_id('EDUCATION_LEVEL')
        if not table_path:
            logger.warning("EDUCATION_LEVEL table not configured")
            return None

        try:
            df = self.client.query_all_variables(
                table_path, self.area_api_value, self.year)
            mask = (
                (df['Kön'] != 'Båda kön')
                & (df['Utbildningsnivå'] != 'Totalt (alla utbildningsnivåer)')
                & (df['Ålder'] != '18- år')
            )
            result = df[mask].copy()
            logger.info(f"Fetched {len(result)} education level records")
            return result
        except Exception as e:
            for fallback_year in [self.year - 1, self.year - 2, 2023, 2022]:
                try:
                    df = self.client.query_all_variables(
                        table_path, self.area_api_value, fallback_year)
                    mask = (
                        (df['Kön'] != 'Båda kön')
                        & (df['Utbildningsnivå']
                           != 'Totalt (alla utbildningsnivåer)')
                        & (df['Ålder'] != '18- år')
                    )
                    result = df[mask].copy()
                    logger.info(
                        f"Fetched {len(result)} education level records "
                        f"(using {fallback_year} data)")
                    return result
                except Exception:
                    continue
            logger.warning(f"Education level data not available: {e}")
            return None

    def fetch_income_source_data(self) -> Optional[pd.DataFrame]:
        """Fetch primary income source distribution by sex.

        NOTE: This table uses numeric area indices instead of area
        names, so we resolve the index from metadata.
        """
        import requests

        table_path = self.config.get_table_id('INCOME_SOURCE')
        if not table_path:
            logger.warning("INCOME_SOURCE table not configured")
            return None

        try:
            url = f"{self.client.BASE_URL}{table_path}"
            metadata = self.client.fetch_metadata(table_path)

            area_index = None
            for var in metadata.get('variables', []):
                if var['code'] == 'Område':
                    for idx, text in zip(var['values'], var['valueTexts']):
                        if text.startswith(self.area_code + ' '):
                            area_index = idx
                            break
                    break

            if area_index is None:
                logger.warning(
                    f"Could not find area index for {self.area_code} "
                    "in income source table")
                return None

            year_str = str(self.year)
            query = {
                'query': [
                    {'code': 'Område',
                     'selection': {'filter': 'item',
                                   'values': [area_index]}},
                    {'code': 'Kön',
                     'selection': {'filter': 'all', 'values': ['*']}},
                    {'code': 'Huvudsaklig inkomstkälla',
                     'selection': {'filter': 'all', 'values': ['*']}},
                    {'code': 'År',
                     'selection': {'filter': 'item',
                                   'values': [year_str]}},
                ],
                'response': {'format': 'json'},
            }
            resp = requests.post(url, json=query, timeout=30)
            resp.raise_for_status()
            df = self.client._parse_json_response(resp.json())
            logger.info(f"Fetched {len(df)} income source records")
            return df
        except Exception as e:
            for fallback_year in [self.year - 1, self.year - 2, 2023, 2022]:
                try:
                    year_str = str(fallback_year)
                    query['query'][-1]['selection']['values'] = [year_str]
                    resp = requests.post(url, json=query, timeout=30)
                    resp.raise_for_status()
                    df = self.client._parse_json_response(resp.json())
                    logger.info(
                        f"Fetched {len(df)} income source records "
                        f"(using {fallback_year} data)")
                    return df
                except Exception:
                    continue
            logger.warning(f"Income source data not available: {e}")
            return None

    def fetch_household_position_data(self) -> Optional[pd.DataFrame]:
        """Fetch detailed household position data (includes child role)."""
        table_path = self.config.get_table_id('HOUSEHOLD_POSITION')
        if not table_path:
            logger.warning(
                "HOUSEHOLD_POSITION table not configured, "
                "using default roles")
            return None

        try:
            df = self.client.query_all_variables(
                table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} household position records")
            return df
        except Exception as e:
            for fallback_year in [self.year - 1, self.year - 2, 2021, 2020]:
                try:
                    df = self.client.query_all_variables(
                        table_path, self.area_api_value, fallback_year)
                    logger.info(
                        f"Fetched {len(df)} household position records "
                        f"(using {fallback_year} data)")
                    return df
                except Exception:
                    continue
            logger.warning(
                f"Household position data not available: {e}")
            return None

    def fetch_hh_type_children_data(self) -> Optional[pd.DataFrame]:
        """Fetch household type × number of children (0-17) distribution."""
        table_path = self.config.get_table_id('HH_TYPE_CHILDREN')
        if not table_path:
            logger.debug("HH_TYPE_CHILDREN table not configured")
            return None

        try:
            df = self.client.query_all_variables(
                table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} HH type×children records")
            return df
        except Exception as e:
            for fallback_year in [self.year - 1, self.year - 2]:
                try:
                    df = self.client.query_all_variables(
                        table_path, self.area_api_value, fallback_year)
                    logger.info(
                        f"Fetched {len(df)} HH type×children records "
                        f"(using {fallback_year})")
                    return df
                except Exception:
                    continue
            logger.warning(f"HH type×children data not available: {e}")
            return None

    def fetch_car_data(self) -> Optional[pd.DataFrame]:
        """Fetch car ownership statistics."""
        table_path = self.config.get_table_id('CARS')
        try:
            df = self.client.query_table(
                table_path, self.area_api_value, self.year)
            logger.info("Fetched car ownership data")
            return df
        except Exception as e:
            logger.warning(f"Car data not available: {e}")
            return None

    def fetch_dwelling_data(self) -> Optional[pd.DataFrame]:
        """Fetch dwelling size distribution from SCB."""
        table_path = self.config.get_table_id('DWELLING_SIZE')
        try:
            df = self.client.query_all_variables(
                table_path, self.area_api_value, self.year)
            logger.info(f"Fetched {len(df)} dwelling records")
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch dwelling data: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discover_income_area_name(self, table_path: str) -> Optional[str]:
        """Discover the correct area name for the income table.

        Different API tables may use slightly different area name
        spellings.  This method searches the table's available area
        values to find one matching our area code.
        """
        try:
            metadata = self.client.fetch_metadata(table_path)
            for var in metadata.get('variables', []):
                if 'område' in var.get('text', '').lower():
                    values = var.get('values', [])
                    for v in values:
                        if v.startswith(self.area_code + ' '):
                            return v
        except Exception as e:
            logger.debug(f"Could not discover income area name: {e}")
        return None
