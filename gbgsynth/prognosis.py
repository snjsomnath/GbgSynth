"""
Population prognosis scaling for future-year projections.

This module fetches population prognosis data from the Gothenburg PxWeb API
at the mellanområde (intermediate area) level and computes age-specific scale
factors that can be applied to a primary-area synthetic population.

The prognosis data is published at the mellanområde level (36 areas), while
synthesis runs at the primärområde level (96 areas). The mapping between
these two geographic levels is stored in ``config/pri_to_mel.json``, derived
from a spatial join of the official shapefiles.

Workflow
--------
1. Synthesise a base-year population using the normal census data.
2. Fetch the prognosis age distribution for the target year from the
   mellanområde-level API.
3. Compute per-age-group scale factors (target / base).
4. Scale the census marginals accordingly and re-synthesise.

The result is a synthetic population whose age distribution matches the
official prognosis while preserving the household structure constraints
from the synthesiser.

Example
-------
>>> from gbgsynth import GbgSynth
>>> city = GbgSynth(year=2024)
>>> future = city.synthesize_future("Haga", target_year=2030)
>>> print(len(future.individuals))
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from gbgsynth.api_client import PxWebClient, APICache
from gbgsynth.exceptions import APIError

logger = logging.getLogger(__name__)

# Prognosis table path (mellanområde-level, 2021 boundaries)
PROGNOSIS_TABLE_PATH = (
    "Befolkning/Befolkningsprognos/14_PrognosMO21.px"
)
PROGNOSIS_BASE_URL = (
    "https://pxweb.goteborg.se/api/v1/sv/"
    "1. Göteborg och dess delområden/Mellanområden 2021-/"
)

# Available prognosis years (2025–2032 as of spring 2025 forecast)
PROGNOSIS_YEARS = list(range(2025, 2033))

# Path to bundled pri→mel mapping
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
_PRI_TO_MEL_PATH = os.path.join(_CONFIG_DIR, "pri_to_mel.json")

# Age group boundaries used in the census tables
AGE_GROUP_RANGES = {
    "0-17 år": (0, 17),
    "18-24 år": (18, 24),
    "25-44 år": (25, 44),
    "45-64 år": (45, 64),
    "65-79 år": (65, 79),
    "80+ år": (80, 99),
}


def _load_pri_to_mel() -> Dict[str, Dict[str, str]]:
    """Load the primärområde → mellanområde mapping from bundled JSON."""
    with open(_PRI_TO_MEL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["mapping"]


# Module-level cache
_PRI_TO_MEL: Optional[Dict[str, Dict[str, str]]] = None


def get_pri_to_mel() -> Dict[str, Dict[str, str]]:
    """
    Get the primärområde → mellanområde mapping (cached).

    Returns:
        Dict mapping primary area code to
        ``{"mel_code": "33", "mel_name": "Guldheden-Landala",
          "mel_api_value": "33 Guldheden-Landala"}``
    """
    global _PRI_TO_MEL
    if _PRI_TO_MEL is None:
        _PRI_TO_MEL = _load_pri_to_mel()
    return _PRI_TO_MEL


def get_mel_for_pri(pri_code: str) -> Dict[str, str]:
    """
    Look up which mellanområde a primärområde belongs to.

    Args:
        pri_code: Primary area code, e.g. ``"112"``

    Returns:
        Dict with ``mel_code``, ``mel_name``, ``mel_api_value``

    Raises:
        KeyError: If the primary area code is unknown
    """
    mapping = get_pri_to_mel()
    if pri_code not in mapping:
        raise KeyError(
            f"Primary area '{pri_code}' not found in pri→mel mapping. "
            f"Known codes: {sorted(mapping.keys())}"
        )
    return mapping[pri_code]


def get_sibling_pri_codes(pri_code: str) -> List[str]:
    """
    Get all primärområden that share the same mellanområde.

    Args:
        pri_code: Primary area code

    Returns:
        List of primary area codes (including the input code)
    """
    mapping = get_pri_to_mel()
    target_mel = mapping[pri_code]["mel_code"]
    return [code for code, info in mapping.items() if info["mel_code"] == target_mel]


class PrognosisClient:
    """
    Client for fetching population prognosis data from the PxWeb API.

    The prognosis is published at the mellanområde level with single-year
    age detail (0–99 år) for years 2025–2032.
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_dir: str = ".gbgsynth_cache",
        cache_ttl_days: int = 30,
        request_delay: float = 0.2,
    ):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "GbgSynth/0.3.0 Python Population Synthesizer",
            "Accept": "application/json",
        })
        self._request_delay = request_delay
        self._last_request_time = 0.0
        self._cache = APICache(
            cache_dir=cache_dir,
            enabled=cache_enabled,
            ttl_days=cache_ttl_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_prognosis(
        self,
        mel_api_value: str,
        year: int,
    ) -> pd.DataFrame:
        """
        Fetch prognosis data for one mellanområde and one target year.

        Args:
            mel_api_value: Full API value, e.g. ``"33 Guldheden-Landala"``
            year: Prognosis year (2025–2032)

        Returns:
            DataFrame with columns ``["age", "count"]`` where *age* is an
            integer (0–99) and *count* is the projected population.

        Raises:
            APIError: On network / API errors
            ValueError: If the year is outside the available range
        """
        if year not in PROGNOSIS_YEARS:
            raise ValueError(
                f"Prognosis year {year} not available. "
                f"Choose from {PROGNOSIS_YEARS}"
            )

        url = f"{PROGNOSIS_BASE_URL}{PROGNOSIS_TABLE_PATH}"

        # Build age values list: "0 år", "1 år", ..., "99 år"
        age_values = [f"{a} år" for a in range(100)]

        payload = {
            "query": [
                {
                    "code": "Område",
                    "selection": {
                        "filter": "item",
                        "values": [mel_api_value],
                    },
                },
                {
                    "code": "Ålder",
                    "selection": {
                        "filter": "item",
                        "values": age_values,
                    },
                },
                {
                    "code": "Prognosår",
                    "selection": {
                        "filter": "item",
                        "values": [str(year)],
                    },
                },
            ],
            "response": {"format": "json"},
        }

        # Check cache
        cached = self._cache.get("POST", url, payload)
        if cached is not None:
            return self._parse_prognosis(cached)

        # Fetch from API
        import time

        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()

        try:
            response = self._session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            json_data = response.json()
        except requests.RequestException as e:
            raise APIError(
                f"Failed to fetch prognosis for {mel_api_value}, year {year}: {e}",
                url=url,
            )

        # Cache
        desc = f"prognosis:{mel_api_value}:{year}"
        self._cache.set("POST", url, json_data, payload, description=desc)

        return self._parse_prognosis(json_data)

    def fetch_prognosis_pair(
        self,
        mel_api_value: str,
        base_year: int,
        target_year: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch prognosis for both base and target year.

        Returns:
            Tuple of ``(base_df, target_df)``
        """
        base = self.fetch_prognosis(mel_api_value, base_year)
        target = self.fetch_prognosis(mel_api_value, target_year)
        return base, target

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_prognosis(json_data: dict) -> pd.DataFrame:
        """Parse PxWeb JSON into a clean age→count DataFrame."""
        rows = []
        for item in json_data.get("data", []):
            keys = item.get("key", [])
            values = item.get("values", [])
            # keys: [area, age_label, year]  values: [count]
            age_label = keys[1] if len(keys) > 1 else keys[0]
            age = int(age_label.split()[0])  # "42 år" → 42
            count = int(values[0]) if values else 0
            rows.append({"age": age, "count": count})

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame({"age": range(100), "count": 0})
        return df.sort_values("age").reset_index(drop=True)


# ======================================================================
# Scale-factor computation
# ======================================================================


def compute_age_scale_factors(
    base_df: pd.DataFrame,
    target_df: pd.DataFrame,
    age_groups: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """
    Compute per-age-group scale factors from two prognosis snapshots.

    The factor for each age group is ``target_total / base_total``.
    If the base total for a group is zero, the factor defaults to 1.0.

    Args:
        base_df: Prognosis DataFrame for the base year (columns: age, count)
        target_df: Prognosis DataFrame for the target year
        age_groups: Mapping of group label → (min_age, max_age).
                    Defaults to the standard census age groups.

    Returns:
        Dict mapping age group label to scale factor, plus an
        ``"_overall"`` key with the total population ratio.
    """
    if age_groups is None:
        age_groups = AGE_GROUP_RANGES

    factors: Dict[str, float] = {}

    for label, (lo, hi) in age_groups.items():
        base_total = base_df.loc[
            base_df["age"].between(lo, hi), "count"
        ].sum()
        target_total = target_df.loc[
            target_df["age"].between(lo, hi), "count"
        ].sum()

        if base_total > 0:
            factors[label] = target_total / base_total
        else:
            factors[label] = 1.0

    # Overall factor
    base_total_all = base_df["count"].sum()
    target_total_all = target_df["count"].sum()
    factors["_overall"] = (
        target_total_all / base_total_all if base_total_all > 0 else 1.0
    )

    return factors


def compute_single_year_scale_factors(
    base_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> Dict[int, float]:
    """
    Compute per-single-year-of-age scale factors.

    Useful for fine-grained scaling of population marginals.

    Args:
        base_df: Prognosis DataFrame for the base year
        target_df: Prognosis DataFrame for the target year

    Returns:
        Dict mapping age (int) to scale factor
    """
    merged = base_df.merge(
        target_df, on="age", suffixes=("_base", "_target")
    )
    factors = {}
    for _, row in merged.iterrows():
        age = int(row["age"])
        base_val = row["count_base"]
        target_val = row["count_target"]
        factors[age] = target_val / base_val if base_val > 0 else 1.0
    return factors


# ======================================================================
# Marginal scaling
# ======================================================================


def _parse_census_age_range(label: str) -> Tuple[int, int]:
    """
    Parse a census ``Ålder`` label into ``(min_age, max_age)``.

    Examples::

        "0-5 år"   → (0, 5)
        "6-15 år"  → (6, 15)
        "85- år"   → (85, 99)   # open-ended
        "25-44 år" → (25, 44)
    """
    s = label.replace("år", "").strip()
    if "-" in s:
        parts = s.split("-", 1)
        lo = int(parts[0].strip())
        hi_str = parts[1].strip()
        hi = int(hi_str) if hi_str else 99  # open-ended like "85-"
    else:
        # Single age value (unlikely but defensive)
        lo = hi = int(s)
    return lo, hi


def _census_label_to_prognosis_group(
    label: str,
    age_groups: Dict[str, Tuple[int, int]],
) -> Optional[str]:
    """
    Map a census age label to the best-matching prognosis age group.

    The census tables use finer bins (e.g. ``"0-5 år"``, ``"6-15 år"``)
    while the prognosis factors use coarser groups (``"0-17 år"``).
    We find the prognosis group whose range contains the **midpoint**
    of the census bin.  If a census bin spans two prognosis groups
    (e.g. ``"16-18 år"`` straddles ``"0-17"`` and ``"18-24"``), we use
    a weighted average — but for the simple lookup here we return the
    group that covers the midpoint.
    """
    lo, hi = _parse_census_age_range(label)
    mid = (lo + hi) // 2  # integer floor so 75-84 → 79 falls in "65-79"
    for group_label, (g_lo, g_hi) in age_groups.items():
        if g_lo <= mid <= g_hi:
            return group_label
    return None


def scale_population_marginals(
    population_data: pd.DataFrame,
    base_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Scale population marginal counts using single-year prognosis data.

    For each census age bin (e.g. ``"0-5 år"``, ``"16-18 år"``), the
    prognosis counts for every individual year in that range are summed
    for both the base and target years, and the ratio ``target / base``
    is used as the scale factor.  This gives the highest possible
    resolution — no intermediate age-group bucketing is needed.

    The scaled counts are rounded to integers and clamped to ≥ 0.

    Args:
        population_data: Census DataFrame with an ``"Ålder"`` column
            containing age-bin labels and a count column (``"Antal"``
            or the last column).
        base_df: Prognosis DataFrame for the base year with columns
            ``["age", "count"]`` (one row per single year 0–99).
        target_df: Prognosis DataFrame for the target year with the
            same structure.

    Returns:
        New DataFrame with scaled counts.
    """
    df = population_data.copy()
    count_col = "Antal" if "Antal" in df.columns else df.columns[-1]

    # Pre-index prognosis by age for fast lookup
    base_by_age = base_df.set_index("age")["count"]
    target_by_age = target_df.set_index("age")["count"]
    overall_factor = target_by_age.sum() / base_by_age.sum() if base_by_age.sum() > 0 else 1.0

    for idx, row in df.iterrows():
        age_label = row.get("Ålder", "")
        try:
            lo, hi = _parse_census_age_range(age_label)
        except (ValueError, IndexError):
            # Can't parse — fall back to overall factor
            factor = overall_factor
        else:
            base_sum = sum(
                base_by_age.get(a, 0) for a in range(lo, hi + 1)
            )
            target_sum = sum(
                target_by_age.get(a, 0) for a in range(lo, hi + 1)
            )
            factor = target_sum / base_sum if base_sum > 0 else overall_factor

        df.at[idx, count_col] = max(0, round(row[count_col] * factor))

    return df


def scale_household_marginals(
    household_data: pd.DataFrame,
    overall_factor: float,
) -> pd.DataFrame:
    """
    Scale household marginals by the overall population growth factor.

    Since the prognosis doesn't provide household-level projections,
    we scale household counts uniformly by the overall population ratio.

    Args:
        household_data: Census DataFrame with household size × type counts
        overall_factor: The ``_overall`` factor from
            :func:`compute_age_scale_factors`

    Returns:
        New DataFrame with scaled counts
    """
    df = household_data.copy()
    count_col = "Antal" if "Antal" in df.columns else df.columns[-1]
    df[count_col] = (df[count_col] * overall_factor).round().astype(int).clip(lower=0)
    return df


# ======================================================================
# High-level convenience
# ======================================================================


class PrognosisScaler:
    """
    High-level interface for scaling a neighbourhood population to a
    future prognosis year.

    This class orchestrates the full workflow:

    1. Look up which mellanområde the primärområde belongs to
    2. Fetch prognosis data for both base and target years
    3. Compute age-specific scale factors
    4. Apply factors to census marginals
    5. Re-synthesise the population with adjusted marginals

    Example:
        >>> scaler = PrognosisScaler(base_year=2025, target_year=2030)
        >>> factors = scaler.get_scale_factors("107")  # Haga
        >>> print(factors)
        {'0-17 år': 1.05, '18-24 år': 0.98, ..., '_overall': 1.03}
    """

    def __init__(
        self,
        base_year: int = 2025,
        target_year: int = 2030,
        cache_enabled: bool = True,
        cache_dir: str = ".gbgsynth_cache",
    ):
        """
        Initialise the scaler.

        Args:
            base_year: Reference year in the prognosis (default 2025,
                the earliest available prognosis year)
            target_year: Future year to project to (2025–2032)
            cache_enabled: Cache API responses
            cache_dir: Cache directory
        """
        if base_year not in PROGNOSIS_YEARS:
            raise ValueError(
                f"base_year={base_year} not in prognosis range {PROGNOSIS_YEARS}"
            )
        if target_year not in PROGNOSIS_YEARS:
            raise ValueError(
                f"target_year={target_year} not in prognosis range {PROGNOSIS_YEARS}"
            )

        self.base_year = base_year
        self.target_year = target_year
        self._client = PrognosisClient(
            cache_enabled=cache_enabled,
            cache_dir=cache_dir,
        )

        # Cache fetched data per mel area
        self._prognosis_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _ensure_prognosis(self, mel_api_value: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch and cache prognosis data for a mel area."""
        if mel_api_value not in self._prognosis_cache:
            base_df, target_df = self._client.fetch_prognosis_pair(
                mel_api_value, self.base_year, self.target_year
            )
            self._prognosis_cache[mel_api_value] = (base_df, target_df)
        return self._prognosis_cache[mel_api_value]

    def get_scale_factors(
        self, pri_code: str
    ) -> Dict[str, float]:
        """
        Get age-group scale factors for a primary area.

        Args:
            pri_code: Primary area code (e.g. ``"107"``)

        Returns:
            Dict of age-group label → scale factor
        """
        mel_info = get_mel_for_pri(pri_code)
        base_df, target_df = self._ensure_prognosis(mel_info["mel_api_value"])
        return compute_age_scale_factors(base_df, target_df)

    def get_single_year_scale_factors(
        self, pri_code: str
    ) -> Dict[int, float]:
        """
        Get per-single-year-of-age scale factors for a primary area.

        Args:
            pri_code: Primary area code

        Returns:
            Dict of age (int) → scale factor
        """
        mel_info = get_mel_for_pri(pri_code)
        base_df, target_df = self._ensure_prognosis(mel_info["mel_api_value"])
        return compute_single_year_scale_factors(base_df, target_df)

    def get_prognosis(self, pri_code: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get raw prognosis DataFrames for the mellanområde containing
        a given primärområde.

        Args:
            pri_code: Primary area code

        Returns:
            Tuple of ``(base_year_df, target_year_df)`` with columns
            ``["age", "count"]``
        """
        mel_info = get_mel_for_pri(pri_code)
        return self._ensure_prognosis(mel_info["mel_api_value"])

    def scale_marginals(
        self,
        pri_code: str,
        population_data: pd.DataFrame,
        household_data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale census marginals to the target prognosis year.

        Uses single-year prognosis counts to compute per-census-bin
        factors at the highest available resolution.

        Args:
            pri_code: Primary area code
            population_data: Original census population marginals
            household_data: Original census household marginals

        Returns:
            Tuple of ``(scaled_population, scaled_households)``
        """
        mel_info = get_mel_for_pri(pri_code)
        base_df, target_df = self._ensure_prognosis(mel_info["mel_api_value"])

        overall = (
            target_df["count"].sum() / base_df["count"].sum()
            if base_df["count"].sum() > 0 else 1.0
        )

        scaled_pop = scale_population_marginals(
            population_data, base_df, target_df
        )
        scaled_hh = scale_household_marginals(household_data, overall)

        logger.info(
            f"Scaled marginals for {pri_code}: "
            f"overall factor={overall:.3f}, "
            f"target year={self.target_year}"
        )

        return scaled_pop, scaled_hh

    def summary(self, pri_code: str) -> Dict:
        """
        Get a summary of prognosis scaling for a primary area.

        Returns:
            Dict with mel area info, base/target populations,
            scale factors, etc.
        """
        mel_info = get_mel_for_pri(pri_code)
        base_df, target_df = self._ensure_prognosis(mel_info["mel_api_value"])
        factors = compute_age_scale_factors(base_df, target_df)

        return {
            "pri_code": pri_code,
            "mel_code": mel_info["mel_code"],
            "mel_name": mel_info["mel_name"],
            "base_year": self.base_year,
            "target_year": self.target_year,
            "base_population": int(base_df["count"].sum()),
            "target_population": int(target_df["count"].sum()),
            "overall_growth": f"{(factors['_overall'] - 1) * 100:+.1f}%",
            "scale_factors": {
                k: round(v, 4) for k, v in factors.items()
            },
        }
