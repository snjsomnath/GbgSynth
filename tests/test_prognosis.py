"""
Tests for the prognosis scaling module.
"""

import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from gbgsynth.prognosis import (
    PrognosisScaler,
    PrognosisClient,
    get_pri_to_mel,
    get_mel_for_pri,
    get_sibling_pri_codes,
    compute_age_scale_factors,
    compute_single_year_scale_factors,
    scale_population_marginals,
    scale_household_marginals,
    _parse_census_age_range,
    _census_label_to_prognosis_group,
    AGE_GROUP_RANGES,
    PROGNOSIS_YEARS,
)


# ======================================================================
# pri_to_mel mapping tests
# ======================================================================


class TestPriToMelMapping:
    """Tests for the geographic mapping lookup."""

    def test_load_mapping(self):
        mapping = get_pri_to_mel()
        assert isinstance(mapping, dict)
        assert len(mapping) > 90  # 96 primary areas

    def test_known_area(self):
        mel = get_mel_for_pri("107")  # Haga
        assert mel["mel_code"] == "34"
        assert mel["mel_name"] == "Olivedal-Haga-Annedal-Änggården"
        assert mel["mel_api_value"] == "34 Olivedal-Haga-Annedal-Änggården"

    def test_guldheden(self):
        mel = get_mel_for_pri("113")
        assert mel["mel_code"] == "33"
        assert mel["mel_name"] == "Guldheden-Landala"

    def test_unknown_area_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_mel_for_pri("999")

    def test_siblings(self):
        # Haga (107) shares mel area 34 with Olivedal, Annedal, Änggården
        siblings = get_sibling_pri_codes("107")
        assert "107" in siblings
        assert "106" in siblings  # Änggården
        assert "108" in siblings  # Annedal
        assert "109" in siblings  # Olivedal
        assert len(siblings) == 4

    def test_all_areas_have_mel(self):
        """Every primary area code in the mapping should have valid mel info."""
        mapping = get_pri_to_mel()
        for code, info in mapping.items():
            assert "mel_code" in info
            assert "mel_name" in info
            assert "mel_api_value" in info
            assert info["mel_api_value"].startswith(info["mel_code"] + " ")


# ======================================================================
# Scale factor computation tests
# ======================================================================


def _make_prognosis_df(counts):
    """Create a prognosis DataFrame from a list of 100 counts."""
    return pd.DataFrame({"age": range(len(counts)), "count": counts})


class TestScaleFactors:
    """Tests for scale factor computation."""

    def test_identical_distributions(self):
        counts = [100] * 100
        df = _make_prognosis_df(counts)
        factors = compute_age_scale_factors(df, df)
        for key, val in factors.items():
            assert val == pytest.approx(1.0)

    def test_double_population(self):
        base = _make_prognosis_df([100] * 100)
        target = _make_prognosis_df([200] * 100)
        factors = compute_age_scale_factors(base, target)
        assert factors["_overall"] == pytest.approx(2.0)
        for key, val in factors.items():
            assert val == pytest.approx(2.0)

    def test_age_specific_growth(self):
        """Only 0-17 age group grows, rest stays same."""
        base_counts = [100] * 100
        target_counts = [100] * 100
        # Double the 0-17 group
        for i in range(18):
            target_counts[i] = 200
        base = _make_prognosis_df(base_counts)
        target = _make_prognosis_df(target_counts)
        factors = compute_age_scale_factors(base, target)

        assert factors["0-17 år"] == pytest.approx(2.0)
        assert factors["18-24 år"] == pytest.approx(1.0)
        assert factors["25-44 år"] == pytest.approx(1.0)
        assert factors["_overall"] == pytest.approx(1.18)  # 18 extra out of 100

    def test_zero_base_group(self):
        """A group with zero base population should get factor 1.0."""
        base_counts = [0] * 18 + [100] * 82
        target_counts = [50] * 18 + [100] * 82
        base = _make_prognosis_df(base_counts)
        target = _make_prognosis_df(target_counts)
        factors = compute_age_scale_factors(base, target)
        assert factors["0-17 år"] == 1.0  # Can't scale from zero

    def test_single_year_factors(self):
        base = _make_prognosis_df([100] * 100)
        target_counts = [100] * 100
        target_counts[25] = 150
        target = _make_prognosis_df(target_counts)
        factors = compute_single_year_scale_factors(base, target)
        assert factors[25] == pytest.approx(1.5)
        assert factors[0] == pytest.approx(1.0)
        assert factors[99] == pytest.approx(1.0)


# ======================================================================
# Marginal scaling tests
# ======================================================================


class TestMarginalScaling:
    """Tests for applying scale factors to census marginals."""

    def test_scale_population_uniform_double(self):
        """Doubling all prognosis counts should double all census bins."""
        pop_data = pd.DataFrame({
            "Ålder": ["0-5 år", "6-15 år", "85- år"],
            "Kön": ["Män"] * 3,
            "Antal": [100, 200, 50],
        })
        base = _make_prognosis_df([100] * 100)
        target = _make_prognosis_df([200] * 100)  # 2× everywhere
        scaled = scale_population_marginals(pop_data, base, target)
        assert scaled["Antal"].iloc[0] == 200  # 100 * 2.0
        assert scaled["Antal"].iloc[1] == 400  # 200 * 2.0
        assert scaled["Antal"].iloc[2] == 100  # 50 * 2.0

    def test_scale_population_age_specific(self):
        """Different growth rates for young vs old ages."""
        pop_data = pd.DataFrame({
            "Ålder": ["0-5 år", "85- år"],
            "Kön": ["Män", "Män"],
            "Antal": [100, 100],
        })
        base_counts = [100] * 100
        target_counts = [100] * 100
        # Triple ages 0-5
        for a in range(6):
            target_counts[a] = 300
        # Halve ages 85-99
        for a in range(85, 100):
            target_counts[a] = 50
        base = _make_prognosis_df(base_counts)
        target = _make_prognosis_df(target_counts)

        scaled = scale_population_marginals(pop_data, base, target)
        assert scaled["Antal"].iloc[0] == 300  # 100 * 3.0
        assert scaled["Antal"].iloc[1] == 50   # 100 * 0.5

    def test_scale_population_no_negative(self):
        pop_data = pd.DataFrame({
            "Ålder": ["0-5 år"],
            "Kön": ["Män"],
            "Antal": [5],
        })
        base = _make_prognosis_df([100] * 100)
        target = _make_prognosis_df([0] * 100)  # Everything goes to 0
        scaled = scale_population_marginals(pop_data, base, target)
        assert scaled["Antal"].iloc[0] == 0  # Clamped, not negative

    def test_scale_population_straddling_bin(self):
        """'16-18 år' spans ages 16,17,18 — each has its own prognosis count."""
        pop_data = pd.DataFrame({
            "Ålder": ["16-18 år"],
            "Kön": ["Män"],
            "Antal": [300],
        })
        base_counts = [100] * 100
        target_counts = [100] * 100
        # Halve age 16,17; triple age 18
        # bin factor = (50+50+300) / (100+100+100) = 400/300 = 1.333...
        target_counts[16] = 50
        target_counts[17] = 50
        target_counts[18] = 300
        base = _make_prognosis_df(base_counts)
        target = _make_prognosis_df(target_counts)

        scaled = scale_population_marginals(pop_data, base, target)
        assert scaled["Antal"].iloc[0] == 400  # 300 * (400/300) = 400

    def test_scale_households(self):
        hh_data = pd.DataFrame({
            "Hushållsstorlek": ["1 person", "2 personer", "3 personer"],
            "Antal": [100, 80, 40],
        })
        scaled = scale_household_marginals(hh_data, overall_factor=1.1)
        assert scaled["Antal"].iloc[0] == 110
        assert scaled["Antal"].iloc[1] == 88
        assert scaled["Antal"].iloc[2] == 44

    def test_original_not_mutated(self):
        pop_data = pd.DataFrame({
            "Ålder": ["0-5 år"],
            "Kön": ["Män"],
            "Antal": [100],
        })
        original_val = pop_data["Antal"].iloc[0]
        base = _make_prognosis_df([100] * 100)
        target = _make_prognosis_df([200] * 100)
        _ = scale_population_marginals(pop_data, base, target)
        assert pop_data["Antal"].iloc[0] == original_val  # Not mutated


# ======================================================================
# PrognosisScaler tests
# ======================================================================


class TestPrognosisScaler:
    """Tests for the high-level PrognosisScaler class."""

    def test_invalid_base_year(self):
        with pytest.raises(ValueError, match="not in prognosis range"):
            PrognosisScaler(base_year=2020, target_year=2030)

    def test_invalid_target_year(self):
        with pytest.raises(ValueError, match="not in prognosis range"):
            PrognosisScaler(base_year=2025, target_year=2040)

    def test_valid_construction(self):
        scaler = PrognosisScaler(base_year=2025, target_year=2032)
        assert scaler.base_year == 2025
        assert scaler.target_year == 2032


# ======================================================================
# PrognosisClient tests
# ======================================================================


class TestPrognosisClient:
    """Tests for the prognosis API client."""

    def test_parse_prognosis(self):
        """Test parsing a PxWeb-style JSON response."""
        json_data = {
            "data": [
                {"key": ["33 Guldheden-Landala", "0 år", "2025"], "values": ["139"]},
                {"key": ["33 Guldheden-Landala", "1 år", "2025"], "values": ["116"]},
                {"key": ["33 Guldheden-Landala", "2 år", "2025"], "values": ["111"]},
            ]
        }
        df = PrognosisClient._parse_prognosis(json_data)
        assert len(df) == 3
        assert df.iloc[0]["age"] == 0
        assert df.iloc[0]["count"] == 139
        assert df.iloc[1]["age"] == 1
        assert df.iloc[1]["count"] == 116

    def test_parse_empty_response(self):
        df = PrognosisClient._parse_prognosis({"data": []})
        assert len(df) == 100  # Fallback: 0-99 with count 0
        assert df["count"].sum() == 0

    def test_invalid_year_raises(self):
        client = PrognosisClient()
        with pytest.raises(ValueError, match="not available"):
            client.fetch_prognosis("33 Guldheden-Landala", year=2020)


# ======================================================================
# Constants tests
# ======================================================================


class TestConstants:
    """Tests for module constants."""

    def test_prognosis_years_range(self):
        assert 2025 in PROGNOSIS_YEARS
        assert 2032 in PROGNOSIS_YEARS
        assert len(PROGNOSIS_YEARS) == 8

    def test_age_groups_cover_full_range(self):
        """All ages 0-99 should be covered by exactly one group."""
        covered = set()
        for label, (lo, hi) in AGE_GROUP_RANGES.items():
            for age in range(lo, hi + 1):
                assert age not in covered, f"Age {age} covered by multiple groups"
                covered.add(age)


# ======================================================================
# Census age label parsing & cross-label scaling tests
# ======================================================================


class TestCensusAgeMapping:
    """Tests for mapping census age labels to prognosis groups."""

    @pytest.mark.parametrize("label,expected", [
        ("0-5 år", (0, 5)),
        ("6-15 år", (6, 15)),
        ("16-18 år", (16, 18)),
        ("19-24 år", (19, 24)),
        ("25-34 år", (25, 34)),
        ("35-44 år", (35, 44)),
        ("45-54 år", (45, 54)),
        ("55-64 år", (55, 64)),
        ("65-74 år", (65, 74)),
        ("75-84 år", (75, 84)),
        ("85- år", (85, 99)),
    ])
    def test_parse_census_age_range(self, label, expected):
        assert _parse_census_age_range(label) == expected

    def test_group_lookup_children(self):
        # "0-5 år" midpoint=2.5 → "0-17 år"
        assert _census_label_to_prognosis_group("0-5 år", AGE_GROUP_RANGES) == "0-17 år"

    def test_group_lookup_teens(self):
        # "6-15 år" midpoint=10.5 → "0-17 år"
        assert _census_label_to_prognosis_group("6-15 år", AGE_GROUP_RANGES) == "0-17 år"

    def test_group_lookup_straddling(self):
        # "16-18 år" midpoint=17 → "0-17 år" (by midpoint)
        assert _census_label_to_prognosis_group("16-18 år", AGE_GROUP_RANGES) == "0-17 år"

    def test_group_lookup_elderly(self):
        # "85- år" → (85,99), midpoint=92 → "80+ år"
        assert _census_label_to_prognosis_group("85- år", AGE_GROUP_RANGES) == "80+ år"

    def test_group_lookup_working_age(self):
        assert _census_label_to_prognosis_group("45-54 år", AGE_GROUP_RANGES) == "45-64 år"
        assert _census_label_to_prognosis_group("55-64 år", AGE_GROUP_RANGES) == "45-64 år"

    def test_all_census_labels_map(self):
        """Every real census label should map to a prognosis group."""
        census_labels = [
            "0-5 år", "6-15 år", "16-18 år", "19-24 år",
            "25-34 år", "35-44 år", "45-54 år", "55-64 år",
            "65-74 år", "75-84 år", "85- år",
        ]
        for label in census_labels:
            result = _census_label_to_prognosis_group(label, AGE_GROUP_RANGES)
            assert result is not None, f"Census label {label!r} didn't map to any group"
            assert result in AGE_GROUP_RANGES


class TestCrossLabelScaling:
    """Tests for scale_population_marginals with real census-style age labels."""

    def test_census_bins_get_precise_factors(self):
        """Each census bin uses the exact prognosis years it covers."""
        pop = pd.DataFrame({
            "Ålder": ["0-5 år", "6-15 år", "65-74 år", "75-84 år"],
            "Kön": ["Män"] * 4,
            "Antal": [60, 100, 100, 100],
        })
        base_counts = [100] * 100
        target_counts = [100] * 100
        # Young children (0-5): double
        for a in range(6):
            target_counts[a] = 200
        # Older children (6-15): unchanged
        # 65-74: 1.5×
        for a in range(65, 75):
            target_counts[a] = 150
        # 75-84: halve
        for a in range(75, 85):
            target_counts[a] = 50

        base = _make_prognosis_df(base_counts)
        target = _make_prognosis_df(target_counts)
        scaled = scale_population_marginals(pop, base, target)

        assert scaled["Antal"].iloc[0] == 120  # 60 * 2.0
        assert scaled["Antal"].iloc[1] == 100  # 100 * 1.0 (unchanged)
        assert scaled["Antal"].iloc[2] == 150  # 100 * 1.5
        assert scaled["Antal"].iloc[3] == 50   # 100 * 0.5

    def test_open_ended_bin(self):
        """'85- år' (open-ended) maps to ages 85-99."""
        pop = pd.DataFrame({
            "Ålder": ["85- år"],
            "Kön": ["Män"],
            "Antal": [100],
        })
        base_counts = [100] * 100
        target_counts = [100] * 100
        # Triple ages 85-99
        for a in range(85, 100):
            target_counts[a] = 300
        base = _make_prognosis_df(base_counts)
        target = _make_prognosis_df(target_counts)

        scaled = scale_population_marginals(pop, base, target)
        assert scaled["Antal"].iloc[0] == 300  # 100 * 3.0

    def test_non_uniform_within_bin(self):
        """When prognosis varies within a census bin, the average is used."""
        pop = pd.DataFrame({
            "Ålder": ["0-5 år"],
            "Kön": ["Män"],
            "Antal": [600],
        })
        base_counts = [100] * 100
        target_counts = [100] * 100
        # Ages 0-2 triple, ages 3-5 stay the same
        # bin factor = (300+300+300+100+100+100) / (100*6) = 1200/600 = 2.0
        for a in range(3):
            target_counts[a] = 300

        base = _make_prognosis_df(base_counts)
        target = _make_prognosis_df(target_counts)
        scaled = scale_population_marginals(pop, base, target)
        assert scaled["Antal"].iloc[0] == 1200  # 600 * 2.0
