"""
Characterisation tests for SweLoadSim exporter — capturing current
behaviour before the API refactor.

Tests cover:
- SweLoadSimExporter instantiation and config
- HeatingConfig sampling
- BuildingEnvelopeConfig (era sampling, U-values, geometry)
- _convert_household (housing mapping, status mapping, age bins)
- _calculate_summary
- Full export round-trip (JSON structure)
"""

import json
import math
import random
import pytest
from pathlib import Path

from gbgsynth.models import Agent, Household
from gbgsynth.exporters.sweloadsim import (
    SweLoadSimExporter,
    SweLoadSimConfig,
    HeatingConfig,
    EVConfig,
    SolarConfig,
    BatteryConfig,
    BuildingEnvelopeConfig,
)
from gbgsynth.exporters import get_exporter, list_exporters


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def villa_household():
    """Villa household with 2 adults and 1 child."""
    hh = Household(
        household_id=1,
        size=3,
        house_type="detached_house",
        cars=2,
    )
    hh.add_member(Agent(agent_id=1, age=42, sex="male", status="employed", income_decile=8))
    hh.add_member(Agent(agent_id=2, age=40, sex="female", status="part_time", income_decile=8))
    hh.add_member(Agent(agent_id=3, age=10, sex="male"))
    hh.floor_area = 140.0
    return hh


@pytest.fixture
def apartment_household():
    """Apartment household with 1 retiree."""
    hh = Household(
        household_id=2,
        size=1,
        house_type="apartment",
        cars=0,
    )
    hh.add_member(Agent(agent_id=4, age=72, sex="female", status="retired", income_decile=4))
    hh.floor_area = 55.0
    return hh


@pytest.fixture
def deterministic_config():
    """Config with seed for reproducibility."""
    return SweLoadSimConfig(seed=42)


# ── Registry ──────────────────────────────────────────────────────────────────

class TestExporterRegistry:
    """Tests for the exporter registry."""

    def test_get_exporter_sweloadsim(self):
        exporter = get_exporter("sweloadsim")
        assert isinstance(exporter, SweLoadSimExporter)

    def test_get_exporter_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown exporter"):
            get_exporter("nonexistent")

    def test_list_exporters(self):
        names = list_exporters()
        assert "sweloadsim" in names


# ── HeatingConfig ─────────────────────────────────────────────────────────────

class TestHeatingConfig:
    """Tests for heating system sampling."""

    def test_apartment_sampling_distribution(self):
        """Apartment heating should be dominated by district heating."""
        cfg = HeatingConfig()
        rng = random.Random(42)
        results = [cfg.sample_apartment(rng) for _ in range(1000)]

        district_pct = results.count("DISTRICT_HEATING") / len(results)
        assert district_pct > 0.80, f"Expected >80% district, got {district_pct:.1%}"

    def test_villa_sampling_diversity(self):
        """Villa heating should include heat pump, district, and others."""
        cfg = HeatingConfig()
        rng = random.Random(42)
        results = set(cfg.sample_villa(rng) for _ in range(1000))

        assert "HEAT_PUMP" in results
        assert "DISTRICT_HEATING" in results
        assert "DIRECT_ELECTRIC" in results

    def test_heating_values_are_valid_strings(self):
        """All sampled values should be valid schema heating type strings."""
        valid = {"DISTRICT_HEATING", "HEAT_PUMP", "DIRECT_ELECTRIC", "WOOD_PELLET"}
        cfg = HeatingConfig()
        rng = random.Random(42)

        for _ in range(500):
            assert cfg.sample_apartment(rng) in valid
            assert cfg.sample_villa(rng) in valid


# ── BuildingEnvelopeConfig ────────────────────────────────────────────────────

class TestBuildingEnvelopeConfig:
    """Tests for building envelope assumptions."""

    def test_sample_era_returns_valid_string(self):
        cfg = BuildingEnvelopeConfig()
        rng = random.Random(42)
        valid_eras = {"pre_1960", "1960_1975", "1976_1990", "1991_2010", "post_2010"}

        for _ in range(200):
            era = cfg.sample_era(rng, "VILLA")
            assert era in valid_eras, f"Invalid era: {era}"
            era = cfg.sample_era(rng, "APARTMENT")
            assert era in valid_eras, f"Invalid era: {era}"

    def test_u_values_structure(self):
        cfg = BuildingEnvelopeConfig()
        for era in ["pre_1960", "1960_1975", "1976_1990", "1991_2010", "post_2010"]:
            for housing in ["VILLA", "APARTMENT"]:
                u = cfg.get_u_values(era, housing)
                assert "u_walls" in u
                assert "u_roof" in u
                assert "u_floor" in u
                assert "u_windows" in u
                assert all(v > 0 for v in u.values())

    def test_u_values_decrease_with_era(self):
        """Newer eras should have lower (better) U-values."""
        cfg = BuildingEnvelopeConfig()
        eras = ["pre_1960", "1960_1975", "1976_1990", "1991_2010", "post_2010"]
        for housing in ["VILLA", "APARTMENT"]:
            u_walls_prev = float("inf")
            for era in eras:
                u = cfg.get_u_values(era, housing)
                assert u["u_walls"] <= u_walls_prev, f"U-values not decreasing: {era}"
                u_walls_prev = u["u_walls"]

    def test_ventilation_params(self):
        cfg = BuildingEnvelopeConfig()
        for era in ["pre_1960", "1960_1975", "1976_1990", "1991_2010", "post_2010"]:
            v = cfg.get_ventilation_params(era)
            assert "ach_infiltration" in v
            assert "ach_ventilation" in v
            assert "heat_recovery_efficiency" in v

    def test_heat_recovery_modern_eras(self):
        """Post-1990 eras should have heat recovery."""
        cfg = BuildingEnvelopeConfig()
        for era in ["1991_2010", "post_2010"]:
            v = cfg.get_ventilation_params(era)
            assert v["heat_recovery_efficiency"] > 0

    def test_no_heat_recovery_old_eras(self):
        """Pre-1990 eras should not have heat recovery."""
        cfg = BuildingEnvelopeConfig()
        for era in ["pre_1960", "1960_1975", "1976_1990"]:
            v = cfg.get_ventilation_params(era)
            assert v["heat_recovery_efficiency"] == 0.0


# ── Household Conversion ─────────────────────────────────────────────────────

class TestHouseholdConversion:
    """Tests for _convert_household internal method."""

    def test_villa_mapping(self, villa_household, deterministic_config):
        exporter = SweLoadSimExporter(config=deterministic_config)
        result = exporter._convert_household(villa_household)

        assert result["housing_type"] == "VILLA"
        assert result["area_m2"] == 140.0
        assert result["household_id"] == 1
        assert result["num_cars"] == 2
        assert result["income_decile"] == 8

    def test_apartment_mapping(self, apartment_household, deterministic_config):
        exporter = SweLoadSimExporter(config=deterministic_config)
        result = exporter._convert_household(apartment_household)

        assert result["housing_type"] == "APARTMENT"
        assert result["area_m2"] == 55.0

    def test_members_conversion(self, villa_household, deterministic_config):
        exporter = SweLoadSimExporter(config=deterministic_config)
        result = exporter._convert_household(villa_household)

        assert len(result["members"]) == 3
        m0 = result["members"][0]
        assert m0["age_group"] == "30-50"
        assert m0["age_exact"] == 42
        assert m0["sex"] == "male"
        assert m0["status"] == "FULL_TIME"

    def test_child_status_mapping(self, villa_household, deterministic_config):
        exporter = SweLoadSimExporter(config=deterministic_config)
        result = exporter._convert_household(villa_household)
        child = result["members"][2]  # 10-year-old
        assert child["age_group"] == "7-12"
        assert child["status"] == "STUDENT"  # Schema: children mapped to STUDENT

    def test_retired_status_mapping(self, apartment_household, deterministic_config):
        exporter = SweLoadSimExporter(config=deterministic_config)
        result = exporter._convert_household(apartment_household)
        assert result["members"][0]["status"] == "RETIRED"

    def test_building_era_present(self, villa_household, deterministic_config):
        exporter = SweLoadSimExporter(config=deterministic_config)
        result = exporter._convert_household(villa_household)
        valid_eras = {"pre_1960", "1960_1975", "1976_1990", "1991_2010", "post_2010"}
        assert result["building_era"] in valid_eras

    def test_envelope_keys_present(self, villa_household, deterministic_config):
        exporter = SweLoadSimExporter(config=deterministic_config)
        result = exporter._convert_household(villa_household)

        env = result["envelope"]
        required_keys = {
            "u_walls", "u_roof", "u_floor", "u_windows",
            "wall_area_m2", "roof_area_m2", "floor_area_ground_m2",
            "window_area_m2", "window_to_floor_ratio", "building_height_m",
            "ach_infiltration", "ach_ventilation", "heat_recovery_efficiency",
        }
        assert required_keys.issubset(env.keys()), f"Missing: {required_keys - env.keys()}"

    def test_villa_geometry(self, villa_household, deterministic_config):
        """Villa geometry should follow 2-storey square plan assumptions."""
        exporter = SweLoadSimExporter(config=deterministic_config)
        result = exporter._convert_household(villa_household)
        env = result["envelope"]

        floor_area = 140.0
        footprint = floor_area / 2
        side = math.sqrt(footprint)
        height = 2.5 * 2

        assert env["building_height_m"] == 5.0
        assert env["num_stories"] == 2
        assert abs(env["wall_area_m2"] - 4 * side * height) < 0.5
        assert abs(env["window_area_m2"] - floor_area * 0.15) < 0.5

    def test_education_and_income_source_in_member(self, deterministic_config):
        """Members with education/income_source should include them in output."""
        hh = Household(household_id=99, size=1)
        agent = Agent(agent_id=99, age=35, sex="male", status="employed",
                      income_decile=7, education="post_secondary",
                      income_source="work")
        agent.income = 450000
        hh.add_member(agent)

        exporter = SweLoadSimExporter(config=deterministic_config)
        result = exporter._convert_household(hh)
        member = result["members"][0]

        assert member["education_level"] == "post_secondary"
        assert member["income_source"] == "work"
        assert member["income_sek"] == 450000

    def test_member_without_enrichment_fields(self, deterministic_config):
        """Members without education/income_source should omit those keys."""
        hh = Household(household_id=99, size=1)
        agent = Agent(agent_id=99, age=10, sex="female")
        hh.add_member(agent)

        exporter = SweLoadSimExporter(config=deterministic_config)
        result = exporter._convert_household(hh)
        member = result["members"][0]

        assert "education_level" not in member or member.get("education_level") is None
        assert "income_source" not in member
        assert "income_sek" not in member


# ── Config Presets ────────────────────────────────────────────────────────────

class TestConfigPresets:
    """Tests for SweLoadSimConfig preset methods."""

    def test_swedish_2024(self):
        cfg = SweLoadSimConfig.swedish_2024()
        assert cfg.ev.base_probability == 0.30
        assert cfg.heating.apartment_district == 0.90

    def test_future_2030(self):
        cfg = SweLoadSimConfig.future_2030()
        assert cfg.ev.base_probability == 0.50

    def test_future_2035(self):
        cfg = SweLoadSimConfig.future_2035()
        assert cfg.ev.base_probability == 0.70

    def test_future_2040(self):
        cfg = SweLoadSimConfig.future_2040()
        assert cfg.ev.base_probability == 0.85

    def test_for_year_interpolation(self):
        """for_year() should pick closest preset."""
        cfg_2024 = SweLoadSimConfig.for_year(2024)
        cfg_2032 = SweLoadSimConfig.for_year(2032)
        cfg_2040 = SweLoadSimConfig.for_year(2040)

        assert cfg_2024.ev.base_probability == 0.30
        assert cfg_2032.ev.base_probability == 0.50  # Closest to 2030
        assert cfg_2040.ev.base_probability == 0.85


# ── Summary ───────────────────────────────────────────────────────────────────

class TestSummaryCalculation:
    """Tests for _calculate_summary."""

    def test_summary_keys(self, deterministic_config):
        exporter = SweLoadSimExporter(config=deterministic_config)
        households = [
            {"heating_type": "HEAT_PUMP", "housing_type": "VILLA",
             "has_ev": True, "solar_pv_kw": 8.0, "battery_kwh": 10.0,
             "has_summerhouse": False, "building_era": "1976_1990"},
            {"heating_type": "DISTRICT", "housing_type": "APARTMENT",
             "has_ev": False, "solar_pv_kw": None, "battery_kwh": None,
             "has_summerhouse": True, "building_era": "1960_1975"},
        ]
        summary = exporter._calculate_summary(households)

        assert "total_households" in summary
        assert "heating_distribution" in summary
        assert "housing_distribution" in summary
        assert "era_distribution" in summary
        assert "ev_adoption_rate" in summary
        assert "solar_adoption_rate" in summary
        assert "battery_adoption_rate" in summary

    def test_summary_percentages(self, deterministic_config):
        exporter = SweLoadSimExporter(config=deterministic_config)
        households = [
            {"heating_type": "HEAT_PUMP", "housing_type": "VILLA",
             "has_ev": True, "solar_pv_kw": None, "battery_kwh": None,
             "has_summerhouse": False, "building_era": "post_2010"},
        ] * 4 + [
            {"heating_type": "DISTRICT_HEATING", "housing_type": "APARTMENT",
             "has_ev": False, "solar_pv_kw": None, "battery_kwh": None,
             "has_summerhouse": False, "building_era": "pre_1960"},
        ] * 6
        summary = exporter._calculate_summary(households)

        assert summary["heating_distribution"]["HEAT_PUMP"] == 40.0
        assert summary["heating_distribution"]["DISTRICT_HEATING"] == 60.0
        assert summary["ev_adoption_rate"] == 40.0

    def test_empty_households(self, deterministic_config):
        exporter = SweLoadSimExporter(config=deterministic_config)
        summary = exporter._calculate_summary([])
        assert summary == {}


# ── Age Binning ───────────────────────────────────────────────────────────────

class TestAgeBinning:
    """Tests for _age_to_bin mapping."""

    def test_age_bins(self):
        exporter = SweLoadSimExporter()
        assert exporter._age_to_bin(0) == "0-6"
        assert exporter._age_to_bin(6) == "0-6"
        assert exporter._age_to_bin(7) == "7-12"
        assert exporter._age_to_bin(12) == "7-12"
        assert exporter._age_to_bin(13) == "13-17"
        assert exporter._age_to_bin(17) == "13-17"
        assert exporter._age_to_bin(18) == "18-30"
        assert exporter._age_to_bin(30) == "18-30"
        assert exporter._age_to_bin(31) == "30-50"
        assert exporter._age_to_bin(50) == "30-50"
        assert exporter._age_to_bin(51) == "50-65"
        assert exporter._age_to_bin(65) == "50-65"
        assert exporter._age_to_bin(66) == "65+"
        assert exporter._age_to_bin(99) == "65+"


# ── Housing Type Mapping ─────────────────────────────────────────────────────

class TestHousingTypeMapping:
    """Tests for HOUSING_MAP coverage."""

    def test_all_known_types(self):
        mapping = SweLoadSimExporter.HOUSING_MAP
        assert mapping["apartment"] == "APARTMENT"
        assert mapping["detached_house"] == "VILLA"
        assert mapping["terraced_house"] == "VILLA"
        assert mapping["semi_detached"] == "VILLA"
        assert mapping["other"] == "APARTMENT"

    def test_status_map_coverage(self):
        mapping = SweLoadSimExporter.STATUS_MAP
        assert mapping["employed"] == "FULL_TIME"
        assert mapping["part_time"] == "PART_TIME"
        assert mapping["student"] == "STUDENT"
        assert mapping["retired"] == "RETIRED"
        assert mapping["child"] == "STUDENT"          # Schema: no CHILD enum
        assert mapping["unemployed"] == "FULL_TIME"    # Schema: no HOME enum
        assert mapping["parental_leave"] == "FULL_TIME" # Schema: no HOME enum
        assert mapping[None] == "FULL_TIME"
