"""
SweLoadSim exporter for GbgSynth synthetic populations.

Converts GbgSynth population data to SweLoadSim-compatible JSON format
for household electricity load simulation.

Features:
- Configurable heating system distribution
- EV ownership sampling with income/housing modifiers
- Solar PV system allocation
- Battery storage allocation
- Summer house ownership inference
- Regional presets (Gothenburg, Stockholm, rural, future scenarios)

Usage:
    >>> from gbgsynth.exporters.sweloadsim import SweLoadSimExporter, SweLoadSimConfig
    >>> 
    >>> # Use defaults
    >>> exporter = SweLoadSimExporter()
    >>> exporter.export(area, "output.json")
    >>> 
    >>> # Use future scenario preset
    >>> config = SweLoadSimConfig.future_2030()
    >>> exporter = SweLoadSimExporter(config=config)
    >>> 
    >>> # Custom configuration
    >>> config = SweLoadSimConfig(
    ...     ev=EVConfig(base_probability=0.5),
    ...     solar=SolarConfig(villa_probability=0.25),
    ... )
"""

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseExporter

if TYPE_CHECKING:
    from gbgsynth.area import GbgArea
    from gbgsynth.models import Household, Agent


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class HeatingConfig:
    """
    Heating system probability distribution by housing type.
    
    Probabilities within each housing type should sum to 1.0.
    Based on Swedish Energy Agency statistics and regional data.
    
    Attributes:
        apartment_district: P(district heating | apartment)
        apartment_heat_pump: P(heat pump | apartment)
        apartment_electric: P(direct electric | apartment)
        villa_district: P(district heating | villa)
        villa_heat_pump: P(heat pump | villa)
        villa_electric: P(direct electric | villa)
        villa_wood: P(wood/pellet heating | villa)
        villa_gas: P(gas heating | villa)
    """
    
    # Apartment heating probabilities
    apartment_district: float = 0.90
    apartment_heat_pump: float = 0.05
    apartment_electric: float = 0.05
    
    # Villa heating probabilities
    villa_district: float = 0.10
    villa_heat_pump: float = 0.55
    villa_electric: float = 0.20
    villa_wood: float = 0.10
    villa_gas: float = 0.05
    
    def sample_apartment(self, rng: random.Random) -> str:
        """Sample heating system for an apartment."""
        return self._sample(rng, [
            (self.apartment_district, "DISTRICT_HEATING"),
            (self.apartment_heat_pump, "HEAT_PUMP"),
            (self.apartment_electric, "DIRECT_ELECTRIC"),
        ])
    
    def sample_villa(self, rng: random.Random) -> str:
        """Sample heating system for a villa."""
        return self._sample(rng, [
            (self.villa_district, "DISTRICT_HEATING"),
            (self.villa_heat_pump, "HEAT_PUMP"),
            (self.villa_electric, "DIRECT_ELECTRIC"),
            (self.villa_wood, "WOOD_PELLET"),
            (self.villa_gas, "DIRECT_ELECTRIC"),  # No gas in schema → resistive fallback
        ])
    
    def _sample(self, rng: random.Random, options: List[tuple]) -> str:
        """Weighted random sample from options."""
        r = rng.random()
        cumulative = 0.0
        for prob, value in options:
            cumulative += prob
            if r < cumulative:
                return value
        return options[-1][1]


@dataclass
class EVConfig:
    """
    Electric vehicle ownership probability configuration.
    
    EV probability is calculated as:
        P(EV) = base_probability 
                × income_multiplier[decile] 
                × housing_multiplier[type]
                × multi_car_multiplier (if cars > 1)
    
    Attributes:
        base_probability: Base P(EV) for car-owning households
        income_multipliers: Multiplier by income decile (1-10)
        housing_multipliers: Multiplier by housing type (charging access)
        multi_car_multiplier: Multiplier for households with 2+ cars
    """
    
    base_probability: float = 0.30
    
    income_multipliers: Dict[int, float] = field(default_factory=lambda: {
        1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6, 5: 0.7,
        6: 0.9, 7: 1.1, 8: 1.3, 9: 1.5, 10: 1.8,
    })
    
    housing_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "VILLA": 1.4,       # Easy home charging
        "APARTMENT": 0.6,   # Harder to charge at home
    })
    
    multi_car_multiplier: float = 1.3
    
    def sample(self, rng: random.Random, cars: int, income_decile: int, 
               housing_type: str) -> bool:
        """
        Sample whether household has an EV.
        
        Args:
            rng: Random number generator
            cars: Number of cars in household
            income_decile: Household income decile (1-10)
            housing_type: "VILLA" or "APARTMENT"
            
        Returns:
            True if household has an EV
        """
        if cars == 0:
            return False
        
        prob = self.base_probability
        prob *= self.income_multipliers.get(income_decile, 1.0)
        prob *= self.housing_multipliers.get(housing_type, 1.0)
        
        if cars > 1:
            prob *= self.multi_car_multiplier
        
        # Cap at 95%
        prob = min(prob, 0.95)
        
        return rng.random() < prob


@dataclass
class SolarConfig:
    """
    Solar PV system probability and sizing configuration.
    
    Attributes:
        villa_probability: P(solar PV | villa)
        apartment_probability: P(solar PV | apartment) - usually 0
        income_multipliers: Probability multiplier by income decile
        size_kw_mean: Mean system size in kW
        size_kw_std: Standard deviation of system size
        size_min_kw: Minimum system size
        size_max_kw: Maximum system size
    """
    
    villa_probability: float = 0.15
    apartment_probability: float = 0.02  # Rare, usually building-level
    
    income_multipliers: Dict[int, float] = field(default_factory=lambda: {
        1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6, 5: 0.7,
        6: 0.9, 7: 1.1, 8: 1.3, 9: 1.6, 10: 2.0,
    })
    
    # System sizing (kW)
    size_kw_mean: float = 8.0
    size_kw_std: float = 3.0
    size_min_kw: float = 3.0
    size_max_kw: float = 20.0
    
    def sample(self, rng: random.Random, housing_type: str, 
               income_decile: int) -> Optional[float]:
        """
        Sample whether household has solar PV and its size.
        
        Args:
            rng: Random number generator
            housing_type: "VILLA" or "APARTMENT"
            income_decile: Household income decile (1-10)
            
        Returns:
            System size in kW, or None if no solar
        """
        base_prob = (self.villa_probability if housing_type == "VILLA" 
                     else self.apartment_probability)
        
        prob = base_prob * self.income_multipliers.get(income_decile, 1.0)
        prob = min(prob, 0.95)
        
        if rng.random() >= prob:
            return None
        
        # Sample system size
        size = rng.gauss(self.size_kw_mean, self.size_kw_std)
        size = max(self.size_min_kw, min(self.size_max_kw, size))
        return round(size, 1)


@dataclass
class BatteryConfig:
    """
    Home battery storage probability and sizing configuration.
    
    Battery ownership is conditional on having solar PV.
    
    Attributes:
        probability_given_solar: P(battery | has solar PV)
        probability_no_solar: P(battery | no solar) - rare
        income_multipliers: Probability multiplier by income decile
        size_kwh_mean: Mean battery capacity in kWh
        size_kwh_std: Standard deviation of capacity
        size_min_kwh: Minimum battery size
        size_max_kwh: Maximum battery size
    """
    
    probability_given_solar: float = 0.25
    probability_no_solar: float = 0.02
    
    income_multipliers: Dict[int, float] = field(default_factory=lambda: {
        1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6, 5: 0.8,
        6: 1.0, 7: 1.2, 8: 1.4, 9: 1.6, 10: 2.0,
    })
    
    # Battery sizing (kWh)
    size_kwh_mean: float = 10.0
    size_kwh_std: float = 4.0
    size_min_kwh: float = 5.0
    size_max_kwh: float = 30.0
    
    def sample(self, rng: random.Random, has_solar: bool, 
               income_decile: int) -> Optional[float]:
        """
        Sample whether household has battery storage and its size.
        
        Args:
            rng: Random number generator
            has_solar: Whether household has solar PV
            income_decile: Household income decile (1-10)
            
        Returns:
            Battery capacity in kWh, or None if no battery
        """
        base_prob = (self.probability_given_solar if has_solar 
                     else self.probability_no_solar)
        
        prob = base_prob * self.income_multipliers.get(income_decile, 1.0)
        prob = min(prob, 0.95)
        
        if rng.random() >= prob:
            return None
        
        # Sample battery size
        size = rng.gauss(self.size_kwh_mean, self.size_kwh_std)
        size = max(self.size_min_kwh, min(self.size_max_kwh, size))
        return round(size, 1)


@dataclass
class BuildingEnvelopeConfig:
    """
    Building envelope and construction era configuration.
    
    GbgSynth does not have year-of-construction or window data.
    All envelope parameters are therefore **assumptions**, not measurements:
    
    - **Construction era** is sampled probabilistically from the Swedish
      housing stock age distribution (SCB / Energimyndigheten statistics).
    - **Window area** is estimated from a configurable window-to-floor-area
      ratio (no window data exists in the building footprints).
    - **Wall / roof / floor geometry** is derived from typological rules
      (e.g. "2-storey square-plan villa") applied to the known floor area.
      GbgSynth has LiDAR building heights in the footprint GeoDataFrame,
      but these are not yet propagated to the Dwelling model.
    - **Storey heights** are configurable assumptions (villa vs apartment
      may differ).
    
    These assumptions are aligned with the defaults in SweLoadSim's
    ``create_swedish_villa`` / ``create_swedish_apartment`` factories so
    that the 5R1C model receives consistent inputs.
    
    Attributes:
        era_probabilities_villa: P(era | villa) from Swedish stock stats
        era_probabilities_apartment: P(era | apartment) from Swedish stock stats
        window_to_floor_ratio_villa: Assumed total window area ÷ heated floor area for villas
        window_to_floor_ratio_apartment: Assumed total window area ÷ heated floor area for apartments
        storey_height_villa_m: Assumed floor-to-floor height for villas [m]
        storey_height_apartment_m: Assumed floor-to-floor (ceiling) height for apartments [m]
        u_values: U-value lookup by era and building type [W/(m²·K)]
    """
    
    # Swedish housing stock age distribution (SCB, Energimyndigheten)
    # Villa: ~25% pre-1960, ~30% miljonprogrammet, ~20% 76-90, ~15% 91-10, ~10% post-2010
    era_probabilities_villa: Dict[str, float] = field(default_factory=lambda: {
        "pre_1960": 0.25,
        "1960_1975": 0.30,
        "1976_1990": 0.20,
        "1991_2010": 0.15,
        "post_2010": 0.10,
    })
    
    # Apartments: ~15% pre-1960, ~35% miljonprogrammet, ~20% 76-90, ~20% 91-10, ~10% post-2010
    era_probabilities_apartment: Dict[str, float] = field(default_factory=lambda: {
        "pre_1960": 0.15,
        "1960_1975": 0.35,
        "1976_1990": 0.20,
        "1991_2010": 0.20,
        "post_2010": 0.10,
    })
    
    # ASSUMPTION: Window-to-floor-area ratio.
    # There is NO window data in GbgSynth — these are design rule-of-thumb
    # values consistent with SweLoadSim's 5R1C factory functions.
    # Swedish BBR guideline ~10-15% of floor area for daylighting.
    window_to_floor_ratio_villa: float = 0.15       # 15% — typical Swedish villa
    window_to_floor_ratio_apartment: float = 0.12   # 12% — mid-block apartment, fewer external walls
    
    # ASSUMPTION: Storey heights [m].
    # Villas: ~2.5 m floor-to-floor (matches SweLoadSim create_swedish_villa).
    # Apartments: ~2.6 m clear height (matches SweLoadSim create_swedish_apartment).
    storey_height_villa_m: float = 2.5
    storey_height_apartment_m: float = 2.6
    
    # U-value tables by era [W/(m²·K)] — aligned with SweLoadSim defaults
    u_values: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=lambda: {
        "pre_1960": {
            "villa": {"u_walls": 0.60, "u_roof": 0.40, "u_floor": 0.40, "u_windows": 2.80},
            "apartment": {"u_walls": 0.50, "u_roof": 0.35, "u_floor": 0.35, "u_windows": 2.50},
        },
        "1960_1975": {
            "villa": {"u_walls": 0.40, "u_roof": 0.25, "u_floor": 0.30, "u_windows": 2.50},
            "apartment": {"u_walls": 0.35, "u_roof": 0.20, "u_floor": 0.25, "u_windows": 2.20},
        },
        "1976_1990": {
            "villa": {"u_walls": 0.25, "u_roof": 0.15, "u_floor": 0.25, "u_windows": 1.80},
            "apartment": {"u_walls": 0.22, "u_roof": 0.12, "u_floor": 0.20, "u_windows": 1.60},
        },
        "1991_2010": {
            "villa": {"u_walls": 0.20, "u_roof": 0.12, "u_floor": 0.20, "u_windows": 1.40},
            "apartment": {"u_walls": 0.18, "u_roof": 0.10, "u_floor": 0.18, "u_windows": 1.20},
        },
        "post_2010": {
            "villa": {"u_walls": 0.15, "u_roof": 0.10, "u_floor": 0.15, "u_windows": 1.00},
            "apartment": {"u_walls": 0.12, "u_roof": 0.08, "u_floor": 0.12, "u_windows": 0.90},
        },
    })
    
    def sample_era(self, rng: random.Random, housing_type: str) -> str:
        """
        Sample a building construction era from stock distribution.
        
        Args:
            rng: Random number generator
            housing_type: "VILLA" or "APARTMENT"
            
        Returns:
            Era string, e.g. "1960_1975"
        """
        probs = (self.era_probabilities_villa if housing_type == "VILLA" 
                 else self.era_probabilities_apartment)
        r = rng.random()
        cumulative = 0.0
        for era, prob in probs.items():
            cumulative += prob
            if r < cumulative:
                return era
        return list(probs.keys())[-1]
    
    def get_u_values(self, era: str, housing_type: str) -> Dict[str, float]:
        """
        Get U-values for a given era and housing type.
        
        Args:
            era: Construction era string
            housing_type: "VILLA" or "APARTMENT"
            
        Returns:
            Dict with u_walls, u_roof, u_floor, u_windows
        """
        btype = "villa" if housing_type == "VILLA" else "apartment"
        return self.u_values.get(era, self.u_values["1976_1990"]).get(
            btype, self.u_values["1976_1990"]["apartment"]
        )
    
    def get_ventilation_params(self, era: str) -> Dict[str, float]:
        """
        Get ventilation parameters by era.
        
        Returns:
            Dict with ach_infiltration, ach_ventilation, heat_recovery_efficiency
        """
        if era in ("pre_1960", "1960_1975"):
            return {"ach_infiltration": 0.3, "ach_ventilation": 0.5, "heat_recovery_efficiency": 0.0}
        elif era == "1976_1990":
            return {"ach_infiltration": 0.15, "ach_ventilation": 0.5, "heat_recovery_efficiency": 0.0}
        else:  # 1991_2010, post_2010
            return {"ach_infiltration": 0.15, "ach_ventilation": 0.5, "heat_recovery_efficiency": 0.7}


@dataclass
class SweLoadSimConfig:
    """
    Complete configuration for SweLoadSim export.
    
    Combines all probability configurations with regional presets
    and custom configuration support.
    
    Attributes:
        heating: Heating system distribution config
        ev: Electric vehicle ownership config
        solar: Solar PV system config
        battery: Battery storage config
        envelope: Building envelope and construction era config
        summerhouse_by_income: P(summerhouse) by income decile
        seed: Random seed for reproducibility (None = random)
    
    Example:
        >>> # Use preset
        >>> config = SweLoadSimConfig.swedish_2024()
        >>> 
        >>> # Custom config
        >>> config = SweLoadSimConfig(
        ...     ev=EVConfig(base_probability=0.5),
        ...     solar=SolarConfig(villa_probability=0.3),
        ...     seed=42,
        ... )
    """
    
    heating: HeatingConfig = field(default_factory=HeatingConfig)
    ev: EVConfig = field(default_factory=EVConfig)
    solar: SolarConfig = field(default_factory=SolarConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    envelope: BuildingEnvelopeConfig = field(default_factory=BuildingEnvelopeConfig)
    
    summerhouse_by_income: Dict[int, float] = field(default_factory=lambda: {
        1: 0.02, 2: 0.03, 3: 0.05, 4: 0.07, 5: 0.10,
        6: 0.15, 7: 0.22, 8: 0.30, 9: 0.40, 10: 0.50,
    })
    
    seed: Optional[int] = None
    
    # -------------------------------------------------------------------------
    # Regional Presets
    # -------------------------------------------------------------------------
    
    @classmethod
    def swedish_2024(cls) -> "SweLoadSimConfig":
        """Swedish national averages for 2024."""
        return cls(
            heating=HeatingConfig(
                apartment_district=0.90,
                apartment_heat_pump=0.05,
                apartment_electric=0.05,
                villa_district=0.10,
                villa_heat_pump=0.55,
                villa_electric=0.20,
                villa_wood=0.10,
                villa_gas=0.05,
            ),
            ev=EVConfig(base_probability=0.30),
            solar=SolarConfig(villa_probability=0.15),
            battery=BatteryConfig(probability_given_solar=0.25),
        )
    
    @classmethod
    def future_2030(cls) -> "SweLoadSimConfig":
        """Projected 2030 scenario (moderate electrification)."""
        return cls(
            heating=HeatingConfig(
                apartment_district=0.88,
                apartment_heat_pump=0.08,
                apartment_electric=0.04,
                villa_district=0.08,
                villa_heat_pump=0.65,
                villa_electric=0.15,
                villa_wood=0.08,
                villa_gas=0.04,
            ),
            ev=EVConfig(base_probability=0.50),
            solar=SolarConfig(villa_probability=0.30),
            battery=BatteryConfig(probability_given_solar=0.40),
        )
    
    @classmethod
    def future_2035(cls) -> "SweLoadSimConfig":
        """Projected 2035 scenario (high electrification)."""
        return cls(
            heating=HeatingConfig(
                apartment_district=0.85,
                apartment_heat_pump=0.12,
                apartment_electric=0.03,
                villa_district=0.06,
                villa_heat_pump=0.75,
                villa_electric=0.10,
                villa_wood=0.06,
                villa_gas=0.03,
            ),
            ev=EVConfig(base_probability=0.70),
            solar=SolarConfig(
                villa_probability=0.45,
                size_kw_mean=10.0,  # Larger systems
            ),
            battery=BatteryConfig(
                probability_given_solar=0.55,
                size_kwh_mean=13.0,  # Larger batteries
            ),
        )
    
    @classmethod
    def future_2040(cls) -> "SweLoadSimConfig":
        """Projected 2040 scenario (near-full electrification)."""
        return cls(
            heating=HeatingConfig(
                apartment_district=0.80,
                apartment_heat_pump=0.17,
                apartment_electric=0.03,
                villa_district=0.04,
                villa_heat_pump=0.85,
                villa_electric=0.05,
                villa_wood=0.04,
                villa_gas=0.02,
            ),
            ev=EVConfig(base_probability=0.85),
            solar=SolarConfig(
                villa_probability=0.60,
                apartment_probability=0.10,
                size_kw_mean=12.0,
            ),
            battery=BatteryConfig(
                probability_given_solar=0.70,
                probability_no_solar=0.10,
                size_kwh_mean=15.0,
            ),
        )

    @classmethod
    def for_year(cls, year: int) -> "SweLoadSimConfig":
        """
        Auto-select the best config preset for a given year.

        Picks the closest matching preset so that prognosis-scaled
        populations get technology adoption rates consistent with
        their target year.

        Available presets: swedish_2024, future_2030, future_2035,
        future_2040.

        Args:
            year: Target year (e.g. 2024, 2030, 2032, 2040)

        Returns:
            The closest SweLoadSimConfig preset

        Example:
            >>> config = SweLoadSimConfig.for_year(2032)
            >>> # Returns future_2030() (closest to 2032)
        """
        presets = [
            (2024, cls.swedish_2024),
            (2030, cls.future_2030),
            (2035, cls.future_2035),
            (2040, cls.future_2040),
        ]

        best_year, best_fn = min(
            presets, key=lambda t: abs(t[0] - year)
        )
        return best_fn()


# =============================================================================
# Exporter Implementation
# =============================================================================

class SweLoadSimExporter(BaseExporter):
    """
    Export GbgSynth population for SweLoadSim energy simulation.
    
    Converts synthetic population data to SweLoadSim-compatible JSON format,
    with probabilistic assignment of:
    - Heating system type
    - Electric vehicle ownership
    - Solar PV systems
    - Battery storage
    - Summer house ownership
    
    Attributes:
        config: SweLoadSimConfig with probability distributions
        
    Example:
        >>> exporter = SweLoadSimExporter()
        >>> exporter.export(area, "population.json")
        >>> 
        >>> # With custom config
        >>> config = SweLoadSimConfig.future_2035()
        >>> exporter = SweLoadSimExporter(config=config)
        >>> exporter.export(area, "population_2035.json")
    """
    
    name = "sweloadsim"
    file_extension = ".json"
    description = "SweLoadSim household energy simulation format"
    
    # Age group mapping (exact age → SweLoadSim bins)
    AGE_BINS = [
        (0, 6, "0-6"),
        (7, 12, "7-12"),
        (13, 17, "13-17"),
        (18, 30, "18-30"),
        (31, 50, "30-50"),
        (51, 65, "50-65"),
        (66, 999, "65+"),
    ]
    
    # Housing type mapping (GbgSynth → SweLoadSim)
    HOUSING_MAP = {
        "apartment": "APARTMENT",
        "detached_house": "VILLA",
        "terraced_house": "VILLA",
        "semi_detached": "VILLA",
        "other": "APARTMENT",
    }
    
    # Employment status mapping (must match schema employment_status enum)
    STATUS_MAP = {
        "employed": "FULL_TIME",
        "part_time": "PART_TIME",
        "student": "STUDENT",
        "retired": "RETIRED",
        "child": "STUDENT",           # Children → STUDENT (schema: no CHILD)
        "unemployed": "FULL_TIME",     # No HOME in schema → FULL_TIME
        "parental_leave": "FULL_TIME", # No HOME in schema → FULL_TIME
        None: "FULL_TIME",
    }
    
    def __init__(self, config: Optional[SweLoadSimConfig] = None):
        """
        Initialize exporter with configuration.
        
        Args:
            config: SweLoadSimConfig instance. If None, uses swedish_2024().
        """
        self.config = config or SweLoadSimConfig.swedish_2024()
        self._rng = random.Random(self.config.seed)
    
    def export(self, area: "GbgArea", output_path: Path) -> Path:
        """
        Export area population to SweLoadSim JSON format.
        
        Args:
            area: GbgArea with generated population
            output_path: Path for output JSON file
            
        Returns:
            Path to created file
        """
        self.validate_area(area)
        output_path = Path(output_path)
        
        # Reset RNG for reproducibility
        self._rng = random.Random(self.config.seed)
        
        data = {
            "schema_version": "1.0.0",
            "metadata": self._build_metadata(area),
            "export_config": self._serialize_config(),
            "households": [
                self._convert_household(hh) 
                for hh in area.households
            ],
        }
        
        # Calculate summary statistics
        data["summary"] = self._calculate_summary(data["households"])
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _build_metadata(self, area: "GbgArea") -> Dict[str, Any]:
        """Build metadata section, including prognosis info if available."""
        meta = {
            "source": "GbgSynth",
            "area_code": area.area_code,
            "area_name": area.area_name,
            "year": area.year,
            "total_households": len(area.households),
            "total_individuals": len(area.individuals),
            "export_timestamp": datetime.now().isoformat(),
            "target_simulator": "SweLoadSim",
        }

        # Include prognosis scaling context when present
        prognosis = area.stats.get("prognosis") if area.stats else None
        if prognosis:
            meta["prognosis"] = {
                "base_year": prognosis.get("base_year"),
                "target_year": prognosis.get("target_year"),
                "mel_code": prognosis.get("mel_code"),
                "mel_name": prognosis.get("mel_name"),
                "base_population": prognosis.get("base_population"),
                "target_population": prognosis.get("target_population"),
                "overall_growth": prognosis.get("overall_growth"),
            }

        return meta
    
    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize configuration for reproducibility."""
        return {
            "seed": self.config.seed,
            "heating": {
                "apartment_district": self.config.heating.apartment_district,
                "apartment_heat_pump": self.config.heating.apartment_heat_pump,
                "apartment_electric": self.config.heating.apartment_electric,
                "villa_district": self.config.heating.villa_district,
                "villa_heat_pump": self.config.heating.villa_heat_pump,
                "villa_electric": self.config.heating.villa_electric,
                "villa_wood": self.config.heating.villa_wood,
                "villa_gas": self.config.heating.villa_gas,
            },
            "ev": {
                "base_probability": self.config.ev.base_probability,
            },
            "solar": {
                "villa_probability": self.config.solar.villa_probability,
                "apartment_probability": self.config.solar.apartment_probability,
                "size_kw_mean": self.config.solar.size_kw_mean,
            },
            "battery": {
                "probability_given_solar": self.config.battery.probability_given_solar,
                "probability_no_solar": self.config.battery.probability_no_solar,
                "size_kwh_mean": self.config.battery.size_kwh_mean,
            },
            "envelope": {
                "era_probabilities_villa": self.config.envelope.era_probabilities_villa,
                "era_probabilities_apartment": self.config.envelope.era_probabilities_apartment,
                "window_to_floor_ratio_villa": self.config.envelope.window_to_floor_ratio_villa,
                "window_to_floor_ratio_apartment": self.config.envelope.window_to_floor_ratio_apartment,
                "storey_height_villa_m": self.config.envelope.storey_height_villa_m,
                "storey_height_apartment_m": self.config.envelope.storey_height_apartment_m,
            },
        }
    
    def _convert_household(self, hh: "Household") -> Dict[str, Any]:
        """Convert a GbgSynth Household to SweLoadSim format.
        
        Includes building envelope parameters for the 5R1C heating model:
        - Construction era (sampled from stock distribution)
        - U-values for walls, roof, floor, windows
        - Building geometry (wall area, roof area, window area)
        - Ventilation parameters
        """
        # Determine housing type
        house_type = hh.house_type or "apartment"
        housing_type = self.HOUSING_MAP.get(house_type, "APARTMENT")
        
        # Get floor area from dwelling or estimate
        floor_area = hh.floor_area
        if floor_area is None and hh.dwelling:
            floor_area = hh.dwelling.floor_area
        if floor_area is None:
            # Estimate based on household size
            floor_area = 50 + (hh.size - 1) * 20
        
        # Get income decile (default to 5 if not available)
        income_decile = self._get_income_decile(hh)
        
        # Sample heating system
        if housing_type == "APARTMENT":
            heating_type = self.config.heating.sample_apartment(self._rng)
        else:
            heating_type = self.config.heating.sample_villa(self._rng)
        
        # Sample EV ownership
        has_ev = self.config.ev.sample(
            self._rng, hh.cars, income_decile, housing_type
        )
        
        # Sample solar PV
        solar_kw = self.config.solar.sample(
            self._rng, housing_type, income_decile
        )
        
        # Sample battery (conditional on solar)
        battery_kwh = self.config.battery.sample(
            self._rng, solar_kw is not None, income_decile
        )
        
        # Sample summer house
        summerhouse_prob = self.config.summerhouse_by_income.get(income_decile, 0.1)
        has_summerhouse = self._rng.random() < summerhouse_prob
        
        # Determine luxury level
        luxury_level = "HIGH" if income_decile >= 7 else "STANDARD"
        
        # =================================================================
        # BUILDING ENVELOPE — for SweLoadSim 5R1C heating model
        # =================================================================
        envelope_cfg = self.config.envelope
        
        # Sample construction era from stock distribution
        building_era = envelope_cfg.sample_era(self._rng, housing_type)
        
        # Get U-values for this era and housing type
        u_values = envelope_cfg.get_u_values(building_era, housing_type)
        
        # Get ventilation parameters for this era
        ventilation = envelope_cfg.get_ventilation_params(building_era)
        
        # Estimate building geometry from typological assumptions
        building_geometry = self._derive_geometry(
            hh, housing_type, floor_area, envelope_cfg
        )
        
        # Which window-to-floor ratio was assumed (for transparency)
        assumed_wfr = (
            envelope_cfg.window_to_floor_ratio_villa if housing_type == "VILLA"
            else envelope_cfg.window_to_floor_ratio_apartment
        )
        
        return {
            "household_id": hh.household_id,
            "area_m2": round(floor_area, 1),
            "housing_type": housing_type,
            "heating_type": heating_type,
            "luxury_level": luxury_level,
            "has_ev": has_ev,
            "has_summerhouse": has_summerhouse,
            "solar_pv_kw": solar_kw,
            "battery_kwh": battery_kwh,
            "num_cars": hh.cars,
            "income_decile": income_decile,
            # Building envelope (all assumptions — no measured window or era data)
            "building_era": building_era,
            "envelope": {
                **u_values,
                **building_geometry,
                "window_to_floor_ratio": assumed_wfr,
                **ventilation,
            },
            "members": [
                self._convert_agent(agent) 
                for agent in hh.members
            ],
        }
    
    def _convert_agent(self, agent: "Agent") -> Dict[str, Any]:
        """Convert a GbgSynth Agent to SweLoadSim member format."""
        return {
            "age_group": self._age_to_bin(agent.age),
            "age_exact": agent.age,
            "sex": agent.sex,
            "status": self.STATUS_MAP.get(agent.status, "FULL_TIME"),
        }
    
    def _age_to_bin(self, age: int) -> str:
        """Map exact age to SweLoadSim age group."""
        for low, high, label in self.AGE_BINS:
            if low <= age <= high:
                return label
        return "65+"
    
    def _get_income_decile(self, hh: "Household") -> int:
        """Get household income decile, with fallback."""
        # Try to get from household head or first adult
        for member in hh.members:
            if hasattr(member, 'income_decile') and member.income_decile:
                return member.income_decile
        # Default to median
        return 5
    
    def _derive_geometry(
        self, hh: "Household", housing_type: str, 
        floor_area: float, envelope_cfg: "BuildingEnvelopeConfig",
    ) -> Dict[str, float]:
        """
        Estimate building geometry from typological assumptions.
        
        ALL values here are assumptions, not measurements:
        
        - **Villa**: Assumed 2-storey, square plan, pitched roof (×1.2).
          Wall area = 4 × side × height.  Matches SweLoadSim's
          ``create_swedish_villa`` factory.
        - **Apartment**: Assumed mid-block unit with 2 external walls.
          Roof/floor transmission reduced to 20% of floor area (thermal
          bridges only).  Matches ``create_swedish_apartment``.
        - **Window area**: floor_area × assumed window-to-floor ratio
          (no window data exists in GbgSynth).
        - **Storey heights**: from ``BuildingEnvelopeConfig``
          (villa 2.5 m, apartment 2.6 m by default).
        
        Future improvement: propagate the LiDAR building height and
        footprint area from the GbgSynth GeoDataFrame onto the Dwelling
        model, then use real values here instead of the square-plan
        assumption.
        
        Args:
            hh: GbgSynth Household (dwelling link currently unused)
            housing_type: "VILLA" or "APARTMENT"
            floor_area: Heated floor area [m²]
            envelope_cfg: Envelope configuration with storey heights and
                          window-to-floor ratios
            
        Returns:
            Dict with wall_area_m2, roof_area_m2, floor_area_ground_m2,
            window_area_m2, building_height_m, num_stories
        """
        if housing_type == "VILLA":
            h_storey = envelope_cfg.storey_height_villa_m
            stories = 2
            footprint = floor_area / stories
            side_length = math.sqrt(footprint)
            height = h_storey * stories
            
            return {
                "wall_area_m2": round(4 * side_length * height, 1),
                "roof_area_m2": round(footprint * 1.2, 1),  # Pitched roof factor
                "floor_area_ground_m2": round(footprint, 1),
                "window_area_m2": round(
                    floor_area * envelope_cfg.window_to_floor_ratio_villa, 1
                ),
                "building_height_m": round(height, 1),
                "num_stories": stories,
            }
        else:
            # Apartment: mid-block unit with 2 external walls
            h_storey = envelope_cfg.storey_height_apartment_m
            width = math.sqrt(floor_area)
            external_wall_area = 2 * width * h_storey
            effective_roof_floor = floor_area * 0.2  # Thermal bridge / edge losses
            
            return {
                "wall_area_m2": round(external_wall_area, 1),
                "roof_area_m2": round(effective_roof_floor, 1),
                "floor_area_ground_m2": round(effective_roof_floor, 1),
                "window_area_m2": round(
                    floor_area * envelope_cfg.window_to_floor_ratio_apartment, 1
                ),
                "building_height_m": round(h_storey, 1),
                "num_stories": 1,  # Single-unit perspective
            }
    
    def _calculate_summary(self, households: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for export."""
        n = len(households)
        if n == 0:
            return {}
        
        # Heating distribution
        heating_counts = {}
        for hh in households:
            ht = hh["heating_type"]
            heating_counts[ht] = heating_counts.get(ht, 0) + 1
        
        # Housing type distribution
        housing_counts = {}
        for hh in households:
            ht = hh["housing_type"]
            housing_counts[ht] = housing_counts.get(ht, 0) + 1
        
        # Technology adoption rates
        ev_count = sum(1 for hh in households if hh["has_ev"])
        solar_count = sum(1 for hh in households if hh["solar_pv_kw"])
        battery_count = sum(1 for hh in households if hh["battery_kwh"])
        summerhouse_count = sum(1 for hh in households if hh["has_summerhouse"])
        
        # Building era distribution
        era_counts: Dict[str, int] = {}
        for hh in households:
            era = hh.get("building_era", "unknown")
            era_counts[era] = era_counts.get(era, 0) + 1
        
        # Solar and battery sizing
        solar_sizes = [hh["solar_pv_kw"] for hh in households if hh["solar_pv_kw"]]
        battery_sizes = [hh["battery_kwh"] for hh in households if hh["battery_kwh"]]
        
        return {
            "total_households": n,
            "heating_distribution": {k: round(v/n*100, 1) for k, v in heating_counts.items()},
            "housing_distribution": {k: round(v/n*100, 1) for k, v in housing_counts.items()},
            "era_distribution": {k: round(v/n*100, 1) for k, v in era_counts.items()},
            "ev_adoption_rate": round(ev_count / n * 100, 1),
            "solar_adoption_rate": round(solar_count / n * 100, 1),
            "battery_adoption_rate": round(battery_count / n * 100, 1),
            "summerhouse_rate": round(summerhouse_count / n * 100, 1),
            "solar_mean_kw": round(sum(solar_sizes) / len(solar_sizes), 1) if solar_sizes else None,
            "battery_mean_kwh": round(sum(battery_sizes) / len(battery_sizes), 1) if battery_sizes else None,
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """Return format documentation."""
        return {
            "description": "SweLoadSim household energy simulation format",
            "url": "https://github.com/your-org/SweLoadSim",
            "version": "1.0",
            "config_presets": [
                "swedish_2024",
                "future_2030",
                "future_2035",
                "future_2040",
            ],
            "fields": {
                "household_id": "Unique household identifier",
                "area_m2": "Living area in square meters",
                "housing_type": "VILLA or APARTMENT",
                "heating_type": "DISTRICT, HEAT_PUMP, DIRECT_ELECTRIC, WOOD_PELLET, or GAS",
                "luxury_level": "STANDARD or HIGH (based on income)",
                "has_ev": "Whether household has an electric vehicle",
                "has_summerhouse": "Whether household has a summer house",
                "solar_pv_kw": "Solar PV system size in kW (null if none)",
                "battery_kwh": "Battery storage capacity in kWh (null if none)",
                "building_era": "Construction era: pre_1960, 1960_1975, 1976_1990, 1991_2010, post_2010",
                "envelope": {
                    "u_walls": "Wall U-value [W/(m²·K)]",
                    "u_roof": "Roof U-value [W/(m²·K)]",
                    "u_floor": "Floor U-value [W/(m²·K)]",
                    "u_windows": "Window U-value [W/(m²·K)]",
                    "wall_area_m2": "External wall area [m²]",
                    "roof_area_m2": "Roof area [m²]",
                    "floor_area_ground_m2": "Ground-contact floor area [m²]",
                    "window_area_m2": "Total window area [m²]",
                    "window_to_floor_ratio": "Assumed window area ÷ floor area (no window data exists)",
                    "building_height_m": "Building height [m]",
                    "ach_infiltration": "Infiltration air changes [1/h]",
                    "ach_ventilation": "Mechanical ventilation [1/h]",
                    "heat_recovery_efficiency": "Heat recovery efficiency (0-1)",
                },
                "members": "List of household members with age_group and status",
            },
        }
