"""
13 · Export to SweLoadSim
==========================
Export a synthetic population in SweLoadSim-compatible JSON format
with configurable technology adoption rates (EVs, solar, heat pumps).
"""

import logging
logging.basicConfig(level=logging.ERROR)

from pathlib import Path
from gbgsynth import GbgSynth, SweLoadSimConfig
from gbgsynth.exporters import HeatingConfig, EVConfig, SolarConfig, BatteryConfig

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sweloadsim_exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

city = GbgSynth(year=2023)
haga = city.synthesize("Haga")

# 1. Default export (Swedish 2024 averages)
haga.export("sweloadsim", OUTPUT_DIR / "haga_default.json")
print("✓ haga_default.json  (Swedish 2024 baseline)")

# 2. Built-in 2035 future scenario
cfg = SweLoadSimConfig.future_2035()
haga.export("sweloadsim", OUTPUT_DIR / "haga_2035.json", config=cfg)
print("✓ haga_2035.json     (2035 high-electrification)")

# 3. Fully custom config
custom = SweLoadSimConfig(
    ev=EVConfig(base_probability=0.50),
    solar=SolarConfig(villa_probability=0.40, size_kw_mean=12.0),
    battery=BatteryConfig(probability_given_solar=0.45),
    seed=42,
)
haga.export("sweloadsim", OUTPUT_DIR / "haga_custom.json", config=custom)
print("✓ haga_custom.json   (50% EV, 40% solar villas)")

# 4. List available presets
print("\nAvailable presets:")
for name in ["swedish_2024", "future_2030", "future_2035", "future_2040"]:
    print(f"  SweLoadSimConfig.{name}()")
