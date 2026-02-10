#!/usr/bin/env python3
"""
Example: Export GbgSynth population for SweLoadSim simulation.

This script demonstrates how to:
1. Generate a synthetic population
2. Export it in SweLoadSim-compatible format
3. Use different configuration presets
4. Customize technology adoption rates

Usage:
    python export_to_sweloadsim.py
"""

import logging
from pathlib import Path

from gbgsynth import GbgSynth, SweLoadSimConfig
from gbgsynth.exporters import (
    SweLoadSimExporter,
    HeatingConfig,
    EVConfig,
    SolarConfig,
    BatteryConfig,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output" / "sweloadsim_exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("GbgSynth → SweLoadSim Export Examples")
    print("=" * 70)
    
    # Initialize GbgSynth
    city = GbgSynth(year=2023)
    
    # Generate population for Haga
    print("\n1. Generating synthetic population for Haga...")
    haga = city.synthesize("Haga")
    print(f"   Generated: {len(haga.individuals)} individuals, {len(haga.households)} households")
    
    # =========================================================================
    # Example 1: Default export (Swedish national 2024 averages)
    # =========================================================================
    print("\n2. Export with default configuration (Swedish 2024)...")
    
    output_file = OUTPUT_DIR / "haga_default.json"
    haga.export("sweloadsim", output_file)
    print(f"   Saved to: {output_file}")
    
    # =========================================================================
    # Example 2: Year-matched preset via for_year()
    # =========================================================================
    print("\n3. Export with year-matched configuration (for_year)...")
    
    config_matched = SweLoadSimConfig.for_year(2030)
    output_file = OUTPUT_DIR / "haga_2030_matched.json"
    haga.export("sweloadsim", output_file, config=config_matched)
    print(f"   Saved to: {output_file}")
    
    # =========================================================================
    # Example 3: Future scenario - 2035
    # =========================================================================
    print("\n4. Export with 2035 future scenario...")
    
    config_2035 = SweLoadSimConfig.future_2035()
    output_file = OUTPUT_DIR / "haga_2035.json"
    haga.export("sweloadsim", output_file, config=config_2035)
    print(f"   Saved to: {output_file}")
    print("   - EV probability: 70%")
    print("   - Solar PV on villas: 45%")
    print("   - Heat pump in villas: 75%")
    
    # =========================================================================
    # Example 4: Custom configuration
    # =========================================================================
    print("\n5. Export with fully custom configuration...")
    
    custom_config = SweLoadSimConfig(
        heating=HeatingConfig(
            apartment_district=0.80,
            apartment_heat_pump=0.15,
            apartment_electric=0.05,
            villa_district=0.05,
            villa_heat_pump=0.80,  # High heat pump scenario
            villa_electric=0.05,
            villa_wood=0.08,
            villa_gas=0.02,
        ),
        ev=EVConfig(
            base_probability=0.45,  # Higher EV adoption
        ),
        solar=SolarConfig(
            villa_probability=0.35,
            apartment_probability=0.05,
            size_kw_mean=10.0,  # Larger systems
        ),
        battery=BatteryConfig(
            probability_given_solar=0.40,
            probability_no_solar=0.05,
            size_kwh_mean=12.0,
        ),
        seed=42,  # Reproducible results
    )
    
    output_file = OUTPUT_DIR / "haga_custom.json"
    haga.export("sweloadsim", output_file, config=custom_config)
    print(f"   Saved to: {output_file}")
    
    # =========================================================================
    # Example 5: Multiple areas with same config
    # =========================================================================
    print("\n6. Export multiple areas...")
    
    config = SweLoadSimConfig.swedish_2024()
    config.seed = 123  # Consistent across areas
    
    for area_name in ["Majorna", "Haga", "Centrum"]:
        try:
            area = city.synthesize(area_name)
            output_file = OUTPUT_DIR / f"{area_name.lower()}_2024.json"
            area.export("sweloadsim", output_file, config=config)
            print(f"   {area_name}: {len(area.households)} households → {output_file.name}")
        except Exception as e:
            print(f"   {area_name}: Skipped ({e})")
    
    # =========================================================================
    # Show available presets
    # =========================================================================
    print("\n" + "=" * 70)
    print("Available configuration presets:")
    print("=" * 70)
    presets = [
        ("swedish_2024", "Swedish national averages"),
        ("future_2030", "2030 scenario (moderate electrification)"),
        ("future_2035", "2035 scenario (high electrification)"),
        ("future_2040", "2040 scenario (near-full electrification)"),
        ("for_year(year)", "Auto-select closest preset for a given year"),
    ]
    for name, desc in presets:
        print(f"  - SweLoadSimConfig.{name}(): {desc}")
    
    print("\n" + "=" * 70)
    print(f"All exports saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
