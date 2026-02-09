#!/usr/bin/env python
"""
Validate synthetic population with out-of-sample census data.

This example demonstrates validation against census tables that 
were NOT used in the synthesis fitting process.
"""

import logging
from gbgsynth import GbgSynth
from gbgsynth.validation import Validator

logging.basicConfig(level=logging.WARNING, format='%(message)s')


def main():
    print("=" * 60)
    print("SYNTHETIC POPULATION VALIDATION")
    print("=" * 60)
    
    # Generate a population
    print("\n🔧 Generating synthetic population for Haga...")
    city = GbgSynth(year=2023)
    area = city.synthesize("Haga")
    
    print(f"   Generated: {len(area.individuals)} individuals, {len(area.households)} households")
    
    # Create validator and run all tests
    validator = Validator(area, tolerance=0.15)  # 15% tolerance
    
    # Run validations (fetches census data and compares)
    report = validator.run_all_validations()
    
    # Show detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    
    for name, result in report.results.items():
        print(f"\n📈 {name.upper()}")
        print(f"   Census total: {result.census_total}")
        print(f"   Synth total:  {result.synth_total}")
        print(f"   RMSE: {result.rmse:.2f}")
        print(f"   Correlation: {result.correlation:.4f}")
        print(f"   Max error: {result.max_error_pct:.1f}%")
        
        # Show category breakdown
        if result.categories and len(result.categories) <= 10:
            print("   Categories:")
            for cat in result.categories:
                if 'synth' in cat and 'census' in cat:
                    print(f"     {cat.get('category', 'N/A')}: synth={cat['synth']}, census={cat['census']}")
    
    # Show derived metrics
    print("\n" + "=" * 60)
    print("DERIVED METRICS (computed from synthetic population)")
    print("=" * 60)
    metrics = validator.compute_derived_metrics()
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"   {name}: {value:.3f}")
        else:
            print(f"   {name}: {value}")


if __name__ == "__main__":
    main()
