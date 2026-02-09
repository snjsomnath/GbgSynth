#!/usr/bin/env python3
"""
Validate synthetic population for unrealistic households.

This script runs comprehensive sanity checks to ensure NO unrealistic
households exist in the synthetic population. Even a single violation
should be investigated, as outliers reduce trust in model results.

Use Cases:
- Activity demand modeling
- Accessibility analysis  
- Household electricity demand estimation

Sanity checks are now integrated into the validation workflow.
Use the Validator class for comprehensive validation including sanity checks.

Checks Performed:
1.  No children-only households (children living without adults)
2.  No empty households
3.  Household size matches member count
4.  No overcrowded dwellings (too many people for room count)
5.  Valid age ranges (0-110)
6.  Parent-child age gaps are realistic
7.  Role-age consistency (e.g., 70-year-old shouldn't be "child")
8.  Single-person households have exactly 1 person
9.  Couple households have at least 2 adults
10. No orphan individuals (everyone belongs to a household)
11. Dwelling assignment coverage
12. Sex values are valid
13. Household type matches actual composition
"""

import sys
sys.path.insert(0, '..')

from gbgsynth import GbgSynth
from gbgsynth.validation import Validator


def validate_single_area(area_code: str):
    """Validate a single area's population using the integrated validator."""
    print(f"\n{'='*60}")
    print(f"VALIDATING AREA: {area_code}")
    print('='*60)
    
    city = GbgSynth(year=2023)
    area = city.get_area(area_code)
    area.generate()
    
    # Use the integrated Validator which includes sanity checks
    validator = Validator(area)
    report = validator.run_all_validations()
    
    print()
    print(report.summary())
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate synthetic population for unrealistic households"
    )
    parser.add_argument(
        "area", 
        nargs="?", 
        default="107",
        help="Area code to validate (e.g., '107'). Default: 107 (Haga)"
    )
    
    args = parser.parse_args()
    validate_single_area(args.area)
