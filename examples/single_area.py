#!/usr/bin/env python
"""
Generate a synthetic population for a single area.

This example demonstrates the simplest workflow using the one-liner synthesize().
"""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    # Initialize and synthesize in minimal steps
    city = GbgSynth(year=2023)
    
    # One-liner: get area + generate in one call
    # Can use area name "Haga" or code "107"
    haga = city.synthesize("Haga")
    
    # Save both individuals and households with one call
    haga.save(output_dir="./output")
    
    # Log all statistics
    haga.log_statistics()


if __name__ == "__main__":
    main()
