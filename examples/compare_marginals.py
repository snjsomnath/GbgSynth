#!/usr/bin/env python
"""Test marginal comparison functionality."""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    synth = GbgSynth(year=2023)
    
    # One-liner synthesis
    haga = synth.synthesize("Haga")
    
    # Log all statistics including marginal comparison
    haga.log_statistics()


if __name__ == '__main__':
    main()
