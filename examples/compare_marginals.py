#!/usr/bin/env python
"""Test marginal comparison functionality."""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    synth = GbgSynth(year=2024)
    
    # One-liner synthesis with IPF
    haga = synth.synthesize("Haga", use_ipf=True)
    
    # Log all statistics including marginal comparison
    haga.log_statistics()


if __name__ == '__main__':
    main()
