"""
07 · Sanity Checks
==================
Verify that the synthetic population contains no unrealistic
households (e.g. children living alone, impossible age gaps).
"""

import logging
logging.basicConfig(level=logging.ERROR)

from gbgsynth import GbgSynth, run_all_checks

city = GbgSynth(year=2023)
haga = city.synthesize("Haga")

result = run_all_checks(haga.households, haga.individuals)

print(result.summary())
