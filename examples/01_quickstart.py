"""
01 · Quick Start
================
Verify your installation and generate your first synthetic population
in five lines of code.
"""

import logging
logging.basicConfig(level=logging.ERROR)

from gbgsynth import GbgSynth

city = GbgSynth(year=2023)
haga = city.synthesize("Haga")

print(f"City:  {city}")
print(f"Area:  {haga}")
print(f"Pop:   {len(haga.individuals):,} individuals")
print(f"HH:    {len(haga.households):,} households")
print(f"\nAge stats:\n{haga.individuals_df['age'].describe()}")
