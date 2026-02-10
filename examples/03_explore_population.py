"""
03 · Explore a Population
=========================
Generate a neighbourhood and explore individuals, households,
and dwellings using the built-in DataFrame properties.
"""

import logging
logging.basicConfig(level=logging.ERROR)

from gbgsynth import GbgSynth

city = GbgSynth(year=2023)
haga = city.synthesize("Haga")

# ── Individuals ──────────────────────────────────────────
ind = haga.individuals_df
print("INDIVIDUALS")
print(f"  Columns: {list(ind.columns)}")
print(f"\n  Sex distribution:\n{ind['sex'].value_counts().to_string()}\n")
print(f"  Household roles:\n{ind['hh_role'].value_counts().to_string()}\n")
print(f"  Age quartiles:\n{ind['age'].describe().to_string()}\n")

# ── Households ───────────────────────────────────────────
hh = haga.households_df
print("HOUSEHOLDS")
print(f"  Columns: {list(hh.columns)}")
print(f"\n  Size distribution:\n{hh['size'].value_counts().sort_index().to_string()}\n")

# ── Raw model objects ────────────────────────────────────
sample_hh = haga.households[0]
print(f"First household object: {sample_hh}")
if sample_hh.members:
    print(f"  Head: {sample_hh.head}")
    print(f"  Members: {[str(m) for m in sample_hh.members]}")
