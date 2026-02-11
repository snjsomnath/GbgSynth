"""
Feasibility test: Can we adopt GenSynthPop's approach?

Key question: If we assign household-position as an individual attribute
(conditioned on age × sex) FIRST, then form households from those
individuals, do we get better role accuracy than the current
containers-first approach?

Tests:
  1. What does the position data look like?
  2. Do individual-level role counts from position data match the census?
  3. Can we derive sensible household counts from individual roles?
  4. Quick comparison: current topdown role error vs deterministic assignment.
"""
import sys
import numpy as np
import pandas as pd
from gbgsynth import GbgSynth
from gbgsynth.config import Config

cfg = Config()

TEST_AREAS = ['107', '301', '103']  # Haga, Gamlestaden, Majorna

city = GbgSynth(year=2024)

print("=" * 70)
print("TEST 1: What does the position data look like?")
print("=" * 70)

# Fetch raw position data for Haga
area = city.get_area('107')
pos_data = area._fetch_household_position_data()

if pos_data is None:
    print("ERROR: No position data available!")
    sys.exit(1)

print(f"\nShape: {pos_data.shape}")
print(f"Columns: {list(pos_data.columns)}")
print(f"\nFirst 20 rows:")
print(pos_data.head(20).to_string())
print(f"\nUnique values per column:")
for col in pos_data.columns:
    vals = pos_data[col].unique()
    print(f"  {col}: {len(vals)} values")
    if len(vals) <= 15:
        print(f"    {list(vals)}")

# Find the role/position column
role_col = None
for col in pos_data.columns:
    if 'ställning' in col.lower() or 'position' in col.lower() or 'hushåll' in col.lower():
        if col != 'Antal':
            role_col = col
            break
if role_col is None:
    # Try to identify by unique values
    for col in pos_data.columns:
        vals = pos_data[col].unique()
        if any('ensam' in str(v).lower() or 'barn' in str(v).lower() for v in vals):
            role_col = col
            break

count_col = 'Antal' if 'Antal' in pos_data.columns else pos_data.columns[-1]
age_col = 'Ålder' if 'Ålder' in pos_data.columns else None
sex_col = 'Kön' if 'Kön' in pos_data.columns else None

print(f"\nIdentified columns:")
print(f"  Role: {role_col}")
print(f"  Age:  {age_col}")
print(f"  Sex:  {sex_col}")
print(f"  Count: {count_col}")

if role_col:
    print(f"\nRole distribution (Haga):")
    role_counts = pos_data.groupby(role_col)[count_col].sum()
    total = role_counts.sum()
    for role, cnt in role_counts.items():
        print(f"  {role:<50} {cnt:>6}  ({100*cnt/total:>5.1f}%)")
    print(f"  {'TOTAL':<50} {total:>6}")

print("\n" + "=" * 70)
print("TEST 2: Role counts — position data vs current synthesis")
print("=" * 70)

for area_code in TEST_AREAS:
    print(f"\n--- Area {area_code} ---")
    
    # Get position data
    a = city.get_area(area_code)
    pd_data = a._fetch_household_position_data()
    
    if pd_data is None or role_col is None:
        print("  Skipped (no position data)")
        continue
    
    # Census role counts from position data
    census_roles = pd_data.groupby(role_col)[count_col].sum()
    census_total = census_roles.sum()
    
    # Synthesize with current topdown
    r = city.synthesize(area_code)
    individuals = r.individuals
    
    # Map census role names to synth role names
    synth_role_counts = {}
    for ind in individuals:
        role = ind.hh_role
        synth_role_counts[role] = synth_role_counts.get(role, 0) + 1
    
    print(f"  Census roles (from position data):")
    for role, cnt in sorted(census_roles.items(), key=lambda x: -x[1]):
        print(f"    {role:<50} {cnt:>6}")
    
    print(f"  Synthesised roles:")
    for role, cnt in sorted(synth_role_counts.items(), key=lambda x: -x[1]):
        print(f"    {role:<50} {cnt:>6}")
    
    print(f"  Census total: {census_total}, Synth total: {len(individuals)}")

print("\n" + "=" * 70)
print("TEST 3: Can we derive household counts from individual roles?")
print("=" * 70)

for area_code in TEST_AREAS:
    print(f"\n--- Area {area_code} ---")
    
    a = city.get_area(area_code)
    pd_data = a._fetch_household_position_data()
    hh_data = a._fetch_household_data()
    
    if pd_data is None or role_col is None:
        print("  Skipped")
        continue
    
    census_roles = pd_data.groupby(role_col)[count_col].sum()
    
    # Try to derive household counts from individual roles
    # singles -> 1 single-person HH each
    # cohabiting (married/sambo) -> pairs, so N/2 couple HHs
    # single parents -> 1 single-parent HH each
    # children -> assigned to couple or single-parent HHs
    
    # Count by mapped role (using Config canonical lookups)
    mapped = {}
    for role_name, cnt in census_roles.items():
        mapped_role = cfg.translate_position(role_name)
        mapped[mapped_role] = mapped.get(mapped_role, 0) + cnt
    
    print(f"  Mapped census roles:")
    for r, c in sorted(mapped.items(), key=lambda x: -x[1]):
        print(f"    {r:<20} {c:>6}")
    
    # Derive household counts
    n_single_hh = mapped.get('single', 0)
    n_couple_hh = mapped.get('cohabiting', 0) // 2
    n_sp_hh = mapped.get('single_parent', 0)
    n_children = mapped.get('child', 0)
    n_other = mapped.get('other', 0)
    
    derived_hh = n_single_hh + n_couple_hh + n_sp_hh
    
    # Actual household count
    hh_count_col = 'Antal' if 'Antal' in hh_data.columns else hh_data.columns[-1]
    actual_hh = int(hh_data[hh_count_col].sum())
    
    print(f"  Derived HH count: {derived_hh} (singles={n_single_hh}, "
          f"couples={n_couple_hh}, single_parent={n_sp_hh})")
    print(f"  + {n_children} children + {n_other} other to distribute")
    print(f"  Actual HH count from census: {actual_hh}")
    print(f"  Ratio: {derived_hh/actual_hh:.2f}" if actual_hh > 0 else "")

print("\n" + "=" * 70)
print("TEST 4: Role MAPE — current topdown vs perfect (position data)")
print("=" * 70)

for area_code in TEST_AREAS:
    print(f"\n--- Area {area_code} ---")
    
    r = city.synthesize(area_code)
    comp = r.compare_to_marginals(print_report=False)
    
    if 'role' in comp and 'comparison' in comp['role']:
        rows = comp['role']['comparison']
        errs = [abs(row.get('error_pct', 0)) for row in rows if row.get('actual', 0) > 0]
        role_mape = np.mean(errs) if errs else float('nan')
        print(f"  Current topdown role MAPE: {role_mape:.1f}%")
        for row in rows:
            if row.get('actual', 0) > 0:
                print(f"    {row.get('category','?'):<30} "
                      f"actual={row['actual']:>5} synth={row.get('synth',0):>5} "
                      f"err={row.get('error_pct',0):>+6.1f}%")
    else:
        print("  No role comparison data")

print("\nDone.")
