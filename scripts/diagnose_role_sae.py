"""Diagnose the root cause of high role SAE."""
import logging
logging.basicConfig(level=logging.WARNING)

from collections import Counter
import pandas as pd
from gbgsynth import GbgSynth
from gbgsynth.config import Config

cfg = Config()

g = GbgSynth(year=2024)

for code in ['107', '108', '302']:
    a = g.get_area(code)
    a.generate()
    c = a.compare_to_marginals(print_report=False)

    role_cmp = c.get('role', {}).get('comparison', [])
    print(f"\n{'='*70}")
    print(f"{a.area_name} ({code})")
    print(f"{'='*70}")
    print("Role comparison:")
    for row in role_cmp:
        print(f"  {row['category']:<20} actual={row['actual']:>5}  "
              f"synth={row['synth']:>5}  diff={row['diff']:>+5}  "
              f"err={row['error_pct']:>+6.1f}%")

    # Agent hh_role counts
    roles = Counter(ind.hh_role for ind in a.individuals)
    print(f"\nAgent hh_role counts: {dict(roles)}")

    # Check cohabiting pairing
    hh_by_id = {h.household_id: h for h in a.households}
    paired = solo = 0
    for ind in a.individuals:
        if ind.hh_role == 'cohabiting':
            hh = hh_by_id.get(ind.household_id)
            if hh:
                n_cohab = sum(1 for m in hh.members if m.hh_role == 'cohabiting')
                if n_cohab >= 2:
                    paired += 1
                else:
                    solo += 1
    print(f"Cohabiting: {paired} paired, {solo} solo (in HH without another cohabiting)")

    # Singles placed in multi-person HH
    singles_in_multi = 0
    singles_alone = 0
    for ind in a.individuals:
        if ind.hh_role == 'single':
            hh = hh_by_id.get(ind.household_id)
            if hh and len(hh.members) > 1:
                singles_in_multi += 1
            else:
                singles_alone += 1
    print(f"Singles: {singles_alone} alone, {singles_in_multi} in multi-person HH")

    # Census position data breakdown
    pos_data = a._marginals.get('household_position')
    pos_col = [c for c in pos_data.columns if 'ställning' in c.lower()][0]
    count_col = 'Antal'
    raw_counts = {}
    for _, row in pos_data.iterrows():
        pos = str(row[pos_col])
        val = int(row[count_col]) if pd.notna(row[count_col]) else 0
        raw_counts[pos] = raw_counts.get(pos, 0) + val
    print(f"\nCensus position data (7 raw categories):")
    for pos, cnt in sorted(raw_counts.items(), key=lambda x: -x[1]):
        print(f"  {pos:<45} {cnt:>5}")

    # Show how census maps to collapsed categories
    collapsed_counts = {}
    for pos, cnt in raw_counts.items():
        cat = cfg.translate_position_collapsed(pos)
        collapsed_counts[cat] = collapsed_counts.get(cat, 0) + cnt
    print(f"\nCollapsed census (5 categories):")
    for cat, cnt in sorted(collapsed_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<20} {cnt:>5}")

    # How translate_hh_position maps
    from gbgsynth.helpers.population_generator import translate_hh_position
    print(f"\ntranslate_hh_position mapping:")
    for pos in raw_counts.keys():
        mapped = translate_hh_position(pos)
        print(f"  {pos:<45} -> {mapped}")
