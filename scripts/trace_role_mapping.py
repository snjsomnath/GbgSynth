"""Trace how raw census positions map to collapsed categories."""
import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
from gbgsynth import GbgSynth
from gbgsynth.config import Config

cfg = Config()

g = GbgSynth()

for name in ['Haga', 'Annedal', 'Utby']:
    area = g.get_area(name)
    area.generate()

    pos_data = area._marginals.get('household_position')
    pos_col = None
    for col in pos_data.columns:
        if 'ställning' in col.lower() or 'position' in col.lower():
            pos_col = col
            break
    count_col = 'Antal' if 'Antal' in pos_data.columns else pos_data.columns[-1]

    # Raw position totals
    raw_pos = {}
    for _, row in pos_data.iterrows():
        p = str(row[pos_col])
        v = int(row[count_col]) if pd.notna(row[count_col]) else 0
        raw_pos[p] = raw_pos.get(p, 0) + v

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"\nRaw census positions:")
    for k in sorted(raw_pos.keys()):
        v = raw_pos[k]
        mapped = cfg.translate_position_collapsed(k)
        print(f"  {v:5d}  {k:55s} -> {mapped}")

    # Collapsed totals
    collapsed = {}
    for k, v in raw_pos.items():
        m = cfg.translate_position_collapsed(k)
        collapsed[m] = collapsed.get(m, 0) + v

    print(f"\nCollapsed census:")
    for k in sorted(collapsed.keys()):
        print(f"  {k:20s} = {collapsed[k]:5d}")

    # Now show the comparison output
    result = area._compare_role_distribution()
    print(f"\nComparison (actual vs synth):")
    for row in result['comparison']:
        print(f"  {row['category']:20s} actual={row['actual']:5d} synth={row['synth']:5d} diff={row['diff']:+5d} err={row['error_pct']:+.1f}%")

    total_actual = sum(r['actual'] for r in result['comparison'])
    sae = sum(abs(r['diff']) for r in result['comparison']) / (2 * total_actual) if total_actual > 0 else 0
    print(f"  SAE = {sae:.4f}")
