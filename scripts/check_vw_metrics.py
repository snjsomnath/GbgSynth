"""Validate the per-dimension Voas & Williamson metrics."""
from gbgsynth import GbgSynth
import numpy as np

g = GbgSynth()
areas = ['107', '108', '109', '103', '302']

for code in areas:
    a = g.get_area(code)
    a.generate()
    c = a.compare_to_marginals(print_report=False)
    ov = c.get('overall', {})
    dm = ov.get('dim_metrics', {})

    print(f"\n{'='*70}")
    print(f"{a.area_name} ({code})")
    print(f"{'='*70}")
    print(f"  MAPE: {ov.get('mape',0):.1f}%   Corr: {ov.get('correlation',0):.4f}")
    print(f"  SAE median: {ov.get('sae_median',0):.4f}  mean: {ov.get('sae_mean',0):.4f}  max: {ov.get('sae_max',0):.4f}")
    print(f"  X² p-value (worst): {ov.get('chi2_p_min',0):.4f}   Z² p-value (worst): {ov.get('z2_p_min',0):.4f}")
    print()
    print(f"  {'Dimension':<20} {'SAE':>8} {'X² p':>8} {'Z² p':>8}  {'MAPE':>6}")
    print(f"  {'-'*52}")
    for dim_key, m in dm.items():
        # compute per-dim MAPE from the comparison rows
        comp = c.get(dim_key, {}).get('comparison', [])
        errs = [abs(r['error_pct']) for r in comp if r['actual'] > 0 and not r.get('exclude_from_mape')]
        dim_mape = np.mean(errs) if errs else 0
        print(f"  {dim_key:<20} {m['sae']:>8.4f} {m['chi2_p']:>8.4f} {m['z2_p']:>8.4f}  {dim_mape:>5.1f}%")

print(f"\n{'='*70}")
print("GenSynthPop benchmark: SAE 0.000-0.005, p-values close to 1.0")
print("="*70)
