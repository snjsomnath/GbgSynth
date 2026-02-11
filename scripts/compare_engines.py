"""Compare per-dimension accuracy across all 3 synthesis engines."""
import sys
import numpy as np
from gbgsynth import GbgSynth

def p(s):
    print(s, flush=True)

city = GbgSynth(year=2024)

test_areas = ['107', '301', '103', '110', '504', '402']

dims = ['sex', 'age', 'role', 'household_size', 'housing_type', 'education', 'income_source']

for engine in ['topdown', 'ipf', 'constrained_ipf']:
    p(f"\n{'='*70}")
    p(f"ENGINE: {engine}")
    p(f"{'='*70}")
    dim_mapes = {d: [] for d in dims}
    overall_mapes = []

    for area_name in test_areas:
        try:
            r = city.synthesize(area_name, engine=engine)
            comp = r.compare_to_marginals(print_report=False)

            area_errs = []
            for d in dims:
                if d in comp and 'comparison' in comp[d]:
                    rows = comp[d]['comparison']
                    pct_errors = [abs(row.get('error_pct', 0))
                                  for row in rows if row.get('actual', 0) > 0]
                    if pct_errors:
                        mape = np.mean(pct_errors)
                        dim_mapes[d].append(mape)
                        area_errs.extend(pct_errors)

            if area_errs:
                overall_mapes.append(np.mean(area_errs))
                p(f"  {area_name}: MAPE={np.mean(area_errs):.1f}%")
        except Exception as e:
            import traceback; traceback.print_exc()
            p(f"  {area_name}: FAILED - {e}")

    # Print per-dimension averages
    p(f"{'Dimension':<20} {'Avg MAPE':>10}")
    p("-" * 32)
    for d in dims:
        if dim_mapes[d]:
            p(f"{d:<20} {np.mean(dim_mapes[d]):>9.1f}%")
        else:
            p(f"{d:<20}    (no data)")
    p("-" * 32)
    if overall_mapes:
        p(f"{'OVERALL':<20} {np.mean(overall_mapes):>9.1f}%")
