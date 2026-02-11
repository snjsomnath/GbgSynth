"""Quick check: does the evaluator fix drop role MAPE?"""
from gbgsynth import GbgSynth
import numpy as np

g = GbgSynth()
test_areas = ['107', '108', '109', '103', '104', '105', '106',
              '201', '202', '301', '302']

print(f"{'Area':<30}  {'role MAPE':>9}  {'overall':>7}  role details")
print("-" * 110)

for code in test_areas:
    a = g.get_area(code)
    a.generate()
    c = a.compare_to_marginals(print_report=False)

    role = c.get('role', {})
    if role and 'comparison' in role:
        errs = [abs(r['error_pct']) for r in role['comparison'] if r['actual'] > 0]
        mape = np.mean(errs) if errs else 0
        cats = ', '.join(
            f"{r['category']}={r['error_pct']:+.1f}%"
            for r in role['comparison']
        )
        ov = c.get('overall', {})
        print(f"{a.area_name:<30}  {mape:>8.1f}%  {ov.get('mape', 0):>6.1f}%  {cats}")
