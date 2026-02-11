"""Check role error with collapsed Gift+Sambo → Cohabiting."""
from gbgsynth import GbgSynth
import numpy as np

g = GbgSynth()
test_areas = ['107', '108', '109', '103', '104', '105', '106',
              '201', '202', '301', '302', '401']

print(f"{'Area':<30}  {'role MAPE':>9}  {'overall':>7}  details")
print("-" * 100)

for code in test_areas:
    a = g.get_area(code)
    a.generate()
    c = a.compare_to_marginals(print_report=False)

    role = c.get('role', {})
    if role and 'comparison' in role:
        # Collapse Gift + Sambo into Cohabiting
        rows = role['comparison']
        collapsed = []
        gift_act = sum(r['actual'] for r in rows if r['category'] == 'Gift/reg.partner')
        gift_syn = sum(r['synth'] for r in rows if r['category'] == 'Gift/reg.partner')
        sambo_act = sum(r['actual'] for r in rows if r['category'] == 'Sambo')
        sambo_syn = sum(r['synth'] for r in rows if r['category'] == 'Sambo')
        
        cohab_act = gift_act + sambo_act
        cohab_syn = gift_syn + sambo_syn
        cohab_err = (cohab_syn - cohab_act) / cohab_act * 100 if cohab_act > 0 else 0

        # Other rows pass through
        detail_parts = []
        for r in rows:
            if r['category'] in ('Gift/reg.partner', 'Sambo'):
                continue
            detail_parts.append(f"{r['category']}={r['error_pct']:+.1f}%")
        detail_parts.append(f"Cohabiting={cohab_err:+.1f}%")

        # MAPE with collapsed
        errs_collapsed = []
        for r in rows:
            if r['category'] in ('Gift/reg.partner', 'Sambo'):
                continue
            if r['actual'] > 0:
                errs_collapsed.append(abs(r['error_pct']))
        if cohab_act > 0:
            errs_collapsed.append(abs(cohab_err))
        
        mape_collapsed = np.mean(errs_collapsed) if errs_collapsed else 0

        ov = c.get('overall', {})
        print(f"{a.area_name:<30}  {mape_collapsed:>8.1f}%  {ov.get('mape', 0):>6.1f}%  {', '.join(detail_parts)}")
