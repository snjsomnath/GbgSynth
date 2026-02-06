#!/usr/bin/env python
"""Compare IPF vs Greedy synthesis methods."""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.WARNING)

synth = GbgSynth(year=2024)

# Compare methods using synthesize() one-liner
print('=== WITH IPF ===')
haga_ipf = synth.synthesize('Haga', use_ipf=True)
stats_ipf = haga_ipf.get_summary_statistics()
comparison_ipf = haga_ipf.compare_to_marginals(print_report=False)
print(f"Households: {stats_ipf['total_households']}")
print(f"Individuals: {stats_ipf['total_population']}")

print()
print('=== WITHOUT IPF (greedy only) ===')
haga_greedy = synth.synthesize('Haga', use_ipf=False)
stats_greedy = haga_greedy.get_summary_statistics()
comparison_greedy = haga_greedy.compare_to_marginals(print_report=False)
print(f"Households: {stats_greedy['total_households']}")
print(f"Individuals: {stats_greedy['total_population']}")

print()
print('=== COMPARISON ===')
print(f"{'Metric':<30} {'IPF':>12} {'Greedy':>12} {'Better':>10}")
print('-' * 64)
ipf_ov = comparison_ipf['overall']
greedy_ov = comparison_greedy['overall']
metrics = ['rmse', 'mae', 'max_error', 'correlation']
for m in metrics:
    iv = ipf_ov[m]
    gv = greedy_ov[m]
    if m == 'correlation':
        better = 'IPF' if iv > gv else 'Greedy'
    else:
        better = 'IPF' if iv < gv else 'Greedy'
    print(f'{m:<30} {iv:>12.2f} {gv:>12.2f} {better:>10}')
