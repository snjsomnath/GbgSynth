#!/usr/bin/env python
"""
Compare synthesis results across multiple runs.

This example demonstrates the reproducibility of the synthesis
by running it twice and comparing the outputs.
"""

import logging
from gbgsynth import GbgSynth

logging.basicConfig(level=logging.WARNING)


def main():
    synth = GbgSynth(year=2023)

    # Run synthesis twice
    print('=== FIRST RUN ===')
    haga1 = synth.synthesize('Haga')
    stats1 = haga1.get_summary_statistics()
    comparison1 = haga1.compare_to_marginals(print_report=False)
    print(f"Households: {stats1['total_households']}")
    print(f"Individuals: {stats1['total_population']}")
    print(f"Correlation: {comparison1['overall']['correlation']:.4f}")

    print()
    print('=== SECOND RUN ===')
    haga2 = synth.synthesize('Haga')
    stats2 = haga2.get_summary_statistics()
    comparison2 = haga2.compare_to_marginals(print_report=False)
    print(f"Households: {stats2['total_households']}")
    print(f"Individuals: {stats2['total_population']}")
    print(f"Correlation: {comparison2['overall']['correlation']:.4f}")

    print()
    print('=== COMPARISON ===')
    print(f"{'Metric':<30} {'Run 1':>12} {'Run 2':>12} {'Diff':>10}")
    print('-' * 64)
    
    ov1 = comparison1['overall']
    ov2 = comparison2['overall']
    
    for m in ['rmse', 'mae', 'correlation']:
        v1 = ov1[m]
        v2 = ov2[m]
        diff = abs(v1 - v2)
        print(f"{m:<30} {v1:>12.4f} {v2:>12.4f} {diff:>10.4f}")

    print()
    print("Note: Small variations between runs are expected due to stochastic")
    print("elements in the synthesis (e.g., agent assignment order).")
    print("Both runs should have similar overall fit quality.")


if __name__ == '__main__':
    main()
