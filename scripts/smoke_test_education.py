"""Quick smoke test: synthesize Haga and check MAPE with new education comparison."""
import sys
sys.path.insert(0, '.')

from gbgsynth import GbgSynth

city = GbgSynth(year=2023)
area = city.get_area("Haga")
area.generate(allocate_dwellings=False)

comparisons = area.compare_to_marginals(print_report=True)

overall = comparisons.get('overall', {})
mape = overall.get('mape', None)
print(f"\n{'='*50}")
print(f"Overall MAPE: {mape:.1f}%")

# Show education specifically
edu = comparisons.get('education', {})
if edu and 'comparison' in edu:
    print(f"\nEducation Level Distribution:")
    for row in edu['comparison']:
        print(f"  {row['category']:>30}: actual={row['actual']:>5}, synth={row['synth']:>5}, error={row['error_pct']:>+6.1f}%")

# Check if individual agents have education
sample = area.individuals[:5]
for ind in sample:
    print(f"  Agent {ind.agent_id}: age={ind.age}, sex={ind.sex}, education={ind.education}")
