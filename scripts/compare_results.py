"""Compare before/after MAPE per area."""
import json

with open("output/results_before_stat_fixes.json") as f:
    old = {r["area_code"]: r for r in json.load(f) if r["status"] == "success" and "overall_fit" in r}
with open("output/results.json") as f:
    new = {r["area_code"]: r for r in json.load(f) if r["status"] == "success" and "overall_fit" in r}

print(f"{'Area':<30} {'Old MAPE':>9} {'New MAPE':>9} {'Delta':>7}  {'Old MaxE':>9} {'New MaxE':>9}")
print("-" * 85)

deltas = []
wmape_deltas = []
for code in sorted(old.keys()):
    if code not in new:
        name = old[code]["area_name"]
        om = old[code]["overall_fit"]["mape"]
        print(f"{name:<30} {om:>8.1f}%  {'FAILED':>8}")
        continue
    name = old[code]["area_name"]
    om = old[code]["overall_fit"]["mape"]
    nm = new[code]["overall_fit"]["mape"]
    ow = old[code]["overall_fit"]["wmape"]
    nw = new[code]["overall_fit"]["wmape"]
    omx = old[code]["overall_fit"]["max_pct_error"]
    nmx = new[code]["overall_fit"]["max_pct_error"]
    d = nm - om
    wd = nw - ow
    deltas.append(d)
    wmape_deltas.append(wd)
    flag = " WORSE" if d > 1.0 else " BETTER" if d < -1.0 else ""
    print(f"{name:<30} {om:>8.1f}% {nm:>8.1f}% {d:>+6.1f}%  {omx:>8.1f}% {nmx:>8.1f}%{flag}")

print("-" * 85)
improved = sum(1 for d in deltas if d < -0.1)
worsened = sum(1 for d in deltas if d > 0.1)
same = len(deltas) - improved - worsened
n = len(deltas)
print(f"Improved: {improved}, Worsened: {worsened}, Unchanged: {same}")
print(f"Avg MAPE delta:  {sum(deltas)/n:+.2f}%")
print(f"Avg WMAPE delta: {sum(wmape_deltas)/n:+.2f}%")
print()
old_avg_mape = sum(old[c]["overall_fit"]["mape"] for c in old) / len(old)
new_avg_mape = sum(new[c]["overall_fit"]["mape"] for c in new) / len(new)
old_avg_wmape = sum(old[c]["overall_fit"]["wmape"] for c in old) / len(old)
new_avg_wmape = sum(new[c]["overall_fit"]["wmape"] for c in new) / len(new)
print(f"Overall MAPE:  {old_avg_mape:.2f}% -> {new_avg_mape:.2f}%")
print(f"Overall WMAPE: {old_avg_wmape:.2f}% -> {new_avg_wmape:.2f}%")
