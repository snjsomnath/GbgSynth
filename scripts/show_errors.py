"""Show error percentage summary from results.json."""
import json

with open("output/results.json") as f:
    results = json.load(f)

header = f"{'Area':<35} {'MAPE':>6} {'WMAPE':>7} {'MaxErr%':>8} {'r':>7} {'RMSE':>8}"
print(header)
print("-" * 75)

mapes, wmapes, maxerrs, cors, rmses = [], [], [], [], []

for r in results:
    if r["status"] != "success" or "overall_fit" not in r:
        print(f"{r['area_name']:<35} FAILED")
        continue
    fit = r["overall_fit"]
    mape = fit.get("mape", 0)
    wmape = fit.get("wmape", 0)
    maxerr = fit.get("max_pct_error", 0)
    cor = fit.get("correlation", 0)
    rmse = fit.get("rmse", 0)
    mapes.append(mape)
    wmapes.append(wmape)
    maxerrs.append(maxerr)
    cors.append(cor)
    rmses.append(rmse)
    print(f"{r['area_name']:<35} {mape:>5.1f}% {wmape:>6.1f}% {maxerr:>7.1f}% {cor:>6.4f} {rmse:>8.1f}")

print("-" * 75)
n = len(mapes)
if n > 0:
    print(f"{'AVERAGE (' + str(n) + ' areas)':<35} {sum(mapes)/n:>5.1f}% {sum(wmapes)/n:>6.1f}% {sum(maxerrs)/n:>7.1f}% {sum(cors)/n:>6.4f} {sum(rmses)/n:>8.1f}")
    print(f"{'WORST':<35} {max(mapes):>5.1f}% {max(wmapes):>6.1f}% {max(maxerrs):>7.1f}% {min(cors):>6.4f} {max(rmses):>8.1f}")
    print(f"{'BEST':<35} {min(mapes):>5.1f}% {min(wmapes):>6.1f}% {min(maxerrs):>7.1f}% {max(cors):>6.4f} {min(rmses):>8.1f}")
