"""
04 · Save Population to CSV
============================
Generate a population and save individuals, households,
and dwelling CSVs. All output goes to <project_root>/output/.
"""

import logging
logging.basicConfig(level=logging.ERROR)

from pathlib import Path
from gbgsynth import GbgSynth

OUTPUT_DIR = str(Path(__file__).parent.parent / "output")

city = GbgSynth(year=2023)
haga = city.synthesize("Haga")

# save() writes individuals + households (+ dwellings if available)
saved = haga.save(output_dir=OUTPUT_DIR)

print("Saved files:")
for kind, path in saved.items():
    print(f"  {kind}: {path}")
