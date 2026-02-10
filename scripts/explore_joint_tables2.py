#!/usr/bin/env python3
"""Explore candidate joint distribution tables."""
from gbgsynth.api_client import PxWebClient
import time

client = PxWebClient()

tables = {
    'HH Type x Children': 'Befolkning/Hushåll/10_HHTypBarnU18_PRI.px',
    'HH Type (detailed)': 'Befolkning/Hushåll/20_Hushallstyp_PRI.px',
    'HH Size x Housing Type': 'Befolkning/Hushåll/31_HHStorlHustyp_PRI.px',
    'Folk HH Ställning': 'Befolkning/Folkmängd/Folkmängd helår/60_FolkmHHStallning_PRI.px',
    'Folk HH Size x Housing': 'Befolkning/Folkmängd/Folkmängd helår/61_FolkmHHstorlHustyp_PRI.px',
    'Folk HH Type': 'Befolkning/Folkmängd/Folkmängd helår/63_FolkmHHtypPRI.px',
}

for name, path in tables.items():
    print(f"\n{'='*70}")
    print(f"TABLE: {name}")
    print(f"Path: {path}")
    print("=" * 70)
    try:
        meta = client.fetch_metadata(path)
        for var in meta.get('variables', []):
            vals = var.get('values', [])
            texts = var.get('valueTexts', vals)
            print(f"  {var['code']}: {var['text']} ({len(vals)} values)")
            for v, t in zip(vals[:20], texts[:20]):
                print(f"    {v} = {t}")
            if len(vals) > 20:
                print(f"    ... ({len(vals)} total)")
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(0.5)

# Fetch sample data for Haga from the most interesting tables
print("\n\n" + "=" * 70)
print("SAMPLE DATA: HH Type x Children (Haga, 2024)")
print("=" * 70)
try:
    df = client.query_all_variables(
        'Befolkning/Hushåll/10_HHTypBarnU18_PRI.px', '107 Haga', 2024
    )
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.to_string())
except Exception as e:
    print(f"ERROR: {e}")
    try:
        df = client.query_all_variables(
            'Befolkning/Hushåll/10_HHTypBarnU18_PRI.px', '107 Haga', 2023
        )
        print(f"Shape (2023): {df.shape}")
        print(df.to_string())
    except Exception as e2:
        print(f"ERROR (2023): {e2}")

print("\n\n" + "=" * 70)
print("SAMPLE DATA: Folk HH Ställning (Haga, 2024)")
print("=" * 70)
try:
    df = client.query_all_variables(
        'Befolkning/Folkmängd/Folkmängd helår/60_FolkmHHStallning_PRI.px', '107 Haga', 2024
    )
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.to_string())
except Exception as e:
    print(f"ERROR: {e}")

print("\n\n" + "=" * 70)
print("SAMPLE DATA: Folk HH Type (Haga, 2024)")
print("=" * 70)
try:
    df = client.query_all_variables(
        'Befolkning/Folkmängd/Folkmängd helår/63_FolkmHHtypPRI.px', '107 Haga', 2024
    )
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.to_string())
except Exception as e:
    print(f"ERROR: {e}")
