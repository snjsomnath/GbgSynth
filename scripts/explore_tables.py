#!/usr/bin/env python3
"""Explore all candidate income/education tables from the Gothenburg API."""
from gbgsynth.api_client import PxWebClient
import requests

client = PxWebClient()

tables = {
    '20_HuvudInk': 'Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/20_HuvudInk_PRI.px',
    '23_InkomsterUtbildning': 'Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/23_InkomsterUtbildning_PRI.px',
    '33_InkomsterBakgrund': 'Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/33_InkomsterBakgrund_PRI.px',
    '10_InkStandard (current)': 'Inkomst och utbildning/Inkomster/Socioekonomi/10_InkStandard_PRI.px',
}

# Also check if there's an education-only table
edu_tables = {
    '10_UtbInr': 'Inkomst och utbildning/Utbildning/10_UtbInr_PRI.px',
}

for name, path in {**tables, **edu_tables}.items():
    print(f"\n{'='*70}")
    print(f"TABLE: {name}")
    print(f"Path: {path}")
    print('='*70)
    
    try:
        meta = client.fetch_metadata(path)
        for var in meta.get('variables', []):
            vals = var.get('values', [])
            texts = var.get('valueTexts', vals)
            print(f"\n  {var['code']}: {var['text']} ({len(vals)} values)")
            for v, t in zip(vals[:20], texts[:20]):
                print(f"    {v} = {t}")
            if len(vals) > 20:
                print(f"    ... ({len(vals)} total)")
    except Exception as e:
        print(f"  ERROR: {e}")

# Now fetch actual data for Haga from each table
print("\n" + "="*70)
print("ACTUAL DATA FOR HAGA (107)")
print("="*70)

# 23_InkomsterUtbildning - income by education level
print("\n--- 23_InkomsterUtbildning (Haga, 2023) ---")
try:
    df = client.query_all_variables(
        'Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/23_InkomsterUtbildning_PRI.px',
        '107 Haga', 2023
    )
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.to_string())
except Exception as e:
    print(f"ERROR: {e}")

# 33_InkomsterBakgrund - income by background 
print("\n--- 33_InkomsterBakgrund (Haga, 2023) ---")
try:
    df = client.query_all_variables(
        'Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/33_InkomsterBakgrund_PRI.px',
        '107 Haga', 2023
    )
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.to_string())
except Exception as e:
    print(f"ERROR: {e}")

# 10_UtbInr - education only
print("\n--- 10_UtbInr (Haga, 2023) ---")
try:
    df = client.query_all_variables(
        'Inkomst och utbildning/Utbildning/10_UtbInr_PRI.px',
        '107 Haga', 2023
    )
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.to_string())
except Exception as e:
    print(f"ERROR: {e}")
