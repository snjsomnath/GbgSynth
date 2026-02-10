#!/usr/bin/env python3
"""Explore candidate tables for joint distribution improvement."""
from gbgsynth.api_client import PxWebClient

client = PxWebClient()

# Tables to explore
tables = {
    # 1. Income source 18+ (may have age breakdown)
    'HuvudInk_18': 'Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/20_HuvudInk_PRI.px',
    
    # 2. Household type × number of children
    'HH_typ_barn': 'Befolkning/Hushåll/13_HHTypBarn_PRI.px',
    
    # 3. Earned income by education level  
    'InkomstUtbildning': 'Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/23_InkomsterUtbildning_PRI.px',
    
    # 4. Household type (detailed)
    'HH_typ': 'Befolkning/Hushåll/10_HHtyp_PRI.px',
    
    # 5. Pop by household type
    'Folk_HHtyp': 'Befolkning/Folkmängd/Folkmängd helår/17_FolkHHtyp_PRI.px',
    
    # 6. Pop by HH size × housing type (joint!)
    'Folk_HHstorlek_Hustyp': 'Befolkning/Folkmängd/Folkmängd helår/15_FolkHHstHustyp_PRI.px',
    
    # 7. Pop by HH position (hushållsställning)
    'Folk_HHstallning': 'Befolkning/Folkmängd/Folkmängd helår/14_FolkHHstllnng_PRI.px',
    
    # 8. HH by size × tenure
    'HH_storlek_upplat': 'Befolkning/Hushåll/12_HHstUpplFrm_PRI.px',

    # 9. New-style income source table 
    'HuvudInk_new': 'Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/30_HuvudInk_PRI.px',
}

for name, path in tables.items():
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
            for v, t in zip(vals[:25], texts[:25]):
                print(f"    {v} = {t}")
            if len(vals) > 25:
                print(f"    ... ({len(vals)} total)")
    except Exception as e:
        print(f"  ERROR: {e}")

# Now fetch actual data for Haga from the most interesting tables
print("\n\n" + "="*70)
print("ACTUAL DATA FOR HAGA (107)")
print("="*70)

# Try the household type × children table
for name, path in [
    ('HH_typ_barn', 'Befolkning/Hushåll/13_HHTypBarn_PRI.px'),
    ('Folk_HHstallning', 'Befolkning/Folkmängd/Folkmängd helår/14_FolkHHstllnng_PRI.px'),
]:
    print(f"\n--- {name} (Haga, 2024) ---")
    try:
        df = client.query_all_variables(path, '107 Haga', 2024)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.to_string())
    except Exception as e:
        print(f"ERROR: {e}")
        # Try 2023
        try:
            df = client.query_all_variables(path, '107 Haga', 2023)
            print(f"Shape (2023): {df.shape}")
            print(df.to_string())
        except Exception as e2:
            print(f"ERROR 2023: {e2}")
