"""Summarize the distributions from candidate tables for Haga."""
import pandas as pd
import requests
from gbgsynth.api_client import PxWebClient

client = PxWebClient()

# ── 1. Education Level from 23_InkomsterUtbildning ──────────────────────
print("=" * 70)
print("1. EDUCATION LEVEL DISTRIBUTION (23_InkomsterUtbildning_PRI.px)")
print("=" * 70)

path = "Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/23_InkomsterUtbildning_PRI.px"
df = client.query_all_variables(path, "107 Haga", 2023)

# Folkmängd rows, individual sexes, non-total education levels, total age
edu = df[
    (df["Tabellvärde"] == "Folkmängd")
    & (df["Kön"] != "Båda kön")
    & (df["Utbildningsnivå"] != "Totalt (alla utbildningsnivåer)")
    & (df["Ålder"] == "18- år")
]
pivot = edu.pivot_table(values="Antal", index="Utbildningsnivå", columns="Kön", aggfunc="sum")
pivot["Total"] = pivot.sum(axis=1)
print("\nBy sex (adults 18+):")
print(pivot.to_string())
print(f"\nGrand total: {int(pivot['Total'].sum())}")

# By age group
print("\nBy age group (both sexes, non-total edu levels):")
edu_age = df[
    (df["Tabellvärde"] == "Folkmängd")
    & (df["Kön"] == "Båda kön")
    & (df["Utbildningsnivå"] != "Totalt (alla utbildningsnivåer)")
    & (df["Ålder"] != "18- år")
]
pivot_age = edu_age.pivot_table(
    values="Antal", index="Ålder", columns="Utbildningsnivå", aggfunc="sum"
)
pivot_age["Total"] = pivot_age.sum(axis=1)
print(pivot_age.to_string())

# ── 2. Income Source from 20_HuvudInk ──────────────────────────────────
print("\n" + "=" * 70)
print("2. INCOME SOURCE DISTRIBUTION (20_HuvudInk_PRI.px)")
print("=" * 70)

table_path = "Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/20_HuvudInk_PRI.px"
url = f"{client.BASE_URL}{table_path}"
query = {
    "query": [
        {"code": "Område", "selection": {"filter": "item", "values": ["6"]}},
        {"code": "Kön", "selection": {"filter": "all", "values": ["*"]}},
        {"code": "Huvudsaklig inkomstkälla", "selection": {"filter": "all", "values": ["*"]}},
        {"code": "År", "selection": {"filter": "item", "values": ["2023"]}},
    ],
    "response": {"format": "json"},
}
resp = requests.post(url, json=query, timeout=30)
ik = client._parse_json_response(resp.json())

pivot_ik = ik.pivot_table(
    values="Antal", index="Huvudsaklig inkomstkälla", columns="Kön", aggfunc="sum"
)
pivot_ik["Total"] = pivot_ik.sum(axis=1)
print("\nBy sex (adults 20+):")
print(pivot_ik.to_string())
print(f"\nGrand total: {int(pivot_ik['Total'].sum())}")

# ── 3. Background from 33_InkomsterBakgrund ────────────────────────────
print("\n" + "=" * 70)
print("3. BACKGROUND DISTRIBUTION (33_InkomsterBakgrund_PRI.px)")
print("=" * 70)

path3 = "Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/33_InkomsterBakgrund_PRI.px"
df3 = client.query_all_variables(path3, "107 Haga", 2023)

bg = df3[
    (df3["Tabellvärde"] == "Folkmängd")
    & (df3["Kön"] != "Båda kön")
    & (df3["Svensk/Utländsk bakgrund"] != "Totalt")
    & (df3["Ålder"] == "18- år")
]
pivot_bg = bg.pivot_table(
    values="Antal", index="Svensk/Utländsk bakgrund", columns="Kön", aggfunc="sum"
)
pivot_bg["Total"] = pivot_bg.sum(axis=1)
print("\nBy sex (adults 18+):")
print(pivot_bg.to_string())
print(f"\nGrand total: {int(pivot_bg['Total'].sum())}")

# ── 4. Current Income Standard from 10_InkStandard ─────────────────────
print("\n" + "=" * 70)
print("4. CURRENT INCOME STANDARD (10_InkStandard_PRI.px) - for reference")
print("=" * 70)

path4 = "Inkomst och utbildning/Inkomster/Socioekonomi/10_InkStandard_PRI.px"
df4 = client.query_all_variables(path4, "107 Haga", 2023)
print("\nAll rows:")
for _, row in df4.iterrows():
    cols = [c for c in df4.columns if c not in ["Område", "År"]]
    vals = " | ".join(f"{c}: {row[c]}" for c in cols)
    print(f"  {vals}")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: What we can compare against")
print("=" * 70)
print("""
CURRENT (broken):
  - Income Standard: 3 categories (low/not_low/not_in_household)
    Problem: 'not_in_household' always -100% error → inflates MAPE to ~35%

PROPOSED REPLACEMENTS:
  1. Education Level (4 categories): Förgymnasial, Gymnasial, Eftergymnasial, Uppgift saknas
     - Available by age × sex → can assign probabilistically during synthesis
     - All categories representable by the synthesizer
     - Adds a NEW attribute to synthetic individuals

  2. Income Source (9 categories): Work, Unemployment, Studies, Pension, etc.
     - Available by sex only (no age breakdown)
     - All categories representable
     - NOTE: Uses numeric area indices (needs mapping)

  3. Background (2 categories): Svensk/Utländsk bakgrund
     - Available by age × sex
     - All categories representable
     - User said 'maybe not necessary'
""")
