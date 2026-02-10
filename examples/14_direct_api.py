"""
14 · Direct API Access
=======================
Use the PxWeb API client directly to fetch metadata and raw
census tables from the Gothenburg statistics database.
"""

import logging
logging.basicConfig(level=logging.ERROR)

from gbgsynth.api_client import PxWebClient

client = PxWebClient()

# Fetch table metadata
meta = client.fetch_metadata(
    "Befolkning/Folkmängd/Folkmängd helår/63_FolkmHHtypPRI.px"
)
print(f"Table:     {meta.get('title', 'N/A')}")
print(f"Variables: {len(meta.get('variables', []))}")
for v in meta.get("variables", []):
    vals = v.get("values", [])
    print(f"  • {v.get('text', '?')} ({len(vals)} values)")

# Query a specific table
print("\nCar ownership data for Haga (107):")
car_data = client.query_table(
    table_path="Övrigt/Personbilar/10_Bilar_PRI.px",
    area_code="107 Haga",
    year=2023,
)
print(car_data.to_string())
