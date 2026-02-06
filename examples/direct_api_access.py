#!/usr/bin/env python
"""
Direct API access for custom queries.

This example shows how to use the PxWebClient directly
to fetch metadata and query specific census tables.
"""

from gbgsynth.api_client import PxWebClient


def main():
    client = PxWebClient()
    
    # Fetch metadata
    print("Fetching metadata for population table...")
    metadata = client.fetch_metadata(
        "Befolkning/Folkmängd/Folkmängd helår/63_FolkmHHtypPRI.px"
    )
    
    print(f"Table title: {metadata.get('title', 'N/A')}")
    print(f"Variables: {len(metadata.get('variables', []))}")
    
    # Query specific data
    print("\nQuerying car ownership data for area 107...")
    car_data = client.query_table(
        table_path="Övrigt/Personbilar/10_Bilar_PRI.px",
        area_code="107",
        year=2023
    )
    
    print(car_data)
    
    # The car data shows:
    # - Personbilar: Total number of cars in the area
    # - Folkmängd: Total population
    print("\nThis data is used for propensity-based car assignment.")


if __name__ == "__main__":
    main()
