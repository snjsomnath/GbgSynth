# GbgSynth - Synthetic Population Generator for Gothenburg

![Tests](https://img.shields.io/badge/tests-124%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A modular Python library for procedurally generating synthetic populations using the Gothenburg PxWeb API.

## Features

- **Simple API**: One-liner synthesis with `city.synthesize("Haga")`
- **Bundled Data**: 96 neighbourhood names, codes, and boundaries included (no API calls to browse)
- **Multiple Synthesis Methods**: Top-down (default), IPF, and greedy matching
- **Agent-Based Modeling**: Individual agents with demographics and socioeconomic attributes
- **Household Synthesis**: Relationship modeling with biological and age constraints
- **Housing Type Integration**: Links households to building types (Småhus, Flerbostadshus)

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from gbgsynth import GbgSynth

# One-liner synthesis
city = GbgSynth(year=2024)
haga = city.synthesize("Haga")  # Generates population immediately
haga.save()  # Saves individuals.csv and households.csv

# Access data
print(f"Population: {len(haga.individuals)}")
print(f"Households: {len(haga.households)}")
```

## API Reference

### Browse Areas (No API Call Needed)

```python
from gbgsynth import GbgSynth

# List all neighbourhood names
GbgSynth.list_areas()
# ['Kungsladugård', 'Sanna', 'Majorna', 'Haga', ...]

# Look up area code
GbgSynth.get_area_code("Haga")  # Returns '107'

# Get all areas as dict
city = GbgSynth()
city.get_all_areas()  # {'107': '107 Haga', ...}
```

### Generate Population

```python
city = GbgSynth(year=2024)

# Method 1: One-liner (generates immediately)
area = city.synthesize("Haga")

# Method 2: Step-by-step
area = city.get_area("Haga")  # or "107" or "107 Haga"
area.generate()  # Top-down method (default)
# area.generate(use_ipf=True)       # IPF method
# area.generate(use_topdown=False)  # Greedy matching
```

### Save Results

```python
# Simple: saves to current directory
area.save()  # Creates 107_Haga_individuals.csv and 107_Haga_households.csv

# Custom directory and prefix
area.save("output/", prefix="haga_2024")

# Or save individually
area.save_to_csv("individuals.csv")
area.save_households_to_csv("households.csv")

# Get as DataFrames (no file I/O)
dfs = area.to_dataframes()
dfs['individuals'].head()
dfs['households'].describe()
```

### Geographic Boundaries

```python
# Get boundary for one area (requires geopandas)
boundary = GbgSynth.get_boundary("107")

# Get all boundaries as GeoDataFrame
gdf = GbgSynth.get_all_boundaries()
gdf.plot()  # Quick visualization
```

## Batch Processing

```python
from gbgsynth import GbgSynth

city = GbgSynth(year=2024)

# Process all areas
for name in GbgSynth.list_areas():
    try:
        area = city.synthesize(name)
        area.save("output/")
    except Exception as e:
        print(f"Failed {name}: {e}")
```

## Architecture
metadata = client.fetch_metadata("Befolkning/Folkmängd/Folkmängd helår/63_FolkmHHtypPRI.px")

# Custom query
data = client.query_table(
    table_path="Övrigt/Personbilar/10_Bilar_PRI.px",
    area_code="107",
    year=2024
)
```

## Architecture

### Core Components

1. **PxWebClient** (`api_client.py`): Handles all API communication
2. **Config** (`config/`): JSON mappings and table definitions
3. **Models** (`models.py`): Agent and Household classes
4. **Synthesizer** (`synthesizer.py`): Population generation algorithms
5. **GbgArea** (`area.py`): Area-specific synthesis orchestrator

### Synthesis Methods

The library supports three synthesis approaches:

#### 1. Top-Down Constrained (Default)
```python
area.generate()  # use_topdown=True by default
```
Starts from household-level constraints and allocates individuals top-down. Anchors exact household containers first, then fills with appropriate individuals. Best for areas with complex household structures.

#### 2. IPF (Iterative Proportional Fitting)
```python
area.generate(use_ipf=True)
```
Uses iterative proportional fitting to match joint distributions of age, sex, and household type more precisely.

#### 3. Greedy Matching
```python
area.generate(use_topdown=False)
```
- **Household Creation**: Generate empty households based on size statistics
- **Couple Formation**: Match partners with age constraints (±10 years)
- **Child Assignment**: Place children with biological age constraints (18+ year gap)
- **Single Placement**: Fill remaining 1-person households
- **Attribute Assignment**: Assign income, status based on neighborhood distributions

### Constraints

- **Partner Age Difference**: Maximum 10 years
- **Biological Parent Age**: Minimum 18 years older than children
- **Housing Compatibility**: Household size must match building capacity

## Data Tables Used

| Table ID | Description | Variables |
|----------|-------------|-----------|
| `63_FolkmHHtypPRI.px` | Population by household type | Area, Age, Sex, HH Role |
| `31_HHStorlHustyp_PRI.px` | Household size by building type | Area, Size, House Type |
| `10_Bilar_PRI.px` | Car ownership | Area, Year |
| `HuvudInk_PRI.px` | Income distribution | Area, Decile |

## Privacy Handling

The library uses Python's `logging` module to track "Sekretess" (privacy suppression) in source data where totals don't match due to small cell sizes.

```python
import logging
logging.basicConfig(level=logging.INFO)

# Will log warnings when data gaps are detected
area.generate()
```

## Contributing

Contributions welcome! Key areas for improvement:

- Additional validation constraints
- Spatial allocation algorithms
- Multi-year temporal synthesis
- Enhanced income modeling

## License

MIT License

## Citation

If you use this library in research, please cite:

```
GbgSynth: A Synthetic Population Generator for Gothenburg
Version 0.1.0 (2026)
```
