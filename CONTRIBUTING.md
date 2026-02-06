# GbgSynth Development Guide

## Installation for Development

```bash
# Clone the repository
git clone https://github.com/yourusername/gbgsynth.git
cd gbgsynth

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=gbgsynth --cov-report=html
```

## Code Style

This project uses:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

```bash
# Format code
black gbgsynth/

# Check linting
flake8 gbgsynth/

# Type checking
mypy gbgsynth/
```

## Project Structure

```
GbgSynth/
├── gbgsynth/              # Main library package
│   ├── __init__.py        # Package initialization
│   ├── api_client.py      # PxWeb API client
│   ├── config.py          # Configuration loader
│   ├── models.py          # Agent and Household classes
│   ├── synthesizer.py     # Population synthesis engine
│   ├── area.py            # Area-specific orchestrator
│   ├── gbgsynth.py        # Main user interface
│   └── config/            # Configuration files
│       └── table_mapping.json
├── examples/              # Usage examples
│   ├── quickstart.py
│   └── usage_examples.py
├── tests/                 # Test suite (to be added)
├── setup.py               # Package setup
├── requirements.txt       # Runtime dependencies
├── requirements-dev.txt   # Development dependencies
├── README.md              # User documentation
└── CONTRIBUTING.md        # This file
```

## Adding New Features

### Adding a New Census Table

1. Update `config/table_mapping.json`:
```json
{
  "tables": {
    "NEW_TABLE": {
      "id": "path/to/table.px",
      "description": "Description",
      "variables": {
        "Swedish Name": "english_name"
      }
    }
  }
}
```

2. Add a fetch method in `area.py`:
```python
def _fetch_new_data(self) -> pd.DataFrame:
    table_path = self.config.get_table_id('NEW_TABLE')
    return self.client.query_all_variables(table_path, self.area_code, self.year)
```

3. Integrate into synthesis pipeline in `synthesizer.py`

### Adding New Synthesis Constraints

1. Add to `config/table_mapping.json`:
```json
{
  "synthesis_constraints": {
    "new_constraint": 10
  }
}
```

2. Implement in `synthesizer.py`:
```python
def _apply_new_constraint(self, agents: List[Agent]) -> List[Agent]:
    constraint_value = self.constraints['new_constraint']
    # Implementation
    return filtered_agents
```

## Testing Guidelines

- Write tests for all new features
- Aim for >80% code coverage
- Use fixtures for common test data
- Mock API calls to avoid network dependencies

Example test structure:
```python
def test_agent_creation():
    agent = Agent(agent_id=1, age=30, sex='male')
    assert agent.is_adult()
    assert not agent.is_child()
```

## Documentation

- Use Google-style docstrings
- Update README.md for user-facing changes
- Add examples to `examples/` directory

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes and commit: `git commit -am "Add my feature"`
3. Run tests and linting: `pytest && black . && flake8`
4. Push and create PR: `git push origin feature/my-feature`

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Tag release: `git tag v0.2.0`
4. Build: `python setup.py sdist bdist_wheel`
5. Upload: `twine upload dist/*`

## Questions?

Open an issue or contact the maintainers.
