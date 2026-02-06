"""
Tests for configuration management.
"""

import pytest
from gbgsynth.config import Config


class TestConfig:
    """Tests for the Config class."""

    @pytest.fixture
    def config(self):
        """Create a Config instance for testing."""
        return Config()

    def test_config_loads(self, config):
        """Test that config loads without errors."""
        assert config is not None

    def test_tables_property(self, config):
        """Test that tables property returns a dictionary."""
        tables = config.tables
        assert isinstance(tables, dict)

    def test_constraints_property(self, config):
        """Test that constraints property returns a dictionary."""
        constraints = config.constraints
        assert isinstance(constraints, dict)

    def test_get_table_id_existing(self, config):
        """Test getting table ID for existing table."""
        # Skip if no tables configured
        if not config.tables:
            pytest.skip("No tables configured")
        
        table_name = list(config.tables.keys())[0]
        table_id = config.get_table_id(table_name)
        assert isinstance(table_id, str)

    def test_get_table_id_nonexistent(self, config):
        """Test getting table ID for non-existent table returns empty string."""
        table_id = config.get_table_id('NONEXISTENT_TABLE')
        assert table_id == ''

    def test_get_variable_mapping(self, config):
        """Test getting variable mapping returns dictionary."""
        if not config.tables:
            pytest.skip("No tables configured")
        
        table_name = list(config.tables.keys())[0]
        mapping = config.get_variable_mapping(table_name)
        assert isinstance(mapping, dict)

    def test_get_value_mappings(self, config):
        """Test getting value mappings returns dictionary."""
        if not config.tables:
            pytest.skip("No tables configured")
        
        table_name = list(config.tables.keys())[0]
        mappings = config.get_value_mappings(table_name)
        assert isinstance(mappings, dict)

    def test_age_group_mappings(self, config):
        """Test age group mappings property."""
        mappings = config.age_group_mappings
        assert isinstance(mappings, dict)

    def test_household_size_mappings(self, config):
        """Test household size mappings property."""
        mappings = config.household_size_mappings
        assert isinstance(mappings, dict)

    def test_translate_column_existing(self, config):
        """Test translating an existing column name."""
        if not config.tables:
            pytest.skip("No tables configured")
        
        table_name = list(config.tables.keys())[0]
        var_mapping = config.get_variable_mapping(table_name)
        
        if var_mapping:
            swedish_name = list(var_mapping.keys())[0]
            english_name = config.translate_column(table_name, swedish_name)
            assert english_name == var_mapping[swedish_name]

    def test_translate_column_nonexistent(self, config):
        """Test translating a non-existent column returns original."""
        result = config.translate_column('ANY_TABLE', 'original_name')
        assert result == 'original_name'
