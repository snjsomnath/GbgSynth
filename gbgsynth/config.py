"""
Configuration loader for GbgSynth.

Provides centralized access to table mappings and synthesis constraints.
"""

import json
import os
from typing import Dict, Any


class Config:
    """
    Configuration manager for the GbgSynth library.
    Loads and provides access to table mappings and constraints.
    """

    def __init__(self):
        """Load configuration from JSON file."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            'config',
            'table_mapping.json'
        )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)

    @property
    def tables(self) -> Dict[str, Any]:
        """Get table mapping configuration."""
        return self._config.get('tables', {})

    def get_table_id(self, table_name: str) -> str:
        """
        Get the PxWeb table ID for a named table.

        Args:
            table_name: Logical table name (e.g., 'BEFOLKNING_HH')

        Returns:
            Full table path for API queries
        """
        return self.tables.get(table_name, {}).get('id', '')

    def get_variable_mapping(self, table_name: str) -> Dict[str, str]:
        """
        Get Swedish-to-English variable mappings for a table.

        Args:
            table_name: Logical table name

        Returns:
            Dictionary mapping Swedish headers to English variable names
        """
        return self.tables.get(table_name, {}).get('variables', {})

    def get_value_mappings(self, table_name: str) -> Dict[str, Dict[str, str]]:
        """
        Get value mappings for categorical variables.

        Args:
            table_name: Logical table name

        Returns:
            Dictionary of variable mappings (e.g., {"sex": {"Män": "male"}})
        """
        return self.tables.get(table_name, {}).get('value_mappings', {})

    @property
    def age_group_mappings(self) -> Dict[str, Dict[str, int]]:
        """Get age group range definitions."""
        return self._config.get('age_group_mappings', {})

    @property
    def household_size_mappings(self) -> Dict[str, int]:
        """Get household size mappings."""
        return self._config.get('household_size_mappings', {})

    @property
    def constraints(self) -> Dict[str, int]:
        """Get synthesis constraints (age gaps, thresholds, etc.)."""
        return self._config.get('synthesis_constraints', {})

    def translate_column(self, table_name: str, swedish_name: str) -> str:
        """
        Translate a Swedish column name to English.

        Args:
            table_name: Logical table name
            swedish_name: Swedish column header

        Returns:
            English variable name
        """
        mapping = self.get_variable_mapping(table_name)
        return mapping.get(swedish_name, swedish_name)

    def translate_value(self, table_name: str, variable: str, swedish_value: str) -> str:
        """
        Translate a Swedish categorical value to English.

        Args:
            table_name: Logical table name
            variable: Variable name
            swedish_value: Swedish value

        Returns:
            English value or original if no mapping exists
        """
        value_maps = self.get_value_mappings(table_name)
        if variable in value_maps:
            return value_maps[variable].get(swedish_value, swedish_value)
        return swedish_value
