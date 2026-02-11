"""
Configuration loader for GbgSynth.

Provides centralized access to table mappings and synthesis constraints.
"""

import json
import os
from typing import Dict, Any, Optional


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

        # ── Pre-built position lookup tables ──────────────────────────
        # Built from the HOUSEHOLD_POSITION.value_mappings in table_mapping.json.
        # These are the canonical Swedish→internal translations.  Every
        # call-site that previously used fragile substring matching should
        # use one of these dicts instead.
        pos_map = self.get_value_mappings('HOUSEHOLD_POSITION').get('hh_position', {})

        # Normalised (lower-cased, stripped) exact lookup: Swedish → role
        #   "Person i gift par/registrerat partnerskap" → "cohabiting"
        #   "Personer i samboförhållande"               → "cohabiting"
        #   "Ensamstående förälder"                     → "single_parent"  (*)
        #   "Barn"                                      → "child"
        #   "Ensamboende"                               → "single"
        #   "Ej ensamboende personer, övriga"           → "other"
        #   "Uppgift saknas …"                          → "unknown"
        #
        # (*) The JSON says "single" for Ensamstående förälder because
        # the HOUSEHOLD_POSITION table lumps them; we override to
        # "single_parent" which is what the synthesis engine needs.
        self._position_to_role: Dict[str, str] = {}
        for swe_label, eng_role in pos_map.items():
            key = swe_label.strip().lower()
            role = eng_role
            # Fix: Ensamstående förälder is single_parent, not single
            if 'ensamstående förälder' in key:
                role = 'single_parent'
            self._position_to_role[key] = role

        # 5 collapsed display categories for compare_to_marginals()
        # maps the same canonical labels → display names
        _collapsed = {
            'person i gift par/registrerat partnerskap': 'Sammanboende',
            'personer i samboförhållande':               'Sammanboende',
            'ensamstående förälder':                     'Ensam förälder',
            'barn':                                      'Barn',
            'ensamboende':                               'Ensamboende',
            'ej ensamboende personer, övriga':           'Övriga',
        }
        self._position_to_collapsed: Dict[str, str] = {}
        for swe_label in pos_map:
            key = swe_label.strip().lower()
            self._position_to_collapsed[key] = _collapsed.get(key, 'Uppgift saknas')

        # 7 detailed display categories for _compare_joint_role_age_sex()
        _detailed = {
            'person i gift par/registrerat partnerskap': 'Gift/reg.partner',
            'personer i samboförhållande':               'Sambo',
            'ensamstående förälder':                     'Ensam förälder',
            'barn':                                      'Barn',
            'ensamboende':                               'Ensamboende',
            'ej ensamboende personer, övriga':           'Övriga',
        }
        self._position_to_detailed: Dict[str, str] = {}
        for swe_label in pos_map:
            key = swe_label.strip().lower()
            self._position_to_detailed[key] = _detailed.get(key, 'Uppgift saknas')

        # Aggregate household role lookup (BEFOLKNING_HH table)
        #   "Ensamstående" → "single"
        #   "Sammanboende" → "cohabiting"
        #   "Övriga hushåll" → "other"
        hh_role_map = self.get_value_mappings('BEFOLKNING_HH').get('hh_role', {})
        self._hh_role_lookup: Dict[str, str] = {
            k.strip().lower(): v for k, v in hh_role_map.items()
        }

        # Sex lookup  (covers both HOUSEHOLD_POSITION and BEFOLKNING_HH tables)
        sex_map = self.get_value_mappings('HOUSEHOLD_POSITION').get('sex', {})
        self._sex_lookup: Dict[str, str] = {
            k.strip().lower(): v for k, v in sex_map.items()
        }
        # Also build Swedish ← English reverse sex map for display
        self._sex_reverse: Dict[str, str] = {v: k for k, v in sex_map.items()}

        # ── Housing-type lookup (HOUSEHOLD_SIZE + DWELLING_SIZE tables) ─
        ht_map = self.get_value_mappings('HOUSEHOLD_SIZE').get('house_type', {})
        # Supplement with DWELLING_SIZE which adds "Övriga hus"
        ht_map2 = self.get_value_mappings('DWELLING_SIZE').get('house_type', {})
        combined_ht = {**ht_map, **ht_map2}
        self._house_type_lookup: Dict[str, str] = {
            k.strip().lower(): v for k, v in combined_ht.items()
        }
        # Also add common census extras not in JSON
        self._house_type_lookup.setdefault('specialbostad, övriga hus', 'special_housing')
        self._house_type_lookup.setdefault('uppgift saknas', 'unknown')
        # Reverse: English → Swedish  (e.g. 'detached_house' → 'Småhus')
        self._house_type_reverse: Dict[str, str] = {v: k for k, v in combined_ht.items()}

        # ── Education-level lookup (EDUCATION_LEVEL table) ────────────
        edu_map = self.get_value_mappings('EDUCATION_LEVEL').get('education_level', {})
        self._education_lookup: Dict[str, str] = {
            k.strip().lower(): v for k, v in edu_map.items()
        }
        # Also add the short-form labels from INCOME table if present
        edu_map_income = self.get_value_mappings('INCOME').get('education_level', {})
        for k, v in edu_map_income.items():
            self._education_lookup.setdefault(k.strip().lower(), v)
        # Uppgift saknas
        self._education_lookup.setdefault('uppgift saknas', 'unknown')

        # ── Income-source lookup (INCOME_SOURCE table) ────────────────
        src_map = self.get_value_mappings('INCOME_SOURCE').get('income_source', {})
        self._income_source_lookup: Dict[str, str] = {
            k.strip().lower(): v for k, v in src_map.items()
        }
        # Reverse: English → Swedish
        self._income_source_reverse: Dict[str, str] = {v: k for k, v in src_map.items()}

        # ── Income-standard lookup (INCOME table) ─────────────────────
        std_map = self.get_value_mappings('INCOME').get('income_standard', {})
        self._income_standard_lookup: Dict[str, str] = {
            k.strip().lower(): v for k, v in std_map.items()
        }

    # ── Position / role translation ──────────────────────────────────

    def translate_position(self, position_str) -> str:
        """Translate a Swedish household position label to an internal role.

        Uses exact lookup from ``HOUSEHOLD_POSITION.value_mappings`` in
        *table_mapping.json*.  Returns ``'unknown'`` for unrecognised
        labels (including NaN / ``'Uppgift saknas'``).

        Returns one of: ``'cohabiting'``, ``'single'``, ``'single_parent'``,
        ``'child'``, ``'other'``, ``'unknown'``.
        """
        import pandas as pd
        if pd.isna(position_str):
            return 'single'
        key = str(position_str).strip().lower()
        return self._position_to_role.get(key, 'unknown')

    def translate_position_collapsed(self, position_str) -> str:
        """Translate a Swedish position label to one of 5 display categories.

        Returns one of: ``'Sammanboende'``, ``'Ensam förälder'``, ``'Barn'``,
        ``'Ensamboende'``, ``'Övriga'``, ``'Uppgift saknas'``.
        """
        key = str(position_str).strip().lower()
        return self._position_to_collapsed.get(key, 'Uppgift saknas')

    def translate_position_detailed(self, position_str) -> str:
        """Translate a Swedish position label to one of 7 display categories.

        Returns one of: ``'Gift/reg.partner'``, ``'Sambo'``,
        ``'Ensam förälder'``, ``'Barn'``, ``'Ensamboende'``, ``'Övriga'``,
        ``'Uppgift saknas'``.
        """
        key = str(position_str).strip().lower()
        return self._position_to_detailed.get(key, 'Uppgift saknas')

    def is_couple_hh_type(self, hh_type_str) -> bool:
        """Check if a Swedish household-type string denotes a couple household.

        Uses the ``BEFOLKNING_HH`` role mapping.  Returns *False* for
        ``'Uppgift saknas'`` and other non-couple types.
        """
        if not hh_type_str:
            return False
        key = str(hh_type_str).strip().lower()
        return self._hh_role_lookup.get(key) == 'cohabiting'

    def translate_hh_role(self, hh_type_str) -> str:
        """Translate an aggregate household-type label to internal role.

        Uses ``BEFOLKNING_HH.value_mappings``.  Returns ``'single'``
        for unrecognised labels.
        """
        if not hh_type_str:
            return 'single'
        key = str(hh_type_str).strip().lower()
        return self._hh_role_lookup.get(key, 'single')

    def find_role_key(self, marginal_keys, role: str) -> Optional[str]:
        """Find which key in a marginal Series index matches *role*.

        Searches the ``BEFOLKNING_HH`` role mapping to find the original
        Swedish key whose English mapping equals *role*.  Falls back to
        substring search if exact match fails (marginal labels can vary
        slightly between tables).
        """
        role_lower = role.lower()
        # First: exact dict lookup
        for key in marginal_keys:
            k = str(key).strip().lower()
            mapped = self._hh_role_lookup.get(k)
            if mapped == role_lower:
                return key
        # Fallback: try the position lookup (for finer-grained tables)
        for key in marginal_keys:
            k = str(key).strip().lower()
            mapped = self._position_to_role.get(k)
            if mapped == role_lower:
                return key
        return None

    # ── Housing-type translation ─────────────────────────────────────

    def translate_house_type(self, hustyp_str) -> str:
        """Translate a Swedish Hustyp label to internal English.

        Returns one of: ``'detached_house'``, ``'apartment'``,
        ``'special_housing'``, or the original string if unknown.
        """
        if not hustyp_str:
            return 'apartment'
        key = str(hustyp_str).strip().lower()
        return self._house_type_lookup.get(key, str(hustyp_str))

    def english_to_hustyp(self, house_type: Optional[str]) -> str:
        """Reverse: English house-type → Swedish Hustyp."""
        if house_type is None:
            return 'Flerbostadshus'
        return self._house_type_reverse.get(house_type, 'Flerbostadshus')

    # ── Sex translation ──────────────────────────────────────────────

    def translate_sex(self, sex_str) -> str:
        """Translate a Swedish sex label to ``'male'`` or ``'female'``."""
        if not sex_str:
            return 'male'
        key = str(sex_str).strip().lower()
        return self._sex_lookup.get(key, 'male')

    def sex_to_swedish(self, sex_en: str) -> str:
        """Reverse: ``'male'``/``'female'`` → Swedish label."""
        return self._sex_reverse.get(sex_en, sex_en)

    # ── Education-level translation ──────────────────────────────────

    def translate_education(self, edu_str) -> str:
        """Translate a Swedish education-level label to internal English."""
        if not edu_str:
            return 'unknown'
        key = str(edu_str).strip().lower()
        return self._education_lookup.get(key, 'unknown')

    # ── Income-source translation ────────────────────────────────────

    def translate_income_source(self, source_str) -> str:
        """Translate a Swedish income-source label to internal English."""
        if not source_str:
            return 'no_income'
        key = str(source_str).strip().lower()
        return self._income_source_lookup.get(key, str(source_str))

    def income_source_to_swedish(self, source_en: str) -> str:
        """Reverse: English income source → Swedish label."""
        return self._income_source_reverse.get(source_en, source_en)

    # ── Income-standard translation ──────────────────────────────────

    def translate_income_standard(self, std_str) -> str:
        """Translate a Swedish income-standard label to internal English."""
        if not std_str:
            return 'adequate_income'
        key = str(std_str).strip().lower()
        return self._income_standard_lookup.get(key, str(std_str))

    def is_low_income_label(self, std_str) -> bool:
        """Check whether a Swedish income-standard label means low income."""
        return self.translate_income_standard(std_str) == 'low_income'

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
