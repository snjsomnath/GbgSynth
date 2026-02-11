"""
Marginal comparison and validation for synthesised populations.

Compares the synthesised population against census marginals across
multiple dimensions (sex, age, household role, education, income
source, etc.) and computes goodness-of-fit statistics including
Voas & Williamson (2001) metrics.
"""

import logging
import re
import statistics
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from gbgsynth.config import Config
from gbgsynth.models import Agent, Household

logger = logging.getLogger(__name__)

__all__ = ['MarginalComparator']


class MarginalComparator:
    """Validates a synthesised population against census marginals.

    Parameters
    ----------
    individuals : list[Agent]
    households : list[Household]
    marginals : dict
        Dictionary of census DataFrames (population, household, …).
    config : Config
        Translation helpers (``translate_position_collapsed``, etc.).
    area_name : str
        For report titles.
    year : int
        Census year.
    """

    def __init__(
        self,
        individuals: List[Agent],
        households: List[Household],
        marginals: dict,
        area_name: str,
        year: int,
        config=None,
    ):
        self.individuals = individuals
        self.households = households
        self._marginals = marginals
        self.config = config if config is not None else Config()
        self.area_name = area_name
        self.year = year

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compare(
        self,
        print_report: bool = True,
        use_logging: bool = False,
    ) -> dict:
        """Run all comparisons and return results.

        Returns
        -------
        dict
            Per-dimension comparison tables plus ``'overall'`` fit
            statistics.
        """
        comparisons = {}

        comparisons['sex'] = self._compare_sex_distribution()
        comparisons['age'] = self._compare_age_distribution()
        comparisons['role'] = self._compare_role_distribution()
        comparisons['household_size'] = (
            self._compare_household_size_distribution())
        comparisons['housing_type'] = (
            self._compare_housing_type_distribution())
        comparisons['education'] = self._compare_education_distribution()
        comparisons['income_source'] = (
            self._compare_income_source_distribution())
        comparisons['median_income'] = self._compare_median_income()
        comparisons['hh_type_children'] = self._compare_hh_type_children()
        comparisons['joint_role_age_sex'] = (
            self._compare_joint_role_age_sex())

        # ── aggregate fit statistics ─────────────────────────────────
        excluded_from_mape = {
            'overall', 'median_income',
            'hh_type_children', 'joint_role_age_sex',
        }
        all_actual: list = []
        all_synth: list = []
        all_pct_errors: list = []
        dim_metrics: dict = {}

        for cat, data in comparisons.items():
            if cat in excluded_from_mape:
                continue
            if data and 'comparison' in data:
                dim_actual: list = []
                dim_synth: list = []
                for row in data['comparison']:
                    if row.get('exclude_from_mape'):
                        continue
                    dim_actual.append(row['actual'])
                    dim_synth.append(row['synth'])
                    all_actual.append(row['actual'])
                    all_synth.append(row['synth'])
                    if row['actual'] > 0:
                        pct_err = (abs(row['synth'] - row['actual'])
                                   / row['actual'] * 100)
                        all_pct_errors.append(pct_err)

                if dim_actual:
                    ea = np.array(dim_actual, dtype=float)
                    oa = np.array(dim_synth, dtype=float)
                    da = oa - ea

                    tae_d = float(np.sum(np.abs(da)))
                    n_d = float(np.sum(ea))
                    sae_d = tae_d / n_d if n_d > 0 else 0.0

                    e_safe = np.where(ea == 0, 1.0, ea)
                    chi2_d = float(np.sum(da ** 2 / e_safe))

                    sum_e = n_d if n_d > 0 else 1.0
                    sum_o = (float(np.sum(oa))
                             if np.sum(oa) > 0 else 1.0)
                    t = oa / sum_o
                    p = np.where(ea != 0, ea / sum_e, 1.0 / sum_e)
                    correction = 1.0 / (2.0 * sum_e)
                    corrected = np.maximum(
                        np.abs(t - p) - correction, 0.0)
                    denom = np.sqrt(p * (1.0 - p) / sum_e)
                    denom_safe = np.where(denom == 0, 1.0, denom)
                    z2_d = float(
                        np.sum((corrected / denom_safe) ** 2))

                    dof_d = len(dim_actual)
                    chi2_p_d = (
                        float(1.0 - sp_stats.chi2.cdf(chi2_d, dof_d))
                        if dof_d > 0 else 1.0)
                    z2_p_d = (
                        float(1.0 - sp_stats.chi2.cdf(z2_d, dof_d))
                        if dof_d > 0 else 1.0)

                    dim_metrics[cat] = {
                        'tae': tae_d, 'sae': sae_d,
                        'chi2': chi2_d, 'chi2_p': chi2_p_d,
                        'z2': z2_d, 'z2_p': z2_p_d,
                        'dof': dof_d,
                    }
                    data['fit'] = dim_metrics[cat]

        if all_actual:
            actual_arr = np.array(all_actual, dtype=float)
            synth_arr = np.array(all_synth, dtype=float)
            diff_arr = synth_arr - actual_arr

            mape = (float(np.mean(all_pct_errors))
                    if all_pct_errors else 0.0)

            weighted_pct = []
            for a, s in zip(all_actual, all_synth):
                if a > 0:
                    weighted_pct.append(abs(s - a) / a * 100 * a)
            wmape = (sum(weighted_pct) / sum(all_actual)
                     if sum(all_actual) > 0 else 0.0)

            sae_values = [m['sae'] for m in dim_metrics.values()]
            chi2_p_values = [m['chi2_p'] for m in dim_metrics.values()]
            z2_p_values = [m['z2_p'] for m in dim_metrics.values()]

            comparisons['overall'] = {
                'total_actual': int(sum(all_actual)),
                'total_synth': int(sum(all_synth)),
                'rmse': float(np.sqrt(np.mean(diff_arr ** 2))),
                'mae': float(np.mean(np.abs(diff_arr))),
                'max_error': int(np.max(np.abs(diff_arr))),
                'correlation': (
                    float(np.corrcoef(actual_arr, synth_arr)[0, 1])
                    if len(actual_arr) > 1 else 1.0),
                'mape': mape,
                'wmape': wmape,
                'n_categories': len(all_actual),
                'max_pct_error': (
                    max(all_pct_errors) if all_pct_errors else 0.0),
                'sae_median': (
                    float(np.median(sae_values))
                    if sae_values else 0.0),
                'sae_max': (
                    float(np.max(sae_values))
                    if sae_values else 0.0),
                'sae_mean': (
                    float(np.mean(sae_values))
                    if sae_values else 0.0),
                'chi2_p_min': (
                    float(np.min(chi2_p_values))
                    if chi2_p_values else 1.0),
                'z2_p_min': (
                    float(np.min(z2_p_values))
                    if z2_p_values else 1.0),
                'dim_metrics': dim_metrics,
            }

        if print_report:
            self._print_comparison_report(comparisons, use_logging)

        return comparisons

    # ------------------------------------------------------------------
    # Summary / logging helpers
    # ------------------------------------------------------------------

    def get_summary_statistics(self) -> dict:
        """Compute summary statistics for the generated population."""
        total_pop = len(self.individuals)
        total_hh = len(self.households)

        hustyp_counts: dict = {}
        linked_to_buildings = 0
        for hh in self.households:
            ht = hh.assigned_hustyp or 'unassigned'
            hustyp_counts[ht] = hustyp_counts.get(ht, 0) + 1
            if hh.building_id is not None:
                linked_to_buildings += 1

        return {
            'area_code': '',  # filled in by GbgArea
            'area_name': self.area_name,
            'year': self.year,
            'total_population': total_pop,
            'total_households': total_hh,
            'avg_household_size': (
                total_pop / total_hh if total_hh > 0 else 0),
            'num_children': sum(
                1 for a in self.individuals if a.is_child()),
            'num_adults': sum(
                1 for a in self.individuals if a.is_adult()),
            'num_couples': sum(
                1 for h in self.households if h.is_couple()),
            'num_single_parent': sum(
                1 for h in self.households if h.is_single_parent()),
            'num_single_person': sum(
                1 for h in self.households if h.is_single()),
            'avg_income': (
                sum(a.income or 0 for a in self.individuals) / total_pop
                if total_pop > 0 else 0),
            'total_cars': sum(h.cars for h in self.households),
            'hustyp_smahus': hustyp_counts.get('Småhus', 0),
            'hustyp_flerbostadshus': hustyp_counts.get(
                'Flerbostadshus', 0),
            'hustyp_specialbostad': hustyp_counts.get(
                'Specialbostad', 0),
            'households_linked_to_buildings': linked_to_buildings,
        }

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Return the marginal comparison as a DataFrame."""
        comparisons = self.compare(print_report=False)
        rows = []
        for dim_key, data in comparisons.items():
            if dim_key == 'overall' or not data or 'comparison' not in data:
                continue
            for row in data['comparison']:
                rows.append({
                    'dimension': data['name'],
                    'category': row['category'],
                    'actual': row['actual'],
                    'synth': row['synth'],
                    'diff': row['diff'],
                    'error_pct': row['error_pct'],
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Per-dimension comparisons
    # ------------------------------------------------------------------

    def _compare_sex_distribution(self) -> dict:
        pop_data = self._marginals.get('population')
        if pop_data is None or pop_data.empty:
            return {}

        sex_col = 'Kön' if 'Kön' in pop_data.columns else 'sex'
        count_col = ('Antal' if 'Antal' in pop_data.columns
                     else pop_data.columns[-1])
        if sex_col not in pop_data.columns:
            return {}

        actual = pop_data.groupby(sex_col)[count_col].sum().to_dict()

        synth: dict = {}
        for ind in self.individuals:
            if 'Man' in actual or 'Kvinna' in actual:
                sex = 'Kvinna' if ind.sex == 'female' else 'Man'
            else:
                sex = 'Kvinnor' if ind.sex == 'female' else 'Män'
            synth[sex] = synth.get(sex, 0) + 1

        comparison = []
        for category in set(list(actual.keys()) + list(synth.keys())):
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        return {'name': 'Sex Distribution', 'comparison': comparison}

    def _compare_age_distribution(self) -> dict:
        pop_data = self._marginals.get('population')
        if pop_data is None or pop_data.empty:
            return {}

        age_col = ('Ålder' if 'Ålder' in pop_data.columns
                   else 'age_group')
        count_col = ('Antal' if 'Antal' in pop_data.columns
                     else pop_data.columns[-1])
        if age_col not in pop_data.columns:
            return {}

        actual = pop_data.groupby(age_col)[count_col].sum().to_dict()

        synth = {cat: 0 for cat in actual.keys()}
        for ind in self.individuals:
            for category in actual.keys():
                age_range = _parse_age_range(category)
                if (age_range
                        and age_range[0] <= ind.age <= age_range[1]):
                    synth[category] = synth.get(category, 0) + 1
                    break

        comparison = []
        for category in actual.keys():
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        comparison.sort(
            key=lambda r: (
                _parse_age_range(r['category'])[0]
                if _parse_age_range(r['category']) else 999))

        return {'name': 'Age Distribution', 'comparison': comparison}

    def _compare_household_size_distribution(self) -> dict:
        hh_data = self._marginals.get('household')
        if hh_data is None or hh_data.empty:
            return {}

        size_col = ('Hushållsstorlek'
                    if 'Hushållsstorlek' in hh_data.columns
                    else 'hh_size')
        count_col = ('Antal' if 'Antal' in hh_data.columns
                     else hh_data.columns[-1])
        if size_col not in hh_data.columns:
            return {}

        actual = hh_data.groupby(size_col)[count_col].sum().to_dict()

        size_labels = {
            1: '1 person', 2: '2 personer', 3: '3 personer',
            4: '4 personer', 5: '5 personer',
            6: '6 eller fler personer',
        }
        synth: dict = {}
        for hh in self.households:
            size = min(hh.size, 6)
            label = size_labels.get(size, f'{size} personer')
            synth[label] = synth.get(label, 0) + 1

        comparison = []
        for category in actual.keys():
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        return {
            'name': 'Household Size Distribution',
            'comparison': comparison,
        }

    def _compare_housing_type_distribution(self) -> dict:
        hh_data = self._marginals.get('household')
        if hh_data is None or hh_data.empty:
            return {}

        type_col = ('Hustyp' if 'Hustyp' in hh_data.columns
                    else 'house_type')
        count_col = ('Antal' if 'Antal' in hh_data.columns
                     else hh_data.columns[-1])
        if type_col not in hh_data.columns:
            return {}

        actual = hh_data.groupby(type_col)[count_col].sum().to_dict()

        synth: dict = {}
        for hh in self.households:
            ht = hh.assigned_hustyp or 'Okänd'
            synth[ht] = synth.get(ht, 0) + 1

        comparison = []
        for category in set(list(actual.keys()) + list(synth.keys())):
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        return {
            'name': 'Housing Type Distribution',
            'comparison': comparison,
        }

    def _compare_role_distribution(self) -> dict:
        pos_data = self._marginals.get('household_position')
        if pos_data is None or (
                hasattr(pos_data, 'empty') and pos_data.empty):
            return {}

        pos_col = None
        for col in pos_data.columns:
            if ('ställning' in col.lower()
                    or 'position' in col.lower()):
                pos_col = col
                break
        if pos_col is None:
            return {}

        count_col = ('Antal' if 'Antal' in pos_data.columns
                     else pos_data.columns[-1])

        collapsed_pos = self.config.translate_position_collapsed

        actual: dict = {}
        for _, row in pos_data.iterrows():
            pos = collapsed_pos(row[pos_col])
            val = int(row[count_col]) if pd.notna(row[count_col]) else 0
            actual[pos] = actual.get(pos, 0) + val

        if not actual:
            return {}

        hh_by_id = {h.household_id: h for h in self.households}
        synth: dict = {}

        for ind in self.individuals:
            role = ind.hh_role
            if role == 'child':
                pos = 'Barn'
            elif role == 'cohabiting':
                pos = 'Sammanboende'
            elif role == 'single_parent':
                pos = 'Ensam förälder'
            elif role == 'single':
                hh = hh_by_id.get(ind.household_id)
                if hh and len(hh.members) == 1:
                    pos = 'Ensamboende'
                elif hh and any(
                        m.hh_role == 'child' for m in hh.members):
                    pos = 'Ensam förälder'
                else:
                    pos = 'Ensamboende'
            elif role == 'other':
                pos = 'Övriga'
            else:
                pos = 'Uppgift saknas'
            synth[pos] = synth.get(pos, 0) + 1

        comparison = []
        all_positions = sorted(
            set(list(actual.keys()) + list(synth.keys())))
        for pos in all_positions:
            if pos == 'Uppgift saknas':
                continue
            act_val = actual.get(pos, 0)
            syn_val = synth.get(pos, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': pos,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        return {
            'name': 'Household Position Distribution',
            'comparison': comparison,
        }

    def _compare_education_distribution(self) -> dict:
        edu_data = self._marginals.get('education_level')
        if edu_data is None or (
                hasattr(edu_data, 'empty') and edu_data.empty):
            return {}

        edu_display = {
            'pre_secondary': 'Förgymnasial utbildning',
            'secondary': 'Gymnasial utbildning',
            'post_secondary': 'Eftergymnasial utbildning',
            'unknown': 'Uppgift saknas',
        }

        actual: dict = {}
        for _, row in edu_data.iterrows():
            metric = row.get('Tabellvärde', 'Folkmängd')
            if metric != 'Folkmängd':
                continue
            edu_sv = row['Utbildningsnivå']
            count = int(row['Antal']) if pd.notna(row['Antal']) else 0
            actual[edu_sv] = actual.get(edu_sv, 0) + count

        synth: dict = {}
        for ind in self.individuals:
            if ind.age < 18:
                continue
            edu = getattr(ind, 'education', None)
            if edu and edu != 'child':
                display_name = edu_display.get(edu, edu)
                synth[display_name] = synth.get(display_name, 0) + 1
            else:
                synth['Uppgift saknas'] = (
                    synth.get('Uppgift saknas', 0) + 1)

        comparison = []
        all_categories = sorted(
            set(list(actual.keys()) + list(synth.keys())))
        for category in all_categories:
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        return {
            'name': 'Education Level Distribution',
            'comparison': comparison,
        }

    def _compare_income_source_distribution(self) -> dict:
        source_data = self._marginals.get('income_source')
        if source_data is None or (
                hasattr(source_data, 'empty') and source_data.empty):
            return {}

        source_display = {
            'work': 'Ersättning för arbete',
            'unemployment': 'Ersättning vid arbetslöshet',
            'studies': 'Ersättning för studier',
            'pension': 'Pension',
            'disability': (
                'Ersättning vid långvarigt nedsatt arbetsförmåga'),
            'sickness': 'Ersättning vid sjukdom',
            'parental_leave': 'Ersättning vid föräldraledighet...',
            'financial_support': 'Ekonomiskt stöd',
            'no_income': 'Saknar ersättningar',
        }

        _cfg = self.config
        source_col = None
        for col in source_data.columns:
            if ('inkomstkälla' in col.lower()
                    or 'huvudsaklig' in col.lower()):
                source_col = col
                break
        if source_col is None:
            return {}

        count_col = ('Antal' if 'Antal' in source_data.columns
                     else source_data.columns[-1])

        actual: dict = {}
        for _, row in source_data.iterrows():
            src_sv = row[source_col]
            src_en = _cfg.translate_income_source(src_sv)
            display = source_display.get(src_en, src_sv)
            count = int(row[count_col]) if pd.notna(row[count_col]) else 0
            actual[display] = actual.get(display, 0) + count

        synth: dict = {}
        for ind in self.individuals:
            if ind.age < 20:
                continue
            src = getattr(ind, 'income_source', None)
            if src:
                display = source_display.get(src, src)
                synth[display] = synth.get(display, 0) + 1

        comparison = []
        all_categories = sorted(
            set(list(actual.keys()) + list(synth.keys())))
        for category in all_categories:
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        return {
            'name': 'Income Source Distribution',
            'comparison': comparison,
        }

    def _compare_median_income(self) -> dict:
        edu_data = self._marginals.get('education_level')
        if edu_data is None or (
                hasattr(edu_data, 'empty') and edu_data.empty):
            return {}
        if 'Tabellvärde' not in edu_data.columns:
            return {}

        median_rows = edu_data[edu_data['Tabellvärde'] == 'Medianinkomst']
        if median_rows.empty:
            return {}

        _cfg = self.config

        age_groups = [
            (18, 24, '18-24 år'), (25, 34, '25-34 år'),
            (35, 44, '35-44 år'), (45, 54, '45-54 år'),
            (55, 64, '55-64 år'), (65, 74, '65-74 år'),
            (75, 120, '75- år'),
        ]

        census_medians: dict = {}
        for _, row in median_rows.iterrows():
            sex_en = _cfg.translate_sex(row.get('Kön', ''))
            edu_en = _cfg.translate_education(row.get(
                'Utbildningsnivå', ''))
            age_label = row.get('Ålder', '')
            val = row.get('Antal', 0)
            if (sex_en and edu_en and age_label
                    and pd.notna(val) and val > 0):
                census_medians[
                    (age_label, sex_en, edu_en)] = float(val)

        if not census_medians:
            return {}

        synth_buckets: dict = {}
        for ind in self.individuals:
            if ind.age < 18 or ind.income is None:
                continue
            age_label = None
            for ag_min, ag_max, label in age_groups:
                if ag_min <= ind.age <= ag_max:
                    age_label = label
                    break
            if age_label is None:
                continue
            edu = getattr(ind, 'education', None) or 'unknown'
            key = (age_label, ind.sex, edu)
            synth_buckets.setdefault(key, []).append(ind.income)

        edu_display = {
            'pre_secondary': 'Förgymnasial',
            'secondary': 'Gymnasial',
            'post_secondary': 'Eftergymnasial',
            'unknown': 'Uppgift saknas',
        }
        sex_display = {'male': 'M', 'female': 'K'}

        comparison = []
        for key in sorted(census_medians.keys()):
            age_label, sex_en, edu_en = key
            census_val = int(round(census_medians[key]))
            synth_incomes = synth_buckets.get(key, [])
            synth_val = (int(round(statistics.median(synth_incomes)))
                         if synth_incomes else 0)
            diff = synth_val - census_val
            error_pct = (
                (diff / census_val * 100) if census_val > 0 else 0)

            cat = (f"{edu_display.get(edu_en, edu_en)} {age_label} "
                   f"{sex_display.get(sex_en, sex_en)}")
            comparison.append({
                'category': cat,
                'actual': census_val,
                'synth': synth_val,
                'diff': diff,
                'error_pct': round(error_pct, 1),
            })

        return {
            'name': 'Median Income (SEK, informational)',
            'comparison': comparison,
        }

    def _compare_hh_type_children(self) -> dict:
        hh_tc_data = self._marginals.get('hh_type_children')
        if hh_tc_data is None or (
                hasattr(hh_tc_data, 'empty') and hh_tc_data.empty):
            return {}

        type_col = child_col = None
        for col in hh_tc_data.columns:
            cl = col.lower()
            if 'hushållstyp' in cl:
                type_col = col
            elif 'barn' in cl:
                child_col = col
        if type_col is None or child_col is None:
            return {}

        count_col = ('Antal' if 'Antal' in hh_tc_data.columns
                     else hh_tc_data.columns[-1])

        actual: dict = {}
        for _, row in hh_tc_data.iterrows():
            ht = str(row[type_col]).strip()
            nc = str(row[child_col]).strip()
            val = int(row[count_col]) if pd.notna(row[count_col]) else 0
            key = f"{ht} | {nc}"
            actual[key] = actual.get(key, 0) + val

        child_bins = [
            '0 barn', '1 barn', '2 barn', '3 barn',
            '4 barn eller fler',
        ]

        synth: dict = {}
        for hh in self.households:
            roles = [m.hh_role for m in hh.members]
            if any(r == 'cohabiting' for r in roles):
                hh_type_sv = 'Sammanboende'
            elif any(r == 'other' for r in roles):
                hh_type_sv = 'Övriga hushåll'
            else:
                hh_type_sv = 'Ensamstående'

            n_children = sum(1 for m in hh.members if m.age <= 17)
            nc_label = (
                '4 barn eller fler' if n_children >= 4
                else child_bins[n_children])

            key = f"{hh_type_sv} | {nc_label}"
            synth[key] = synth.get(key, 0) + 1

        comparison = []
        all_keys = sorted(
            set(list(actual.keys()) + list(synth.keys())))
        for key in all_keys:
            act_val = actual.get(key, 0)
            syn_val = synth.get(key, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': key,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        return {
            'name': 'HH Type × Children 0-17 (informational)',
            'comparison': comparison,
        }

    def _compare_joint_role_age_sex(self) -> dict:
        pos_data = self._marginals.get('household_position')
        if pos_data is None or (
                hasattr(pos_data, 'empty') and pos_data.empty):
            return {}

        age_col = sex_col = pos_col = None
        for col in pos_data.columns:
            cl = col.lower()
            if 'ålder' in cl:
                age_col = col
            elif 'kön' in cl:
                sex_col = col
            elif 'ställning' in cl or 'position' in cl:
                pos_col = col
        if not all([age_col, sex_col, pos_col]):
            return {}

        count_col = ('Antal' if 'Antal' in pos_data.columns
                     else pos_data.columns[-1])

        short_pos = self.config.translate_position_detailed

        actual: dict = {}
        for _, row in pos_data.iterrows():
            pos = short_pos(row[pos_col])
            val = int(row[count_col]) if pd.notna(row[count_col]) else 0
            actual[pos] = actual.get(pos, 0) + val

        if not actual:
            return {}

        hh_by_id = {h.household_id: h for h in self.households}

        act_gift = actual.get('Gift/reg.partner', 0)
        act_sambo = actual.get('Sambo', 0)
        total_cohab = act_gift + act_sambo
        gift_frac = (act_gift / total_cohab
                     if total_cohab > 0 else 0.5)

        cohab_ids = [
            ind.agent_id for ind in self.individuals
            if ind.hh_role == 'cohabiting']
        n_gift = round(len(cohab_ids) * gift_frac)
        gift_set = set(sorted(cohab_ids)[:n_gift])

        synth: dict = {}
        for ind in self.individuals:
            role = ind.hh_role
            if role == 'child':
                pos = 'Barn'
            elif role == 'cohabiting':
                pos = ('Gift/reg.partner'
                       if ind.agent_id in gift_set else 'Sambo')
            elif role == 'single_parent':
                pos = 'Ensam förälder'
            elif role == 'single':
                hh = hh_by_id.get(ind.household_id)
                if hh and len(hh.members) == 1:
                    pos = 'Ensamboende'
                elif hh and any(
                        m.hh_role == 'child' for m in hh.members):
                    pos = 'Ensam förälder'
                else:
                    pos = 'Ensamboende'
            elif role == 'other':
                pos = 'Övriga'
            else:
                pos = 'Uppgift saknas'
            synth[pos] = synth.get(pos, 0) + 1

        comparison = []
        all_positions = sorted(
            set(list(actual.keys()) + list(synth.keys())))
        for pos in all_positions:
            if pos == 'Uppgift saknas':
                continue
            act_val = actual.get(pos, 0)
            syn_val = synth.get(pos, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': pos,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        return {
            'name': 'Detailed HH Position (7-cat, informational)',
            'comparison': comparison,
        }

    def _compare_income_distribution(self) -> dict:
        income_data = self._marginals.get('income')
        if income_data is None or income_data.empty:
            return {}

        income_col = None
        for col in ['Inkomststandard', 'Inkomst', 'income']:
            if col in income_data.columns:
                income_col = col
                break
        if income_col is None:
            return {}

        count_col = None
        for col in income_data.columns:
            if (income_data[col].dtype in ['int64', 'float64']
                    and col not in ['År']):
                count_col = col
                break
        if count_col is None:
            count_col = income_data.columns[-1]

        actual = income_data.groupby(
            income_col)[count_col].sum().to_dict()

        low_income_key = not_low_key = other_key = None
        for cat in actual.keys():
            cat_lower = str(cat).lower()
            if 'inte har låg' in cat_lower or 'not low' in cat_lower:
                not_low_key = cat
            elif 'har låg' in cat_lower or 'low' in cat_lower:
                low_income_key = cat
            elif 'ej i' in cat_lower or 'not in' in cat_lower:
                other_key = cat

        synth: dict = {}
        for ind in self.individuals:
            if (hasattr(ind, 'income_standard')
                    and ind.income_standard):
                if ind.income_standard == 'low':
                    key = low_income_key or 'Low income'
                else:
                    key = not_low_key or 'Not low income'
                synth[key] = synth.get(key, 0) + 1
            elif (hasattr(ind, 'income_decile')
                    and ind.income_decile):
                if ind.income_decile <= 2:
                    key = low_income_key or 'Low income'
                else:
                    key = not_low_key or 'Not low income'
                synth[key] = synth.get(key, 0) + 1
            else:
                key = not_low_key or 'Not low income'
                synth[key] = synth.get(key, 0) + 1

        comparison = []
        for category in set(list(actual.keys()) + list(synth.keys())):
            act_val = actual.get(category, 0)
            syn_val = synth.get(category, 0)
            diff = syn_val - act_val
            error_pct = (diff / act_val * 100) if act_val > 0 else 0
            comparison.append({
                'category': category,
                'actual': int(act_val),
                'synth': int(syn_val),
                'diff': int(diff),
                'error_pct': round(error_pct, 1),
            })

        return {
            'name': 'Income Standard Distribution',
            'comparison': comparison,
        }

    # ------------------------------------------------------------------
    # Report printer
    # ------------------------------------------------------------------

    def _print_comparison_report(
        self, comparisons: dict, use_logging: bool = False
    ) -> None:
        output = (print if not use_logging
                  else lambda msg: logger.info(msg))

        output("\n" + "=" * 70)
        output(
            f"MARGINAL COMPARISON REPORT: "
            f"{self.area_name} ({self.year})")
        output("=" * 70)

        for key, data in comparisons.items():
            if key == 'overall':
                continue
            if not data or 'comparison' not in data:
                continue

            output(f"\n{data['name']}")
            output("-" * 50)
            output(
                f"{'Category':<20} {'Actual':>10} {'Synth':>10} "
                f"{'Diff':>8} {'Error':>8}")
            output("-" * 50)

            for row in data['comparison']:
                cat = (row['category'][:18]
                       if len(row['category']) > 18
                       else row['category'])
                excluded = ' *' if row.get('exclude_from_mape') else ''
                output(
                    f"{cat:<20} {row['actual']:>10} "
                    f"{row['synth']:>10} {row['diff']:>+8} "
                    f"{row['error_pct']:>7.1f}%{excluded}")

            total_actual = sum(
                r['actual'] for r in data['comparison'])
            total_synth = sum(
                r['synth'] for r in data['comparison'])
            n_rows = len(data['comparison'])
            output("-" * 50)
            if key == 'median_income' and n_rows > 0:
                avg_a = total_actual // n_rows
                avg_s = total_synth // n_rows
                avg_d = avg_s - avg_a
                output(
                    f"{'AVERAGE':<20} {avg_a:>10} "
                    f"{avg_s:>10} {avg_d:>+8}")
            else:
                total_diff = total_synth - total_actual
                output(
                    f"{'TOTAL':<20} {total_actual:>10} "
                    f"{total_synth:>10} {total_diff:>+8}")

            if key in ('median_income', 'hh_type_children',
                       'joint_role_age_sex'):
                output(
                    "  (informational — excluded from MAPE grade)")
            elif data.get('fit'):
                f = data['fit']
                output(
                    f"  SAE={f['sae']:.4f}  "
                    f"X²={f['chi2']:.2f}(p={f['chi2_p']:.4f})  "
                    f"Z²={f['z2']:.2f}(p={f['z2_p']:.4f})")

        if 'overall' in comparisons:
            ov = comparisons['overall']
            output("\n" + "=" * 70)
            output("OVERALL FIT STATISTICS")
            output("=" * 70)
            output(
                f"  Total Population (Actual):  "
                f"{ov['total_actual']:,}")
            output(
                f"  Total Population (Synth):   "
                f"{ov['total_synth']:,}")
            output(
                f"  Categories Compared:        "
                f"{ov.get('n_categories', 'N/A')}")
            output(
                f"  Root Mean Square Error:     {ov['rmse']:.2f}")
            output(
                f"  Mean Absolute Error:        {ov['mae']:.2f}")
            output(
                f"  Max Category Error:         {ov['max_error']}")
            output(
                f"  Pearson Correlation:        "
                f"{ov['correlation']:.4f}")
            output("")
            output("  Percentage Error Metrics:")
            output(
                f"    MAPE (unweighted):        "
                f"{ov.get('mape', 0):.1f}%  "
                "← treats all categories equally")
            output(
                f"    Weighted MAPE:            "
                f"{ov.get('wmape', 0):.1f}%  "
                "← weighted by category size")
            output(
                f"    Max Category Error:       "
                f"{ov.get('max_pct_error', 0):.1f}%")
            output("")
            output(
                "  Voas & Williamson (2001) — "
                "per-dimension then aggregated:")
            output(
                f"    SAE median:               "
                f"{ov.get('sae_median', 0):.4f}  ← lower is better")
            output(
                f"    SAE mean:                 "
                f"{ov.get('sae_mean', 0):.4f}")
            output(
                f"    SAE max:                  "
                f"{ov.get('sae_max', 0):.4f}")
            output(
                f"    X² p-value (worst dim):   "
                f"{ov.get('chi2_p_min', 0):.4f}  ← higher is better")
            output(
                f"    Z² p-value (worst dim):   "
                f"{ov.get('z2_p_min', 0):.4f}")
            if ov.get('dim_metrics'):
                output("")
                output(
                    f"    {'Dimension':<20} {'SAE':>8} {'X²':>10} "
                    f"{'X² p':>8} {'Z²':>10} {'Z² p':>8}")
                output("    " + "-" * 64)
                for dim_key, dm in ov['dim_metrics'].items():
                    output(
                        f"    {dim_key:<20} {dm['sae']:>8.4f} "
                        f"{dm['chi2']:>10.2f} {dm['chi2_p']:>8.4f} "
                        f"{dm['z2']:>10.2f} {dm['z2_p']:>8.4f}")

        output("=" * 70 + "\n")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _parse_age_range(label):
    """Parse ``'16-18 år'`` → ``(16, 18)``."""
    label = str(label).strip()
    match = re.match(r'(\d+)-(\d+)\s*år', label)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    match = re.match(r'(\d+)[-+]\s*år', label)
    if match:
        return (int(match.group(1)), 150)
    match = re.match(r'(\d+)-\w\s*år', label)
    if match:
        return (int(match.group(1)), 150)
    return None
