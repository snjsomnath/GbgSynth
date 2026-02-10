#!/usr/bin/env python3
"""
Synthesize populations for ALL neighbourhoods in Gothenburg.

This script:
1. Discovers all primary areas (neighbourhoods) in Gothenburg
2. Generates a synthetic population for each neighbourhood
3. Produces a comprehensive error report per neighbourhood
4. Saves individual population CSVs and a consolidated error report

Usage:
    python synthesize_all_neighbourhoods.py [--year YEAR] [--output-dir DIR]

Output:
    - output/populations/{code}_{name}_individuals.csv
    - output/populations/{code}_{name}_households.csv
    - output/reports/{code}_{name}_error_report.txt
    - output/summary_report.csv (consolidated all-area comparison)
    - output/synthesis_log.txt (detailed execution log)
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gbgsynth import GbgSynth


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "synthesis_log.txt")
    
    # Create logger
    logger = logging.getLogger('synth_all')
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed logging
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(fh_formatter)
    
    # Console handler - summary logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def generate_error_report(
    area,
    comparisons: Dict[str, Any],
    execution_time: float,
    error: Optional[str] = None
) -> str:
    """
    Generate a comprehensive text error report for a single neighbourhood.
    
    Args:
        area: GbgArea object (may be None if generation failed)
        comparisons: Dictionary from compare_to_marginals()
        execution_time: Time taken to generate in seconds
        error: Error message if generation failed
        
    Returns:
        Formatted error report as string
    """
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines.append("=" * 80)
    lines.append(f"SYNTHETIC POPULATION ERROR REPORT")
    lines.append("=" * 80)
    
    if area:
        lines.append(f"Neighbourhood: {area.area_name}")
        lines.append(f"Area Code:     {area.area_code}")
        lines.append(f"Year:          {area.year}")
    else:
        lines.append(f"Neighbourhood: GENERATION FAILED")
    
    lines.append(f"Generated:     {timestamp}")
    lines.append(f"Execution Time: {execution_time:.2f} seconds")
    lines.append("")
    
    # If there was an error, report it prominently
    if error:
        lines.append("!" * 80)
        lines.append("GENERATION ERROR")
        lines.append("!" * 80)
        lines.append(error)
        lines.append("!" * 80)
        lines.append("")
        return "\n".join(lines)
    
    # Summary Statistics
    lines.append("-" * 80)
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 80)
    
    stats = area.get_summary_statistics()
    lines.append(f"  Total Population:           {stats['total_population']:,}")
    lines.append(f"  Total Households:           {stats['total_households']:,}")
    lines.append(f"  Average Household Size:     {stats['avg_household_size']:.2f}")
    lines.append(f"  Number of Children:         {stats['num_children']:,}")
    lines.append(f"  Number of Adults:           {stats['num_adults']:,}")
    lines.append(f"  Couple Households:          {stats['num_couples']:,}")
    lines.append(f"  Single-Parent Households:   {stats['num_single_parent']:,}")
    lines.append(f"  Single-Person Households:   {stats['num_single_person']:,}")
    lines.append(f"  Average Income:             {stats['avg_income']:,.0f} SEK")
    lines.append(f"  Total Cars:                 {stats['total_cars']:,}")
    lines.append(f"  Småhus (Small houses):      {stats['hustyp_smahus']:,}")
    lines.append(f"  Flerbostadshus (Apartments):{stats['hustyp_flerbostadshus']:,}")
    lines.append(f"  Specialbostad (Special):    {stats['hustyp_specialbostad']:,}")
    lines.append("")
    
    # Detailed Marginal Comparisons
    for key, data in comparisons.items():
        if key == 'overall':
            continue
        if not data or 'comparison' not in data:
            continue
        
        lines.append("-" * 80)
        lines.append(f"MARGINAL COMPARISON: {data['name'].upper()}")
        lines.append("-" * 80)
        
        # Header
        lines.append(f"{'Category':<35} {'Census':>10} {'Synth':>10} {'Diff':>10} {'Error%':>10}")
        lines.append("-" * 75)
        
        # Sort by absolute error percentage (worst first)
        sorted_rows = sorted(
            data['comparison'],
            key=lambda x: abs(x['error_pct']),
            reverse=True
        )
        
        total_actual = 0
        total_synth = 0
        max_abs_error_pct = 0
        
        for row in sorted_rows:
            cat = row['category'][:33] if len(str(row['category'])) > 33 else row['category']
            
            # Mark high errors
            error_marker = ""
            if abs(row['error_pct']) > 20:
                error_marker = " ⚠️ HIGH"
            elif abs(row['error_pct']) > 10:
                error_marker = " ⚡"
            
            lines.append(
                f"{cat:<35} {row['actual']:>10,} {row['synth']:>10,} "
                f"{row['diff']:>+10,} {row['error_pct']:>9.1f}%{error_marker}"
            )
            
            total_actual += row['actual']
            total_synth += row['synth']
            max_abs_error_pct = max(max_abs_error_pct, abs(row['error_pct']))
        
        # Subtotals / averages
        lines.append("-" * 75)
        is_sek_comparison = key == 'median_income'
        is_informational = key in ('median_income', 'hh_type_children', 'joint_role_age_sex')
        n_rows = len(sorted_rows)
        if is_sek_comparison and n_rows > 0:
            # For SEK comparisons, show average rather than sum
            avg_actual = total_actual // n_rows
            avg_synth = total_synth // n_rows
            avg_diff = avg_synth - avg_actual
            avg_error_pct = (avg_diff / avg_actual * 100) if avg_actual > 0 else 0
            lines.append(
                f"{'AVERAGE':<35} {avg_actual:>10,} {avg_synth:>10,} "
                f"{avg_diff:>+10,} {avg_error_pct:>9.1f}%"
            )
        else:
            total_diff = total_synth - total_actual
            total_error_pct = (total_diff / total_actual * 100) if total_actual > 0 else 0
            lines.append(
                f"{'TOTAL':<35} {total_actual:>10,} {total_synth:>10,} "
                f"{total_diff:>+10,} {total_error_pct:>9.1f}%"
            )
        
        if is_informational:
            lines.append("  (informational — excluded from MAPE grade)")
        
        # Quality Assessment for this dimension
        if max_abs_error_pct > 25:
            quality = "POOR - Significant discrepancies detected"
        elif max_abs_error_pct > 15:
            quality = "FAIR - Some notable discrepancies"
        elif max_abs_error_pct > 5:
            quality = "GOOD - Minor discrepancies"
        else:
            quality = "EXCELLENT - Very close match"
        
        lines.append(f"Quality Assessment: {quality}")
        lines.append("")
    
    # Overall Fit Statistics
    if 'overall' in comparisons:
        lines.append("=" * 80)
        lines.append("OVERALL FIT STATISTICS")
        lines.append("=" * 80)
        
        ov = comparisons['overall']
        lines.append(f"  Total Census Population:        {ov['total_actual']:>15,}")
        lines.append(f"  Total Synthetic Population:     {ov['total_synth']:>15,}")
        lines.append(f"  Population Difference:          {ov['total_synth'] - ov['total_actual']:>+15,}")
        lines.append("")
        lines.append(f"  Root Mean Square Error (RMSE):  {ov['rmse']:>15.2f}")
        lines.append(f"  Mean Absolute Error (MAE):      {ov['mae']:>15.2f}")
        lines.append(f"  Maximum Category Error:         {ov['max_error']:>15,}")
        lines.append(f"  Pearson Correlation:            {ov['correlation']:>15.4f}")
        lines.append("")
        
        # Overall quality rating
        rmse = ov['rmse']
        correlation = ov['correlation']
        
        if correlation > 0.99 and rmse < 50:
            overall_quality = "⭐⭐⭐⭐⭐ EXCELLENT"
        elif correlation > 0.98 and rmse < 100:
            overall_quality = "⭐⭐⭐⭐ VERY GOOD"
        elif correlation > 0.95 and rmse < 200:
            overall_quality = "⭐⭐⭐ GOOD"
        elif correlation > 0.90:
            overall_quality = "⭐⭐ FAIR"
        else:
            overall_quality = "⭐ NEEDS IMPROVEMENT"
        
        lines.append(f"  Overall Quality Rating:         {overall_quality}")
    
    # IPF Statistics (if available)
    if hasattr(area, 'ipf_stats') and area.ipf_stats:
        lines.append("")
        lines.append("-" * 80)
        lines.append("IPF CONVERGENCE STATISTICS")
        lines.append("-" * 80)
        for key, value in area.ipf_stats.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}")
            else:
                lines.append(f"  {key}: {value}")
    
    # Recommendations
    lines.append("")
    lines.append("=" * 80)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 80)
    
    recommendations = []
    
    # Check for specific issues
    for key, data in comparisons.items():
        if key == 'overall' or not data or 'comparison' not in data:
            continue
        
        for row in data['comparison']:
            if abs(row['error_pct']) > 20:
                recommendations.append(
                    f"- Consider reviewing '{row['category']}' in {data['name']}: "
                    f"{abs(row['error_pct']):.1f}% error"
                )
    
    if 'overall' in comparisons:
        if comparisons['overall']['correlation'] < 0.95:
            recommendations.append(
                "- Low correlation suggests systematic mismatch. "
                "Consider using constrained IPF (use_constrained_ipf=True)"
            )
        if comparisons['overall']['rmse'] > 100:
            recommendations.append(
                "- High RMSE indicates large absolute errors in some categories"
            )
    
    if not recommendations:
        recommendations.append("- Synthesis quality is acceptable. No specific recommendations.")
    
    for rec in recommendations[:10]:  # Limit to top 10
        lines.append(rec)
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def create_summary_dataframe(all_results: List[Dict]) -> pd.DataFrame:
    """
    Create a consolidated summary DataFrame from all area results.
    
    Args:
        all_results: List of dictionaries with results per area
        
    Returns:
        DataFrame with one row per area and comparison metrics
    """
    rows = []
    
    for result in all_results:
        row = {
            'area_code': result.get('area_code'),
            'area_name': result.get('area_name'),
            'status': result.get('status'),
            'execution_time_sec': result.get('execution_time', 0),
            'total_population': None,
            'total_households': None,
            'avg_hh_size': None,
            'rmse': None,
            'mae': None,
            'correlation': None,
            'max_error': None,
            'error_message': result.get('error')
        }
        
        if result.get('stats'):
            stats = result['stats']
            row['total_population'] = stats.get('total_population')
            row['total_households'] = stats.get('total_households')
            row['avg_hh_size'] = stats.get('avg_household_size')
            row['num_children'] = stats.get('num_children')
            row['num_adults'] = stats.get('num_adults')
            row['total_cars'] = stats.get('total_cars')
        
        if result.get('comparisons') and 'overall' in result['comparisons']:
            ov = result['comparisons']['overall']
            row['rmse'] = ov.get('rmse')
            row['mae'] = ov.get('mae')
            row['correlation'] = ov.get('correlation')
            row['max_error'] = ov.get('max_error')
            row['census_population'] = ov.get('total_actual')
            row['synth_population'] = ov.get('total_synth')
            
            # Add quality grade based on correlation
            corr = ov.get('correlation', 0)
            if corr >= 0.99:
                row['quality_grade'] = 'A'
            elif corr >= 0.98:
                row['quality_grade'] = 'B'
            elif corr >= 0.97:
                row['quality_grade'] = 'C'
            elif corr >= 0.95:
                row['quality_grade'] = 'D'
            else:
                row['quality_grade'] = 'F'
        
        # Per-dimension error summaries
        if result.get('comparisons'):
            for dim_key, data in result['comparisons'].items():
                if dim_key == 'overall' or not data or 'comparison' not in data:
                    continue
                
                errors = [abs(r['error_pct']) for r in data['comparison']]
                if errors:
                    dim_name = dim_key.replace(' ', '_').lower()
                    row[f'{dim_name}_max_error_pct'] = max(errors)
                    row[f'{dim_name}_mean_error_pct'] = np.mean(errors)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_detailed_comparison_csv(all_results: List[Dict], output_path: str):
    """
    Create a detailed CSV with all comparison rows from all areas.
    
    Args:
        all_results: List of dictionaries with results per area
        output_path: Path to save the CSV
    """
    rows = []
    
    for result in all_results:
        if not result.get('comparisons'):
            continue
        
        for dim_key, data in result['comparisons'].items():
            if dim_key == 'overall' or not data or 'comparison' not in data:
                continue
            
            for comp in data['comparison']:
                rows.append({
                    'area_code': result.get('area_code'),
                    'area_name': result.get('area_name'),
                    'dimension': data['name'],
                    'category': comp['category'],
                    'census_count': comp['actual'],
                    'synth_count': comp['synth'],
                    'difference': comp['diff'],
                    'error_pct': comp['error_pct']
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def main():
    """Main entry point for the synthesis script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic populations for all Gothenburg neighbourhoods"
    )
    parser.add_argument(
        '--year', type=int, default=2023,
        help='Census year to use (default: 2023)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./output',
        help='Output directory (default: ./output)'
    )
    parser.add_argument(
        '--method', type=str, default='topdown',
        choices=['topdown', 'constrained_ipf', 'ipf', 'greedy'],
        help='Synthesis method (default: topdown)'
    )
    parser.add_argument(
        '--areas', type=str, nargs='+',
        help='Specific area codes to process (default: all)'
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip areas that already have output files'
    )
    
    args = parser.parse_args()
    
    # Setup directories
    output_dir = args.output_dir
    pop_dir = os.path.join(output_dir, 'populations')
    report_dir = os.path.join(output_dir, 'reports')
    
    os.makedirs(pop_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 70)
    logger.info("GOTHENBURG SYNTHETIC POPULATION GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Year:             {args.year}")
    logger.info(f"Method:           {args.method}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("")
    
    # Initialize GbgSynth
    logger.info("Initializing GbgSynth...")
    city = GbgSynth(year=args.year, log_level='WARNING')
    
    # Discover all areas
    logger.info("Discovering all neighbourhoods...")
    all_areas = city.get_all_areas()
    
    # Filter areas if specific ones requested
    if args.areas:
        all_areas = {k: v for k, v in all_areas.items() if k in args.areas}
    
    total_areas = len(all_areas)
    logger.info(f"Found {total_areas} neighbourhoods to process")
    logger.info("")
    
    # Set synthesis method flags
    use_topdown = args.method == 'topdown'
    use_constrained_ipf = args.method == 'constrained_ipf'
    use_ipf = args.method == 'ipf'
    
    # Areas to skip (zero population or known data issues)
    SKIP_AREAS = {
        '199',  # Ospecificerat (unspecified) - zero population, no geographic area
    }
    
    # Areas with known data quality warnings (will still be processed)
    WARN_AREAS = {
        '108': 'Annedal has unusual household/population ratio - results may be unreliable',
        '516': 'Högsbo is primarily industrial - very small residential population',
        '707': 'Arendal is primarily industrial - very small residential population',
    }
    
    # Track all results
    all_results = []
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each area
    for idx, (code, name) in enumerate(all_areas.items(), 1):
        safe_name = name.replace(' ', '_').replace('/', '-')
        
        logger.info(f"[{idx}/{total_areas}] Processing {name} ({code})...")
        
        # Check if area is in skip list
        if code in SKIP_AREAS:
            logger.info(f"  ⏭️  Skipping {name} - zero population area")
            print(f"  ⏭️  Skipping {name} - zero population/data quality issues")
            skipped += 1
            continue
        
        # Check if area has a warning
        if code in WARN_AREAS:
            logger.warning(f"  ⚠️  {WARN_AREAS[code]}")
            print(f"  ⚠️  Warning: {WARN_AREAS[code]}")
        
        # Check if we should skip
        if args.skip_existing:
            individuals_file = os.path.join(pop_dir, f"{code}_{safe_name}_individuals.csv")
            if os.path.exists(individuals_file):
                logger.info(f"  ⏭️ Skipping (already exists)")
                continue
        
        result = {
            'area_code': code,
            'area_name': name,
            'status': 'pending',
            'execution_time': 0,
            'error': None,
            'stats': None,
            'comparisons': None
        }
        
        start_time = datetime.now()
        
        try:
            # Get area object
            area = city.get_area(code)
            
            # Generate synthetic population
            area.generate()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time'] = execution_time
            
            # Get statistics
            stats = area.get_summary_statistics()
            result['stats'] = stats
            
            # Get comparisons
            comparisons = area.compare_to_marginals(print_report=False)
            result['comparisons'] = comparisons
            
            # Save population files
            individuals_file = os.path.join(pop_dir, f"{code}_{safe_name}_individuals.csv")
            households_file = os.path.join(pop_dir, f"{code}_{safe_name}_households.csv")
            
            area.save_to_csv(individuals_file)
            area.save_households_to_csv(households_file)
            
            # Generate and save error report
            error_report = generate_error_report(area, comparisons, execution_time)
            report_file = os.path.join(report_dir, f"{code}_{safe_name}_error_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(error_report)
            
            result['status'] = 'success'
            successful += 1
            
            # Log summary with new MAPE metric
            ov = comparisons.get('overall', {})
            rmse = ov.get('rmse', 'N/A')
            mape = ov.get('mape', 'N/A')  # Mean Absolute Percentage Error - more honest than correlation
            corr = ov.get('correlation', 'N/A')
            if isinstance(rmse, float):
                rmse = f"{rmse:.1f}"
            if isinstance(mape, float):
                mape = f"{mape:.1f}%"
            if isinstance(corr, float):
                corr = f"{corr:.4f}"
            
            logger.info(
                f"  ✅ {stats['total_population']:,} individuals, "
                f"{stats['total_households']:,} households | "
                f"MAPE: {mape}, Corr: {corr} | "
                f"{execution_time:.1f}s"
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time'] = execution_time
            result['status'] = 'failed'
            result['error'] = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            
            failed += 1
            logger.error(f"  ❌ Failed: {e}")
            
            # Generate error-only report
            error_report = generate_error_report(None, {}, execution_time, result['error'])
            report_file = os.path.join(report_dir, f"{code}_{safe_name}_error_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(error_report)
        
        all_results.append(result)
    
    # Generate consolidated reports
    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERATING CONSOLIDATED REPORTS")
    logger.info("=" * 70)
    
    # Summary CSV
    summary_df = create_summary_dataframe(all_results)
    summary_file = os.path.join(output_dir, 'summary_report.csv')
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"  Saved summary report to {summary_file}")
    
    # Detailed comparison CSV
    detailed_file = os.path.join(output_dir, 'detailed_comparisons.csv')
    create_detailed_comparison_csv(all_results, detailed_file)
    logger.info(f"  Saved detailed comparisons to {detailed_file}")
    
    # JSON results for programmatic access
    json_results = []
    for r in all_results:
        json_result = {
            'area_code': r['area_code'],
            'area_name': r['area_name'],
            'status': r['status'],
            'execution_time': r['execution_time'],
            'error': r['error']
        }
        if r['stats']:
            json_result['stats'] = r['stats']
        if r['comparisons'] and 'overall' in r['comparisons']:
            json_result['overall_fit'] = r['comparisons']['overall']
        json_results.append(json_result)
    
    json_file = os.path.join(output_dir, 'results.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved JSON results to {json_file}")
    
    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SYNTHESIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total Neighbourhoods:    {total_areas}")
    logger.info(f"  Successful:              {successful}")
    logger.info(f"  Skipped:                 {skipped}")
    logger.info(f"  Failed:                  {failed}")
    
    if successful > 0:
        success_results = [r for r in all_results if r['status'] == 'success']
        
        total_pop = sum(r['stats']['total_population'] for r in success_results if r['stats'])
        total_hh = sum(r['stats']['total_households'] for r in success_results if r['stats'])
        
        # Collect metrics
        correlations = []
        mapes = []
        rmses = []
        for r in success_results:
            if r['comparisons'] and 'overall' in r['comparisons']:
                ov = r['comparisons']['overall']
                correlations.append(ov.get('correlation', 0))
                mapes.append(ov.get('mape', 0))
                rmses.append(ov.get('rmse', 0))
        
        avg_rmse = np.mean(rmses) if rmses else 0
        avg_corr = np.mean(correlations) if correlations else 0
        avg_mape = np.mean(mapes) if mapes else 0
        
        # Quality grade breakdown by MAPE (more honest grading)
        grade_a = sum(1 for m in mapes if m <= 5)      # ≤5% MAPE
        grade_b = sum(1 for m in mapes if 5 < m <= 10)  # 5-10% MAPE
        grade_c = sum(1 for m in mapes if 10 < m <= 15) # 10-15% MAPE
        grade_d = sum(1 for m in mapes if 15 < m <= 25) # 15-25% MAPE
        grade_f = sum(1 for m in mapes if m > 25)       # >25% MAPE
        
        logger.info(f"  Total Population:        {total_pop:,}")
        logger.info(f"  Total Households:        {total_hh:,}")
        logger.info(f"  Average RMSE:            {avg_rmse:.2f}")
        logger.info(f"  Average MAPE:            {avg_mape:.1f}%")
        logger.info(f"  Average Correlation:     {avg_corr:.4f}")
        logger.info("")
        logger.info("  Quality Grades (by MAPE - treats all categories equally):")
        logger.info(f"    A (≤5%):   {grade_a:3d} areas")
        logger.info(f"    B (≤10%):  {grade_b:3d} areas")
        logger.info(f"    C (≤15%):  {grade_c:3d} areas")
        logger.info(f"    D (≤25%):  {grade_d:3d} areas")
        logger.info(f"    F (>25%):  {grade_f:3d} areas")
    
    logger.info("")
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("=" * 70)
    
    # Return exit code based on failures
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())