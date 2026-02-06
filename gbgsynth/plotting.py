"""
Plotting module for visualizing synthetic populations.

This module provides convenient plotting functions for analyzing and comparing
synthetic populations against census marginals.

Requires matplotlib. Install with: pip install matplotlib
Optional: seaborn for enhanced styling (pip install seaborn)
"""

import logging
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gbgsynth.area import GbgArea
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None  # type: ignore


def _check_matplotlib():
    """Raise error if matplotlib is not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def set_style(style: str = "seaborn-v0_8-whitegrid") -> None:
    """
    Set the plotting style.
    
    Args:
        style: Style name ('seaborn-v0_8-whitegrid', 'ggplot', 'classic', 'dark_background')
    """
    _check_matplotlib()
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")
    elif style.startswith("seaborn"):
        # Use a built-in style if seaborn style not available
        plt.style.use("ggplot")
    else:
        plt.style.use(style)


def plot_age_distribution(
    area: "GbgArea",
    show_marginals: bool = True,
    ax: Optional["Axes"] = None,
    figsize: tuple = (10, 6),
    title: Optional[str] = None
) -> "Figure":
    """
    Plot age distribution of the synthetic population.
    
    Args:
        area: GbgArea object with generated population
        show_marginals: If True, overlay the census marginal targets
        ax: Optional matplotlib axes to plot on
        figsize: Figure size (width, height) in inches
        title: Custom title (defaults to area name)
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if not area._is_generated:
        raise RuntimeError("Must call generate() before plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get synthesized ages
    ages = [ind.age for ind in area.individuals]
    
    # Create age groups matching census categories
    age_bins = [0, 6, 16, 19, 25, 35, 45, 55, 65, 75, 85, 100]
    age_labels = ['0-5', '6-15', '16-18', '19-24', '25-34', '35-44', 
                  '45-54', '55-64', '65-74', '75-84', '85+']
    
    # Plot synthesized distribution
    ax.hist(ages, bins=age_bins, alpha=0.7, label='Synthesized', 
            color='steelblue', edgecolor='white')
    
    # Overlay marginals if available and requested
    if show_marginals and area._marginals.get('population') is not None:
        comparison = area._compare_age_distribution()
        if comparison and 'comparison' in comparison:
            x_positions = []
            y_values = []
            for i, row in enumerate(comparison['comparison']):
                x_positions.append(i)
                y_values.append(row['actual'])
            
            # Plot as line overlay
            bin_centers = [(age_bins[i] + age_bins[i+1])/2 for i in range(len(age_bins)-1)]
            if len(y_values) == len(bin_centers):
                ax.plot(bin_centers, y_values, 'ro-', linewidth=2, 
                       markersize=8, label='Census Target')
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.set_title(title or f'Age Distribution: {area.area_name}')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_household_size(
    area: "GbgArea",
    show_marginals: bool = True,
    ax: Optional["Axes"] = None,
    figsize: tuple = (8, 6),
    title: Optional[str] = None
) -> "Figure":
    """
    Plot household size distribution.
    
    Args:
        area: GbgArea object with generated population
        show_marginals: If True, show census marginal targets side by side
        ax: Optional matplotlib axes to plot on
        figsize: Figure size (width, height) in inches
        title: Custom title
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if not area._is_generated:
        raise RuntimeError("Must call generate() before plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Count household sizes
    size_counts = {}
    for hh in area.households:
        size = min(hh.size, 6)  # Cap at 6+
        label = f'{size}' if size < 6 else '6+'
        size_counts[label] = size_counts.get(label, 0) + 1
    
    labels = ['1', '2', '3', '4', '5', '6+']
    synth_values = [size_counts.get(l, 0) for l in labels]
    
    x = range(len(labels))
    width = 0.35
    
    if show_marginals and area._marginals.get('household') is not None:
        comparison = area._compare_household_size_distribution()
        if comparison and 'comparison' in comparison:
            actual_values = []
            for label in labels:
                found = False
                for row in comparison['comparison']:
                    cat = row['category']
                    if label == '6+' and '6' in cat:
                        actual_values.append(row['actual'])
                        found = True
                        break
                    elif label in cat and label != '6+':
                        actual_values.append(row['actual'])
                        found = True
                        break
                if not found:
                    actual_values.append(0)
            
            ax.bar([i - width/2 for i in x], actual_values, width, 
                   label='Census Target', color='coral', alpha=0.8)
            ax.bar([i + width/2 for i in x], synth_values, width,
                   label='Synthesized', color='steelblue', alpha=0.8)
        else:
            ax.bar(x, synth_values, color='steelblue', alpha=0.8)
    else:
        ax.bar(x, synth_values, color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Household Size')
    ax.set_ylabel('Number of Households')
    ax.set_title(title or f'Household Size Distribution: {area.area_name}')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    
    # Only show legend if there are labeled artists
    handles, labels_legend = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_marginal_comparison(
    area: "GbgArea",
    dimensions: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
    title: Optional[str] = None
) -> "Figure":
    """
    Create a multi-panel comparison of synthesized vs census marginals.
    
    Args:
        area: GbgArea object with generated population
        dimensions: List of dimensions to plot. Options: 'sex', 'age', 
                   'household_size', 'housing_type', 'income'
                   If None, plots all available.
        figsize: Figure size (width, height) in inches
        title: Overall figure title
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if not area._is_generated:
        raise RuntimeError("Must call generate() before plotting")
    
    comparison = area.compare_to_marginals(print_report=False)
    
    # Determine which dimensions to plot
    available = [k for k in comparison.keys() if k != 'overall' and comparison[k]]
    if dimensions:
        to_plot = [d for d in dimensions if d in available]
    else:
        to_plot = available[:6]  # Max 6 panels
    
    n_plots = len(to_plot)
    if n_plots == 0:
        logger.warning("No comparison data available for plotting")
        return None
    
    # Create subplot grid
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, dim in enumerate(to_plot):
        ax = axes[i]
        data = comparison[dim]
        
        if not data or 'comparison' not in data:
            continue
        
        categories = [r['category'][:15] for r in data['comparison']]
        actual = [r['actual'] for r in data['comparison']]
        synth = [r['synth'] for r in data['comparison']]
        
        x = range(len(categories))
        width = 0.35
        
        ax.bar([p - width/2 for p in x], actual, width, label='Census', 
               color='coral', alpha=0.8)
        ax.bar([p + width/2 for p in x], synth, width, label='Synth',
               color='steelblue', alpha=0.8)
        
        ax.set_title(data.get('name', dim))
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title or f'Marginal Comparison: {area.area_name}', fontsize=14)
    plt.tight_layout()
    return fig


def plot_car_ownership(
    area: "GbgArea",
    by: str = "household_size",
    ax: Optional["Axes"] = None,
    figsize: tuple = (8, 6),
    title: Optional[str] = None
) -> "Figure":
    """
    Plot car ownership patterns.
    
    Args:
        area: GbgArea object with generated population
        by: Grouping variable ('household_size', 'housing_type')
        ax: Optional matplotlib axes to plot on
        figsize: Figure size (width, height) in inches
        title: Custom title
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if not area._is_generated:
        raise RuntimeError("Must call generate() before plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    if by == "household_size":
        # Group by household size
        groups = {}
        for hh in area.households:
            size = min(hh.size, 6)
            label = f'{size}' if size < 6 else '6+'
            if label not in groups:
                groups[label] = {'count': 0, 'cars': 0}
            groups[label]['count'] += 1
            groups[label]['cars'] += hh.cars
        
        labels = ['1', '2', '3', '4', '5', '6+']
        avg_cars = [groups.get(l, {'count': 1, 'cars': 0})['cars'] / 
                    max(groups.get(l, {'count': 1})['count'], 1) for l in labels]
        
        bars = ax.bar(labels, avg_cars, color='steelblue', alpha=0.8)
        ax.set_xlabel('Household Size')
        ax.set_ylabel('Average Cars per Household')
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_cars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    elif by == "housing_type":
        # Group by housing type
        groups = {}
        for hh in area.households:
            ht = hh.assigned_hustyp or 'Unknown'
            if ht not in groups:
                groups[ht] = {'count': 0, 'cars': 0}
            groups[ht]['count'] += 1
            groups[ht]['cars'] += hh.cars
        
        labels = list(groups.keys())
        avg_cars = [groups[l]['cars'] / max(groups[l]['count'], 1) for l in labels]
        
        bars = ax.bar(labels, avg_cars, color='steelblue', alpha=0.8)
        ax.set_xlabel('Housing Type')
        ax.set_ylabel('Average Cars per Household')
        plt.xticks(rotation=45, ha='right')
        
        for bar, val in zip(bars, avg_cars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title(title or f'Car Ownership: {area.area_name}')
    
    plt.tight_layout()
    return fig


def plot_population_pyramid(
    area: "GbgArea",
    ax: Optional["Axes"] = None,
    figsize: tuple = (10, 8),
    title: Optional[str] = None
) -> "Figure":
    """
    Plot a population pyramid (age-sex distribution).
    
    Args:
        area: GbgArea object with generated population
        ax: Optional matplotlib axes to plot on
        figsize: Figure size (width, height) in inches
        title: Custom title
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if not area._is_generated:
        raise RuntimeError("Must call generate() before plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Create age groups
    age_bins = list(range(0, 101, 5))
    age_labels = [f'{i}-{i+4}' for i in range(0, 100, 5)]
    
    male_counts = [0] * len(age_labels)
    female_counts = [0] * len(age_labels)
    
    for ind in area.individuals:
        age_group = min(ind.age // 5, len(age_labels) - 1)
        if ind.sex == 'male':
            male_counts[age_group] += 1
        else:
            female_counts[age_group] += 1
    
    y = range(len(age_labels))
    
    # Males on left (negative), females on right (positive)
    ax.barh(y, [-c for c in male_counts], height=0.8, 
            label='Male', color='steelblue', alpha=0.8)
    ax.barh(y, female_counts, height=0.8,
            label='Female', color='coral', alpha=0.8)
    
    # Center the axis
    max_count = max(max(male_counts), max(female_counts)) if male_counts and female_counts else 1
    ax.set_xlim(-max_count * 1.1, max_count * 1.1)
    
    # Fix x-axis labels to show absolute values
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{abs(int(x))}'))
    
    ax.set_yticks(list(y))
    ax.set_yticklabels(age_labels)
    ax.set_xlabel('Population Count')
    ax.set_ylabel('Age Group')
    ax.set_title(title or f'Population Pyramid: {area.area_name}')
    ax.legend(loc='upper right')
    ax.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def compare_areas(
    areas: List["GbgArea"],
    metric: str = "population",
    figsize: tuple = (10, 6),
    title: Optional[str] = None
) -> "Figure":
    """
    Compare multiple areas side by side.
    
    Args:
        areas: List of GbgArea objects with generated populations
        metric: Metric to compare ('population', 'households', 'avg_hh_size', 
                'cars', 'cars_per_capita')
        figsize: Figure size (width, height) in inches
        title: Custom title
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    for area in areas:
        if not area._is_generated:
            raise RuntimeError(f"Area {area.area_name} must have generate() called first")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    names = [a.area_name.split()[-1] for a in areas]  # Short names
    
    if metric == "population":
        values = [len(a.individuals) for a in areas]
        ylabel = "Population"
    elif metric == "households":
        values = [len(a.households) for a in areas]
        ylabel = "Households"
    elif metric == "avg_hh_size":
        values = [len(a.individuals) / max(len(a.households), 1) for a in areas]
        ylabel = "Avg Household Size"
    elif metric == "cars":
        values = [sum(h.cars for h in a.households) for a in areas]
        ylabel = "Total Cars"
    elif metric == "cars_per_capita":
        values = [sum(h.cars for h in a.households) / max(len(a.individuals), 1) 
                  for a in areas]
        ylabel = "Cars per Capita"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    bars = ax.bar(names, values, color='steelblue', alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, values):
        label = f'{val:,.0f}' if val >= 10 else f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
               label, ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Area')
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'{ylabel} by Area')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_error_analysis(
    area: "GbgArea",
    figsize: tuple = (14, 10),
    title: Optional[str] = None
) -> "Figure":
    """
    Create detailed error analysis plots comparing synthesized vs census data.
    
    Shows:
    - Bar chart comparing actual vs synthesized counts
    - Error percentages for each category
    - Overall fit statistics
    
    Args:
        area: GbgArea object with generated population
        figsize: Figure size (width, height) in inches
        title: Custom title
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if not area._is_generated:
        raise RuntimeError("Must call generate() before plotting")
    
    comparison = area.compare_to_marginals(print_report=False)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Collect all data for plotting
    all_categories = []
    all_actual = []
    all_synth = []
    all_errors = []
    dimension_labels = []
    
    for dim_key, data in comparison.items():
        if dim_key == 'overall' or not data or 'comparison' not in data:
            continue
        
        dim_name = data.get('name', dim_key)
        for row in data['comparison']:
            all_categories.append(f"{dim_name[:10]}\n{row['category'][:12]}")
            all_actual.append(row['actual'])
            all_synth.append(row['synth'])
            all_errors.append(row['error_pct'])
            dimension_labels.append(dim_name)
    
    if not all_categories:
        logger.warning("No comparison data available")
        return fig
    
    # Upper plot: Actual vs Synthesized
    ax1 = fig.add_subplot(2, 1, 1)
    x = range(len(all_categories))
    width = 0.35
    
    bars1 = ax1.bar([i - width/2 for i in x], all_actual, width, 
                     label='Census (Actual)', color='coral', alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], all_synth, width,
                     label='Synthesized', color='steelblue', alpha=0.8)
    
    ax1.set_ylabel('Count')
    ax1.set_title('Census vs Synthesized Population by Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_categories, rotation=45, ha='right', fontsize=7)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Lower plot: Error percentages
    ax2 = fig.add_subplot(2, 1, 2)
    colors = ['green' if e <= 5 else 'orange' if e <= 15 else 'red' 
              for e in [abs(e) for e in all_errors]]
    
    bars = ax2.bar(x, all_errors, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='±5% threshold')
    ax2.axhline(y=-5, color='green', linestyle='--', alpha=0.5)
    
    ax2.set_ylabel('Error (%)')
    ax2.set_xlabel('Category')
    ax2.set_title('Synthesis Error by Category (Green=<5%, Orange=5-15%, Red=>15%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_categories, rotation=45, ha='right', fontsize=7)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add overall statistics text
    if 'overall' in comparison:
        ov = comparison['overall']
        stats_text = (f"Overall Fit:  RMSE={ov['rmse']:.1f}  "
                     f"MAE={ov['mae']:.1f}  "
                     f"Max Error={ov['max_error']}  "
                     f"Correlation={ov['correlation']:.3f}")
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title or f'Error Analysis: {area.area_name} ({area.year})', fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def plot_scatter_comparison(
    area: "GbgArea",
    figsize: tuple = (8, 8),
    title: Optional[str] = None
) -> "Figure":
    """
    Create a scatter plot of actual vs synthesized counts.
    
    Perfect synthesis would show all points on the diagonal line.
    
    Args:
        area: GbgArea object with generated population
        figsize: Figure size (width, height) in inches
        title: Custom title
        
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if not area._is_generated:
        raise RuntimeError("Must call generate() before plotting")
    
    comparison = area.compare_to_marginals(print_report=False)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect data points by dimension
    dimensions = {}
    for dim_key, data in comparison.items():
        if dim_key == 'overall' or not data or 'comparison' not in data:
            continue
        
        dim_name = data.get('name', dim_key)
        actual = [r['actual'] for r in data['comparison']]
        synth = [r['synth'] for r in data['comparison']]
        dimensions[dim_name] = (actual, synth)
    
    # Plot each dimension with different colors
    colors = plt.cm.tab10.colors
    for i, (dim_name, (actual, synth)) in enumerate(dimensions.items()):
        color = colors[i % len(colors)]
        ax.scatter(actual, synth, label=dim_name, color=color, alpha=0.7, s=50)
    
    # Add diagonal line (perfect fit)
    all_vals = []
    for actual, synth in dimensions.values():
        all_vals.extend(actual)
        all_vals.extend(synth)
    
    if all_vals:
        max_val = max(all_vals) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect fit')
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
    
    ax.set_xlabel('Census (Actual)')
    ax.set_ylabel('Synthesized')
    ax.set_title(title or f'Actual vs Synthesized: {area.area_name}')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add correlation coefficient
    if 'overall' in comparison:
        corr = comparison['overall']['correlation']
        ax.text(0.95, 0.05, f'r = {corr:.4f}', transform=ax.transAxes, 
               ha='right', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def save_all_plots(
    area: "GbgArea",
    output_dir: str = "./plots",
    format: str = "png",
    dpi: int = 150
) -> List[str]:
    """
    Generate and save all standard plots for an area.
    
    Args:
        area: GbgArea object with generated population
        output_dir: Directory to save plots
        format: Image format ('png', 'pdf', 'svg')
        dpi: Resolution for raster formats
        
    Returns:
        List of saved file paths
    """
    _check_matplotlib()
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    prefix = f"{area.area_code}_{area.area_name.split()[-1]}"
    saved = []
    
    plots = [
        ('age_distribution', plot_age_distribution),
        ('household_size', plot_household_size),
        ('population_pyramid', plot_population_pyramid),
        ('car_ownership', lambda a: plot_car_ownership(a, by='household_size')),
        ('marginal_comparison', plot_marginal_comparison),
        ('error_analysis', plot_error_analysis),
        ('scatter_comparison', plot_scatter_comparison),
    ]
    
    for name, plot_func in plots:
        try:
            fig = plot_func(area)
            if fig is not None:
                filepath = os.path.join(output_dir, f"{prefix}_{name}.{format}")
                fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                saved.append(filepath)
                logger.info(f"Saved {filepath}")
        except Exception as e:
            logger.warning(f"Failed to create {name} plot: {e}")
    
    return saved
