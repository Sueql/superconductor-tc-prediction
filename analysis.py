from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import ELEMENTS, FEATURE_COLUMNS, TARGET_COLUMN


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_element_proportion_analysis(unique_df: pd.DataFrame, outdir: Path) -> pd.Series:
    _ensure_dir(outdir)
    proportions = (unique_df[ELEMENTS] > 0).sum(axis=0) / len(unique_df)
    proportions = proportions[proportions > 0].sort_values(ascending=False)
    proportions.to_csv(outdir / 'element_proportions.csv', header=['proportion'])

    def _plot(series: pd.Series, fname: str, title: str):
        x = np.arange(1, len(series) + 1)
        plt.figure(figsize=(max(10, len(series) * 0.18), 6))
        plt.scatter(x, series.values, s=18)
        for i, (name, val) in enumerate(series.items(), start=1):
            plt.text(i, val + 0.005, name, fontsize=7, rotation=45, ha='left', va='bottom')
        plt.ylabel('Element proportion')
        plt.title(title)
        plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=220)
        plt.close()

    _plot(proportions, 'element_proportions.png', 'Element proportions in unique superconductors')
    _plot(proportions.head(20), 'element_proportions_top_20.png', 'Top 20 element proportions')
    return proportions


def run_element_temperature_summary(unique_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    _ensure_dir(outdir)
    rows = []
    for el in ELEMENTS:
        temps = unique_df.loc[unique_df[el] > 0, TARGET_COLUMN].astype(float)
        if len(temps) == 0:
            continue
        rows.append({
            'element': el,
            'Min': temps.min(),
            'Q1': temps.quantile(0.25),
            'Med': temps.median(),
            'Q3': temps.quantile(0.75),
            'Max': temps.max(),
            'Mean': temps.mean(),
            'SD': temps.std(ddof=1),
        })
    summary = pd.DataFrame(rows).dropna().sort_values('Mean', ascending=False)
    summary.to_csv(outdir / 'element_temperature_summary.csv', index=False)

    def _plot(col: str, fname: str, title: str, top_n: int | None = None):
        df = summary.head(top_n) if top_n else summary
        x = np.arange(1, len(df) + 1)
        y = df[col].to_numpy()
        plt.figure(figsize=(max(10, len(df) * 0.2), 6))
        plt.scatter(x, y, s=18)
        for i, (name, val) in enumerate(zip(df['element'], y), start=1):
            plt.text(i, val + 0.5, name, fontsize=7, rotation=45, ha='left', va='bottom')
        plt.ylabel(f'{col} critical temperature (K)')
        plt.title(title)
        plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=220)
        plt.close()

    _plot('Mean', 'mean_crit_temp_per_element.png', 'Mean critical temperature by element')
    _plot('Mean', 'mean_crit_temp_per_element_top_20.png', 'Top 20 mean critical temperatures', top_n=20)

    summary_sd = summary.sort_values('SD', ascending=False)
    summary_sd.to_csv(outdir / 'element_temperature_summary_by_sd.csv', index=False)
    x = np.arange(1, len(summary_sd) + 1)
    plt.figure(figsize=(max(10, len(summary_sd) * 0.2), 6))
    plt.scatter(x, summary_sd['SD'].to_numpy(), s=18)
    for i, (name, val) in enumerate(zip(summary_sd['element'], summary_sd['SD']), start=1):
        plt.text(i, val + 0.5, name, fontsize=7, rotation=45, ha='left', va='bottom')
    plt.ylabel('SD critical temperature (K)')
    plt.title('SD of critical temperature by element')
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(outdir / 'sd_crit_temp_per_element.png', dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(summary['Mean'], summary['SD'], s=24)
    plt.xlabel('Mean critical temperature (K)')
    plt.ylabel('SD critical temperature (K)')
    plt.title('Mean vs SD of critical temperature by element')
    plt.tight_layout()
    plt.savefig(outdir / 'sd_vs_mean_crit_temp.png', dpi=220)
    plt.close()
    return summary


def run_temperature_distribution_analysis(unique_df: pd.DataFrame, outdir: Path) -> Dict[str, float]:
    _ensure_dir(outdir)
    temps = unique_df[TARGET_COLUMN].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(temps, bins=40, color='lightgray', edgecolor='black', density=True)
    axes[0].set_xlabel('Critical temperature (K)')
    axes[0].set_title('Histogram')

    axes[1].boxplot(temps, vert=False)
    axes[1].set_xlabel('Critical temperature (K)')
    axes[1].set_title('Boxplot')

    axes[2].plot(np.sort(temps.to_numpy())[::-1], marker='o', ms=2, lw=0.5)
    axes[2].set_ylabel('Critical temperature (K)')
    axes[2].set_title('Sorted temperatures')
    plt.tight_layout()
    plt.savefig(outdir / 'critical_temp_distribution.png', dpi=220)
    plt.close(fig)

    summary = temps.describe().to_dict()
    pd.Series(summary).to_csv(outdir / 'critical_temp_summary.csv', header=['value'])
    return summary


def run_indicator_analysis(train_with_indicators: pd.DataFrame, outdir: Path) -> None:
    _ensure_dir(outdir)
    for indicator in ['iron', 'cuprate']:
        groups = [
            train_with_indicators.loc[train_with_indicators[indicator] == 0, TARGET_COLUMN].astype(float),
            train_with_indicators.loc[train_with_indicators[indicator] == 1, TARGET_COLUMN].astype(float),
        ]
        plt.figure(figsize=(6, 4))
        plt.boxplot(groups, labels=['No', 'Yes'])
        plt.xlabel(indicator.capitalize())
        plt.ylabel('Critical temperature (K)')
        plt.title(f'{indicator.capitalize()} vs critical temperature')
        plt.tight_layout()
        plt.savefig(outdir / f'{indicator}_vs_temp.png', dpi=220)
        plt.close()

        stats = pd.DataFrame({
            'group': ['No', 'Yes'],
            'count': [len(groups[0]), len(groups[1])],
            'mean': [groups[0].mean(), groups[1].mean()],
            'median': [groups[0].median(), groups[1].median()],
            'std': [groups[0].std(ddof=1), groups[1].std(ddof=1)],
            'min': [groups[0].min(), groups[1].min()],
            'max': [groups[0].max(), groups[1].max()],
        })
        stats.to_csv(outdir / f'{indicator}_group_stats.csv', index=False)


def run_feature_plots(train_with_indicators: pd.DataFrame, outdir: Path) -> None:
    _ensure_dir(outdir)
    cols = FEATURE_COLUMNS + ['iron', 'cuprate']
    with PdfPages(outdir / 'univariate_plots.pdf') as pdf:
        for col in cols:
            plt.figure(figsize=(6, 4))
            plt.scatter(train_with_indicators[col], train_with_indicators[TARGET_COLUMN], s=6, alpha=0.45)
            plt.xlabel(col)
            plt.ylabel('Critical temperature (K)')
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def run_correlation_and_pca(train_df: pd.DataFrame, outdir: Path) -> None:
    _ensure_dir(outdir)
    X = train_df[FEATURE_COLUMNS].astype(float)
    corr = X.corr()
    corr.to_csv(outdir / 'feature_correlation_matrix.csv')
    mean_abs_corr = np.abs(corr.values[np.tril_indices_from(corr.values, k=-1)]).mean()
    pd.Series({'mean_absolute_correlation': mean_abs_corr}).to_csv(
        outdir / 'correlation_summary.csv', header=['value']
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Feature correlation matrix')
    plt.tight_layout()
    plt.savefig(outdir / 'feature_correlation_matrix.png', dpi=220)
    plt.close()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    pd.DataFrame({
        'component': np.arange(1, len(cum_var) + 1),
        'cumulative_variance': cum_var,
    }).to_csv(outdir / 'pca_cumulative_variance.csv', index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o', ms=3)
    plt.axhline(0.90, linestyle='--', color='gray')
    plt.axhline(0.99, linestyle='--', color='gray')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('PCA cumulative explained variance')
    plt.tight_layout()
    plt.savefig(outdir / 'pca_cumulative_variance.png', dpi=220)
    plt.close()


def run_all_analyses(train_df: pd.DataFrame, unique_df: pd.DataFrame, train_with_indicators: pd.DataFrame, outdir: Path) -> None:
    _ensure_dir(outdir)
    run_element_proportion_analysis(unique_df, outdir / 'element_proportions')
    run_element_temperature_summary(unique_df, outdir / 'element_temperature_summary')
    run_temperature_distribution_analysis(unique_df, outdir / 'critical_temp_distribution')
    run_indicator_analysis(train_with_indicators, outdir / 'indicator_analysis')
    run_feature_plots(train_with_indicators, outdir / 'feature_plots')
    run_correlation_and_pca(train_df, outdir / 'correlation_and_pca')
