"""
fig_gen.py - Generate Section 6 figures + stats table.

Usage:
    python fig_gen.py

Outputs to ./outputs/figures/
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba
from scipy import stats as sp_stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -- Paths --------------------------------------------------------------------

OUT = Path('./outputs/figures')
OUT.mkdir(parents=True, exist_ok=True)

EXP1_CSV = Path('./data/exp1_baseline.csv')
EXP2_CSV = Path('./data/exp2_sweep.csv')


# -- Fixed sweep slice for gamma-lambda heatmaps -------------------------------

TARGET_TAU_R = 0.5
TARGET_D_C = 30.0
FIG_FONT_DELTA = 3
HEATMAP_CELL_FONT_SIZE = 9 + FIG_FONT_DELTA
COMP_HEATMAP_QRANGE = (15, 95)
MEDK_HEATMAP_QRANGE = (10, 90)


# -- Palette ------------------------------------------------------------------

NAVY = '#1E3765'
SAND = '#8F8174'
SKY = '#9DBBD8'
CYAN = '#6FC7EA'


def _alpha_cmap(name, base_hex, low_alpha=0.12, high_alpha=0.95):
    """Single-hue map where larger values appear darker on a white background."""
    base = to_rgba(base_hex)
    return LinearSegmentedColormap.from_list(
        name,
        [
            (base[0], base[1], base[2], low_alpha),
            (base[0], base[1], base[2], high_alpha),
        ],
        N=256,
    )


CMAP_COMP = _alpha_cmap('comp_alpha', NAVY, low_alpha=0.0)
CMAP_K = _alpha_cmap('medk_alpha', SAND)


# -- Global rcParams -----------------------------------------------------------

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10 + FIG_FONT_DELTA,
    'axes.labelsize': 11 + FIG_FONT_DELTA,
    'axes.titlesize': 12 + FIG_FONT_DELTA,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9 + FIG_FONT_DELTA,
    'ytick.labelsize': 9 + FIG_FONT_DELTA,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def _bool_series(series):
    """Normalize CSV bool-ish columns into actual booleans."""
    return series.astype(str).str.strip().str.lower().isin({'1', 'true', 'yes'})


def _load_dataframes():
    df1 = pd.read_csv(EXP1_CSV)
    df2 = pd.read_csv(EXP2_CSV)

    for frame in (df1, df2):
        for col in ('complete', 'fail'):
            if col in frame.columns:
                frame[col] = _bool_series(frame[col])

    for col in ('gamma', 'lambda', 'tau_r', 'k_final', 'avg_C_c'):
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors='coerce')

    if 'D_c' in df2.columns:
        df2['D_c_num'] = pd.to_numeric(df2['D_c'], errors='coerce')
        df2.loc[df2['D_c'].astype(str) == 'inf', 'D_c_num'] = np.inf

    return df1, df2


df1, df2 = _load_dataframes()


def exp2_heatmap_slice():
    """Use the fixed slice that was explicitly backfilled for the paper heatmaps."""
    return df2[
        (df2['tau_r'] == TARGET_TAU_R) &
        (df2['D_c_num'] == TARGET_D_C)
    ].copy()


def _heatmap_axes(sub):
    """Only show the gamma-lambda grid that actually exists in the fixed slice."""
    gammas = sorted(sub['gamma'].dropna().astype(int).unique().tolist())
    lambdas = sorted(sub['lambda'].dropna().astype(int).unique().tolist())
    return gammas, lambdas


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def heatmap_spines(ax):
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color('#444444')


def auto_text_color(cmap, val, vmin, vmax):
    norm = np.clip((val - vmin) / (vmax - vmin), 0, 1)
    r, g, b, a = cmap(norm)
    # Heatmaps are drawn over a white background, so perceived darkness depends
    # on alpha compositing as well as the base RGB value.
    r = a * r + (1 - a) * 1.0
    g = a * g + (1 - a) * 1.0
    b = a * b + (1 - a) * 1.0
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if lum < 0.55 else '#222222'


def _reindexed_pivot(series, gammas, lambdas):
    return series.unstack().reindex(index=gammas, columns=lambdas)


def _value_limits(values, *, mode='minmax', qrange=None):
    """Compute color scaling limits from finite values."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0

    if mode == 'percentile' and qrange is not None:
        lo, hi = np.percentile(finite, qrange)
    else:
        lo, hi = float(np.min(finite)), float(np.max(finite))

    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


# ============================================================================
# FIG 1: gamma x lambda completion heatmap
# ============================================================================
def fig_heatmap_completion():
    sub = exp2_heatmap_slice()
    gammas, lambdas = _heatmap_axes(sub)
    pivot = _reindexed_pivot(
        sub.groupby(['gamma', 'lambda'])['complete'].mean(),
        gammas, lambdas,
    ) * 100
    vmin, vmax = _value_limits(
        pivot.values, mode='percentile', qrange=COMP_HEATMAP_QRANGE
    )
    masked = np.ma.masked_invalid(pivot.values)
    cmap = CMAP_COMP.copy()
    cmap.set_bad('white')

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    im = ax.imshow(
        masked, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect='equal', origin='lower'
    )
    heatmap_spines(ax)

    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f'{int(l)}' for l in lambdas])
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f'{int(g)}' for g in gammas])
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\gamma$')

    for i in range(len(gammas)):
        for j in range(len(lambdas)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            color = auto_text_color(cmap, val, vmin, vmax)
            ax.text(
                j, i, f'{val:.0f}',
                ha='center', va='center',
                fontsize=HEATMAP_CELL_FONT_SIZE, color=color, fontweight='medium'
            )

    fig.savefig(OUT / 'fig_heatmap_completion.pdf')
    fig.savefig(OUT / 'fig_heatmap_completion.png')
    plt.close(fig)
    print('  ✓ heatmap_completion')


# ============================================================================
# FIG 2: gamma x lambda median successful k heatmap
# ============================================================================
def fig_heatmap_medk():
    sub = exp2_heatmap_slice()
    gammas, lambdas = _heatmap_axes(sub)
    completed = sub[sub['complete']]
    pivot = _reindexed_pivot(
        completed.groupby(['gamma', 'lambda'])['k_final'].median(),
        gammas, lambdas,
    )
    vmin, vmax = _value_limits(
        pivot.values, mode='percentile', qrange=MEDK_HEATMAP_QRANGE
    )
    masked = np.ma.masked_invalid(pivot.values)
    cmap = CMAP_K.copy()
    cmap.set_bad('white')

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    im = ax.imshow(
        masked, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect='equal', origin='lower'
    )
    heatmap_spines(ax)

    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f'{int(l)}' for l in lambdas])
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f'{int(g)}' for g in gammas])
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\gamma$')

    for i in range(len(gammas)):
        for j in range(len(lambdas)):
            val = pivot.values[i, j]
            if np.isnan(val):
                ax.text(
                    j, i, 'NA',
                    ha='center', va='center',
                    fontsize=HEATMAP_CELL_FONT_SIZE, color='#7a746d', fontweight='medium'
                )
                continue
            color = auto_text_color(CMAP_K, val, vmin, vmax)
            ax.text(
                j, i, f'{val:.0f}',
                ha='center', va='center',
                fontsize=HEATMAP_CELL_FONT_SIZE, color=color, fontweight='medium'
            )

    fig.savefig(OUT / 'fig_heatmap_medk.pdf')
    fig.savefig(OUT / 'fig_heatmap_medk.png')
    plt.close(fig)
    print('  ✓ heatmap_medk')


# ============================================================================
# FIG 2b: literature taxonomy venn
# ============================================================================
def fig_lit_venn():
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    circles = [
        ((-1.1, -0.3), 2.0, SKY),
        ((1.1, -0.3), 2.0, CYAN),
        ((0.0, 1.3), 2.0, SAND),
    ]

    for (x, y), radius, color in circles:
        ax.add_patch(
            plt.Circle(
                (x, y), radius,
                facecolor=to_rgba(color, 0.15),
                edgecolor=color,
                linewidth=1.8,
            )
        )

    ax.text(
        -3.2, -2.7,
        'Temporal Logic\n$\\times$ Multi-Robot',
        fontsize=10, ha='center', color=NAVY, fontweight='bold',
    )
    ax.text(
        3.2, -2.7,
        'Game Theory\n$\\times$ Multi-Robot',
        fontsize=10, ha='center', color=NAVY, fontweight='bold',
    )
    ax.text(
        0.0, 3.7,
        'Temporal Logic $\\times$ Game Theory',
        fontsize=10, ha='center', color=NAVY, fontweight='bold',
    )

    ax.text(
        -2.2, 0.0,
        'Kantaros 2020\nSchillinger 2018\nHashimoto 2022\nChen & Kan 2025',
        fontsize=5.5, ha='center', va='center', color='#555555',
    )
    ax.text(
        2.2, 0.0,
        'Monderer & Shapley 1996\nMarden 2012\nLi & Marden 2014\nLe et al. 2024',
        fontsize=5.5, ha='center', va='center', color='#555555',
    )
    ax.text(
        0.0, 2.5,
        'Gutierrez 2016\nWen & Topcu 2016\nFu & Tanner 2015\nSTLGame 2024',
        fontsize=6.5, ha='center', va='center', color='#333333',
    )
    ax.text(
        0.0, 0.5, 'Ours',
        fontsize=15, ha='center', va='center',
        color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=NAVY, edgecolor='none', alpha=0.92),
    )
    ax.text(
        0.0, -0.1,
        'Shared scLTL Mission\n+ Private Costs + Utility Design\n+ Potential-Game Structure',
        fontsize=6.5, ha='center', va='center', color=NAVY, fontweight='bold',
        linespacing=1.3,
    )

    ax.set_xlim(-4.2, 4.2)
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-3.5, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.savefig(OUT / 'fig_venn.pdf')
    fig.savefig(OUT / 'fig_venn.png')
    fig.savefig(OUT / 'fig_venn_taxonomy.pdf')
    fig.savefig(OUT / 'fig_venn_taxonomy.png')
    plt.close(fig)
    print('  ✓ venn')


# ============================================================================
# FIG 3: tau_r violin
# ============================================================================
def fig_tau_r_violin():
    sub = df2[
        (df2['gamma'].isin([1, 2, 3])) &
        (df2['lambda'] == 7) &
        (df2['D_c_num'] == TARGET_D_C)
    ].copy()
    tau_vals = sorted(sub['tau_r'].dropna().unique())

    comp_rates, comp_ci = [], []
    for tau_r in tau_vals:
        group = sub[sub['tau_r'] == tau_r]['complete']
        rate = group.mean()
        se = np.sqrt(rate * (1 - rate) / len(group))
        comp_rates.append(rate * 100)
        comp_ci.append(se * 100 * 1.96)

    data_k = [
        sub[(sub['tau_r'] == tau_r) & (sub['complete'])]['k_final'].dropna().values
        for tau_r in tau_vals
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    ax1.bar(
        range(len(tau_vals)), comp_rates, width=0.55,
        color=SKY, edgecolor=NAVY, linewidth=0.8, alpha=0.85
    )
    ax1.errorbar(
        range(len(tau_vals)), comp_rates, yerr=comp_ci,
        fmt='none', color=NAVY, capsize=5, lw=1.5
    )
    ax1.set_xticks(range(len(tau_vals)))
    ax1.set_xticklabels([f'{tau_r:.1f}' for tau_r in tau_vals])
    ax1.set_xlabel(r'$\tau_r$')
    ax1.set_ylabel('Completion Rate (%)')
    ax1.set_ylim([75, 100])
    if 0.5 in tau_vals:
        prior_idx = tau_vals.index(0.5)
        ax1.axvline(x=prior_idx, color=SAND, ls=':', lw=1.2, alpha=0.7)
        ax1.text(
            prior_idx + 0.15, 77,
            r'$B_O^{\mathrm{prior}}$', fontsize=8.5, color=SAND
        )
    despine(ax1)

    valid_data = [data for data in data_k if len(data) > 0]
    valid_pos = [i for i, data in enumerate(data_k) if len(data) > 0]
    if valid_data:
        parts = ax2.violinplot(
            valid_data, positions=valid_pos,
            showmeans=True, showmedians=True, widths=0.6
        )
        for pc in parts['bodies']:
            pc.set_facecolor(SKY)
            pc.set_edgecolor(NAVY)
            pc.set_alpha(0.7)
            pc.set_linewidth(0.8)
        for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
            if key in parts:
                parts[key].set_color(NAVY)
                parts[key].set_linewidth(1.0)
    ax2.set_xticks(range(len(tau_vals)))
    ax2.set_xticklabels([f'{tau_r:.1f}' for tau_r in tau_vals])
    ax2.set_xlabel(r'$\tau_r$')
    ax2.set_ylabel('$k$ at Completion')
    despine(ax2)

    fig.suptitle(
        rf'$\tau_r$ Sensitivity ($\gamma \in \{{1,2,3\}}$, $\lambda=7$, $D_c={int(TARGET_D_C)}$)',
        fontsize=11, fontweight='bold', y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / 'fig_tau_r.pdf')
    fig.savefig(OUT / 'fig_tau_r.png')
    plt.close(fig)
    print('  ✓ tau_r')


# ============================================================================
# FIG 4: D_c violin
# ============================================================================
def fig_dc_violin():
    sub = df2[
        (df2['gamma'].isin([1, 2, 3])) &
        (df2['lambda'] == 7) &
        (df2['tau_r'] == TARGET_TAU_R)
    ].copy()
    dc_order = sorted(sub['D_c_num'].dropna().unique())
    dc_labels = [f'{int(x)}' if np.isfinite(x) else r'$\infty$' for x in dc_order]

    comp_rates, comp_ci = [], []
    for d_c in dc_order:
        group = sub[sub['D_c_num'] == d_c]['complete']
        rate = group.mean()
        se = np.sqrt(rate * (1 - rate) / max(len(group), 1))
        comp_rates.append(rate * 100)
        comp_ci.append(se * 100 * 1.96)

    data_k = [
        sub[(sub['D_c_num'] == d_c) & (sub['complete'])]['k_final'].dropna().values
        for d_c in dc_order
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    ax1.bar(
        range(len(dc_order)), comp_rates, width=0.55,
        color=CYAN, edgecolor=NAVY, linewidth=0.8, alpha=0.85
    )
    ax1.errorbar(
        range(len(dc_order)), comp_rates, yerr=comp_ci,
        fmt='none', color=NAVY, capsize=5, lw=1.5
    )
    ax1.set_xticks(range(len(dc_order)))
    ax1.set_xticklabels(dc_labels)
    ax1.set_xlabel(r'$D_c$')
    ax1.set_ylabel('Completion Rate (%)')
    ax1.set_ylim([50, 100])
    despine(ax1)

    valid_data = [data for data in data_k if len(data) > 0]
    valid_pos = [i for i, data in enumerate(data_k) if len(data) > 0]
    if valid_data:
        parts = ax2.violinplot(
            valid_data, positions=valid_pos,
            showmeans=True, showmedians=True, widths=0.6
        )
        for pc in parts['bodies']:
            pc.set_facecolor(CYAN)
            pc.set_edgecolor(NAVY)
            pc.set_alpha(0.7)
            pc.set_linewidth(0.8)
        for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
            if key in parts:
                parts[key].set_color(NAVY)
                parts[key].set_linewidth(1.0)
    ax2.set_xticks(range(len(dc_order)))
    ax2.set_xticklabels(dc_labels)
    ax2.set_xlabel(r'$D_c$')
    ax2.set_ylabel('$k$ at Completion')
    despine(ax2)

    fig.suptitle(
        r'$D_c$ Sensitivity ($\gamma \in \{1,2,3\}$, $\lambda=7$, $\tau_r=0.5$)',
        fontsize=11, fontweight='bold', y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / 'fig_D_c.pdf')
    fig.savefig(OUT / 'fig_D_c.png')
    plt.close(fig)
    print('  ✓ D_c')


# ============================================================================
# FIG 5: copter flight cost vs lambda
# ============================================================================
def fig_copter_cost():
    sub = df2[
        (df2['gamma'] == 2) &
        (df2['tau_r'] == TARGET_TAU_R) &
        (df2['D_c_num'] == TARGET_D_C)
    ].copy()
    agg = sub.groupby('lambda')['avg_C_c'].agg(['mean', 'std', 'count']).reset_index()
    agg['se'] = agg['std'] / np.sqrt(agg['count'])

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.fill_between(
        agg['lambda'],
        agg['mean'] - 1.96 * agg['se'],
        agg['mean'] + 1.96 * agg['se'],
        color=CYAN, alpha=0.25
    )
    ax.plot(
        agg['lambda'], agg['mean'], 'o-',
        color=NAVY, lw=2, markersize=5, markerfacecolor=CYAN,
        markeredgecolor=NAVY, markeredgewidth=1.0
    )
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Mean $C_c$ per Epoch')
    despine(ax)
    ax.grid(True, alpha=0.15, lw=0.5)

    fig.tight_layout()
    fig.savefig(OUT / 'fig_copter_cost.pdf')
    fig.savefig(OUT / 'fig_copter_cost.png')
    plt.close(fig)
    print('  ✓ copter_cost')


# ============================================================================
# FIG 5b: mean copter flight cost per epoch vs lambda
# ============================================================================
def fig_copter_cost_vs_lambda():
    base = df2[
        (df2['gamma'] >= 1) &
        (df2['complete'])
    ].copy()

    thresholds = [
        (10, SAND, r'$D_c > 10$'),
        (20, CYAN, r'$D_c > 20$'),
        (30, NAVY, r'$D_c = \infty$'),
    ]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for dc_thresh, color, label in thresholds:
        sub = base[base['D_c_num'] > dc_thresh].copy()
        agg = (
            sub.groupby('lambda')['avg_C_c']
            .agg(['mean', 'std', 'count'])
            .reset_index()
            .rename(columns={'mean': 'avg_Cc', 'std': 'std_Cc', 'count': 'n'})
            .sort_values('lambda')
        )
        agg['se'] = agg['std_Cc'] / np.sqrt(agg['n'])

        print(f'\n  {label}: {len(sub)} runs')
        print(agg[['lambda', 'avg_Cc', 'n']].to_string(
            index=False, float_format=lambda x: f'{x:.4f}'
        ))

        ax.fill_between(
            agg['lambda'],
            agg['avg_Cc'] - 1.96 * agg['se'],
            agg['avg_Cc'] + 1.96 * agg['se'],
            color=color, alpha=0.12
        )
        ax.plot(
            agg['lambda'], agg['avg_Cc'],
            'o-', color=color, linewidth=1.8, markersize=3.5,
            label=f'{label} (n={len(sub):,})'
        )

    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Mean copter flight cost per epoch', fontsize=10)
    ax.set_xlim(-0.5, 10.5)
    ax.legend(fontsize=7.5, frameon=False)
    despine(ax)

    fig.tight_layout()
    fig.savefig(OUT / 'fig_copter_cost_vs_lambda.pdf')
    fig.savefig(OUT / 'fig_copter_cost_vs_lambda.png')
    plt.close(fig)
    print('  ✓ copter_cost_vs_lambda')


# ============================================================================
# FIG 6: soft costs vs hard constraints
# ============================================================================
def fig_soft_vs_hard():
    """2x2: tau_r and D_c sensitivity at gamma=0 vs gamma=2, lambda=7."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, param, label in [
        (axes[0], 'tau_r', r'$\tau_r$'),
        (axes[1], 'D_c_num', r'$D_c$'),
    ]:
        for gamma, color, marker in [(0, SAND, 's'), (2, NAVY, 'o')]:
            sub = df2[(df2['gamma'] == gamma) & (df2['lambda'] == 7)].copy()
            vals = sorted(
                sub[param].dropna().unique(),
                key=lambda value: (np.isinf(value), value),
            )
            rates = [sub[sub[param] == value]['complete'].mean() * 100 for value in vals]
            tick_labels = [
                r'$\infty$' if np.isinf(value)
                else (f'{value:.1f}' if param == 'tau_r' else f'{int(value)}')
                for value in vals
            ]
            ax.plot(
                range(len(vals)), rates, f'{marker}-',
                color=color, lw=2, markersize=6,
                label=rf'$\gamma={gamma}$'
            )
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(tick_labels)

        ax.set_xlabel(label)
        ax.set_ylabel('Completion Rate (%)')
        ax.set_ylim([-5, 105])
        ax.legend(frameon=False)
        despine(ax)

    fig.suptitle(
        r'Hard constraints ($\tau_r$, $D_c$) at $\lambda=7$: '
        r'with vs without soft cost $\gamma$',
        fontsize=11, fontweight='bold', y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / 'fig_soft_vs_hard.pdf')
    fig.savefig(OUT / 'fig_soft_vs_hard.png')
    plt.close(fig)
    print('  ✓ soft_vs_hard')


# ============================================================================
# STATS TABLE
# ============================================================================
def stats_table():
    sweep = exp2_heatmap_slice()

    def _mean_text(values, precision=0):
        values = values.dropna()
        if len(values) == 0:
            return 'NA'
        mean = values.mean()
        return f"{mean:.{precision}f}"

    print('\n' + '=' * 65)
    print('STATS TABLE FOR §6')
    print('=' * 65)

    print(
        f"\n{'Config':<20s} {'Runs':>5s} {'Completion':>10s} {'Failure':>8s} "
        f"{'Timeout':>8s} {'Median k':>8s} {'Mean k':>8s} {'Avg Phi':>8s} "
        f"{'Rover path':>12s} {'Copter path':>12s}"
    )
    print('-' * 114)
    for name in [
        'baseline_local', 'baseline_global', 'altruistic',
        'full_game', 'rover_only', 'copter_only',
    ]:
        group = df1[df1['config'] == name]
        n = len(group)
        n_complete = int(group['complete'].sum())
        n_fail = int(group['fail'].sum())
        n_timeout = n - n_complete - n_fail
        completed = group[group['complete']]
        ks = completed['k_final']
        med_k = int(ks.median()) if len(ks) > 0 else -1
        mean_k = _mean_text(ks, precision=0)
        avg_phi = _mean_text(completed['avg_Phi'], precision=1)
        rover_l = _mean_text(completed['rover_path_length'], precision=1)
        copter_l = _mean_text(completed['copter_path_length'], precision=1)
        print(
            f"  {name:<18s} {n:5d} {n_complete / n:10.1%} {n_fail / n:8.1%} "
            f"{n_timeout / n:8.1%} {med_k:8d} {mean_k:>8s} {avg_phi:>8s} "
            f"{rover_l:>12s} {copter_l:>12s}"
        )

    print(
        f"\n{'Comparison':<40s} {'Difference':>10s} {'t statistic':>12s} "
        f"{'p value':>12s} {'Significance':>14s}"
    )
    print('-' * 94)
    pairs = [
        (
            'gamma = 0 vs gamma = 1 (slice)',
            sweep[sweep['gamma'] == 0]['complete'],
            sweep[sweep['gamma'] == 1]['complete'],
        ),
        (
            'full_game vs rover_only (baseline)',
            df1[df1['config'] == 'full_game']['complete'],
            df1[df1['config'] == 'rover_only']['complete'],
        ),
        (
            'copter_only vs altruistic (baseline)',
            df1[df1['config'] == 'copter_only']['complete'],
            df1[df1['config'] == 'altruistic']['complete'],
        ),
        (
            'altruistic vs baseline_local',
            df1[df1['config'] == 'altruistic']['complete'],
            df1[df1['config'] == 'baseline_local']['complete'],
        ),
    ]
    for label, a, b in pairs:
        t_val, p_val = sp_stats.ttest_ind(a, b)
        diff = a.mean() - b.mean()
        significance = 'significant' if p_val < 0.05 else 'not significant'
        print(
            f"  {label:<38s} {diff:+10.3f} {t_val:12.1f} {p_val:12.2e} "
            f"{significance:>14s}"
        )

    lo = sweep[(sweep['gamma'] == 2) & (sweep['lambda'] <= 2)]['avg_C_c'].dropna()
    hi = sweep[(sweep['gamma'] == 2) & (sweep['lambda'] >= 6)]['avg_C_c'].dropna()
    t_val, p_val = sp_stats.ttest_ind(lo, hi)
    significance = 'significant' if p_val < 0.05 else 'not significant'
    print(
        f"\n  Copter cost lambda <= 2: {_mean_text(lo, precision=2)}"
        f"  vs lambda >= 6: {_mean_text(hi, precision=2)}"
        f"  (t statistic={t_val:.1f}, p value={p_val:.2e}, {significance})"
    )

    g0 = sweep[sweep['gamma'] == 0]
    print(
        f"\n  gamma = 0 on slice: completion={g0['complete'].mean():.3f} "
        f"(runs={len(g0)}, max by lambda: "
        f"{g0.groupby('lambda')['complete'].mean().max():.3f})"
    )

    # -- LaTeX table --
    configs_display = [
        ('Baseline local~\\cite{hashimoto2022}', 'baseline_local', True),
        ('Baseline global~\\cite{hashimoto2022}', 'baseline_global', True),
        (None, None, None),  # midrule
        ('Both altruistic', 'altruistic', False),
        ('Copter-only', 'copter_only', False),
        ('Rover-only', 'rover_only', False),
        ('Full game (ours)', 'full_game', False),
    ]

    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\caption{Baseline comparison across 1000 shared random starts.')
    lines.append(r'  Mean $k$, average $\Phi$, and path lengths are reported for completed runs')
    lines.append(r'  only.')
    lines.append(r'  All game-theoretic configurations use hard constraints')
    lines.append(r'  $\tau_r = 0.5$, $D_c = 30$.')
    lines.append(r'  Each run terminates in one of three outcomes:')
    lines.append(r'  completion (rover reaches an accepting FSA state),')
    lines.append(r'  failure (rover enters an obstacle cell),')
    lines.append(r'  or timeout (the horizon is reached without completion or failure).}')
    lines.append(r'\label{tab:baseline}')
    lines.append(r'\centering\small')
    lines.append(r'\renewcommand{\arraystretch}{1.3}')
    lines.append(r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}')
    lines.append(r'  l cc ccc cc cc@{}}')
    lines.append(r'\toprule')
    lines.append(r' & \multicolumn{2}{c}{Parameters}')
    lines.append(r' & \multicolumn{3}{c}{Outcome (\%)}')
    lines.append(r' & \multicolumn{2}{c}{Efficiency}')
    lines.append(r' & \multicolumn{2}{c}{Path Length} \\')
    lines.append(r'\cmidrule(lr){2-3} \cmidrule(lr){4-6}')
    lines.append(r'\cmidrule(lr){7-8} \cmidrule(lr){9-10}')
    lines.append(r'Method & $\gamma$ & $\lambda$')
    lines.append(r'  & Completion & Failure & Timeout')
    lines.append(r'  & Mean $k$ & Avg. $\Phi$')
    lines.append(r'  & Rover & Copter \\')
    lines.append(r'\midrule')

    for label, key, is_baseline in configs_display:
        if key is None:
            lines.append(r'\midrule')
            continue

        g = df1[df1['config'] == key]
        comp = g['complete'].mean() * 100
        fail = g['fail'].mean() * 100
        to = 100 - comp - fail
        completed = g[g['complete']]
        n_comp = len(completed)
        mean_k = completed['k_final'].mean() if n_comp > 0 else float('nan')
        avg_phi = completed['avg_Phi'].mean() if n_comp > 0 else float('nan')
        rov_l = completed['rover_path_length'].mean() if n_comp > 0 else float('nan')
        cop_l = completed['copter_path_length'].mean() if n_comp > 0 else float('nan')

        is_ours = (key == 'full_game')
        b = lambda s: f'\\textbf{{{s}}}' if is_ours else s

        if is_baseline:
            gamma_str = '--'
            lambda_str = '--'
        else:
            gamma_val = pd.to_numeric(g['gamma'], errors='coerce').iloc[0]
            lambda_val = pd.to_numeric(g['lambda'], errors='coerce').iloc[0]
            gamma_str = str(int(gamma_val))
            lambda_str = str(int(lambda_val))

        k_str = b(f'{mean_k:.0f}')
        phi_str = f'{avg_phi:.1f}'
        rov_str = f'{rov_l:.1f}'
        cop_str = b(f'{cop_l:.1f}')

        line = f'{label}\n'
        line += f'  & {gamma_str} & {lambda_str}\n'
        line += f'  & {b(f"{comp:.1f}")} & {b(f"{fail:.1f}")} & {b(f"{to:.1f}")}\n'
        line += f'  & {k_str} & {phi_str}\n'
        line += f'  & {rov_str} & {cop_str} \\\\'
        lines.append(line)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular*}')
    lines.append(r'\end{table*}')

    tex = '\n'.join(lines)
    tex_path = OUT / 'tab_baseline.tex'
    tex_path.write_text(tex)
    print(f'\n  ✓ LaTeX table → {tex_path}')
    print('\n' + tex)


if __name__ == '__main__':
    print('Generating figures...')
    fig_heatmap_completion()
    fig_heatmap_medk()
    fig_lit_venn()
    fig_soft_vs_hard()
    fig_tau_r_violin()
    fig_dc_violin()
    fig_copter_cost()
    fig_copter_cost_vs_lambda()
    stats_table()
    print(f'\nAll outputs in {OUT}')
