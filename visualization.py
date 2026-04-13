"""
visualization.py

- render plots and animations from one run history
- format titles, metadata, and helper overlays for figures

function
- `make_trajectories_plot`: plot rover and copter paths
- `make_convergence_plot`: plot epoch metrics
- `make_beliefs_animation`: animate belief updates
- `make_unified_animation`: animate beliefs and trajectories together
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter

from environment import GRID, AP_LIST, TRUE_L

# --- Shared Plot Helpers ---
PLOT_FONT_FAMILY = 'Times New Roman'
plt.rcParams.update({
    'font.family': PLOT_FONT_FAMILY,
    'font.serif': [PLOT_FONT_FAMILY],
    'mathtext.fontset': 'custom',
    'mathtext.rm': PLOT_FONT_FAMILY,
    'mathtext.it': f'{PLOT_FONT_FAMILY}:italic',
    'mathtext.bf': f'{PLOT_FONT_FAMILY}:bold',
})

# Colour scheme for each atomic proposition
# AP_COLOURS = {
#     "A": "#D2A0A0",
#     "B": "#D4BC8A",
#     "C": "#A8B79F",
#     "D": "#9CAAC4",
#     "O": "#6A6663",
# }
AP_COLOURS = {
    "A": "#C97C7C",  # dusty rose
    "B": "#C9A66B",  # muted ochre
    "C": "#8FA889",  # sage green
    "D": "#7E93B2",  # dusty blue
    "O": "#4A4745",  # warm charcoal
}
ROVER_COLOR = '#1E3765'
COPTER_COLOR = '#6FC7EA'
PATH_ALPHA = 0.6
PATH_LINEWIDTH = 4.7
ROVER_PATH_ROW_OFFSET = 0.07
ROVER_PATH_COL_OFFSET = -0.07
COPTER_PATH_ROW_OFFSET = -0.07
COPTER_PATH_COL_OFFSET = 0.07
AP_GRID_ALPHA = 0.15
AP_OVERLAY_ALPHA = 1.0
AP_OVERLAY_FONT_SIZE = 26
AP_OVERLAY_COMPACT_FONT_SIZE = 26
AP_OVERLAY_LINESPACING = 1.5
LEGEND_FONT_SIZE = 18
TRAJECTORY_LEGEND_FONT_SIZE = 12
AXIS_TICK_FONT_SIZE = 20
ANIMATION_HEADER_FONT_SIZE = 14
ANIMATION_META_FONT_SIZE = 14
ANIMATION_LEGEND_FONT_SIZE = LEGEND_FONT_SIZE
ANIMATION_TICK_FONT_SIZE = AXIS_TICK_FONT_SIZE
ANIMATION_AP_FONT_SIZE = AP_OVERLAY_COMPACT_FONT_SIZE
ANIMATION_V_FONT_SIZE = 4.5


def _is_baseline_history(history: dict) -> bool:
    """Return whether one exported history came from the baseline path."""
    return history.get('copter_mode') in ('local', 'global')


def _agent_palette(history: dict) -> dict[str, str]:
    """Return the baseline or game-theoretic agent palette for one run."""
    return {
        'rover': ROVER_COLOR,
        'rover_path': ROVER_COLOR,
        'copter': COPTER_COLOR,
        'copter_path': COPTER_COLOR,
    }

def _branch_aps(which_phi) -> list[str]:
    """Return the AP panels associated with the completed mission branch."""
    which_phi = str(which_phi)
    if 'φ1' in which_phi:
        return ['A', 'O']
    if 'φ2' in which_phi:
        return ['B', 'C', 'O']
    return ['C', 'D', 'O']


def _run_status(history: dict, *, obstacle_detail: bool = False) -> str:
    """Return one human-readable run status string for plot titles."""
    which_phi = history.get('which_phi', '')
    if history['complete']:
        return f"Complete via {which_phi}"
    if history['fail']:
        return "Failed (obstacle)" if obstacle_detail else "Failed"
    return "Timeout"


def _format_math_float(value, *, decimals: int | None = None) -> str | None:
    """Format one value for compact mathtext metadata."""
    if value is None:
        return None
    value = float(value)
    if np.isinf(value):
        return r'\infty'
    if decimals is not None:
        return f'{value:.{decimals}f}'
    return f'{value:g}'


def _format_run_metadata(history: dict, *, include_seed: bool = True) -> str:
    """Build one compact metadata string for figure titles."""
    seed = history.get('seed')
    rover_path_length = history.get('rover_path_length')
    copter_path_length = history.get('copter_path_length')
    solve_time_s = history.get('solve_time_s')

    fields = []
    if include_seed and seed is not None:
        fields.append(rf'$\mathrm{{seed}}={int(float(seed))}$')
    if rover_path_length is not None:
        fields.append(rf'$L_r={_format_math_float(rover_path_length, decimals=2)}$')
    if copter_path_length is not None:
        fields.append(rf'$L_c={_format_math_float(copter_path_length, decimals=2)}$')
    if solve_time_s is not None:
        fields.append(rf'$\mathrm{{solve}}={_format_math_float(solve_time_s, decimals=2)}s$')
    return '  |  '.join(fields)


def _epoch_span(history: dict) -> int:
    """Return one planning-cycle span in scheduled timesteps."""
    return max(1, int(history.get('T_c', 0)) + int(history.get('T_r', 0)))


def _title_with_scenario(history: dict, title: str) -> str:
    """Prefix one figure title with scenario name."""
    name = str(history.get('scenario_name', '')).strip()
    return f'[{name}]  {title}' if name else title


def _agent_legend_labels(history: dict) -> tuple[str, str]:
    """Return rover/copter legend labels with relevant parameters when present."""
    if _is_baseline_history(history):
        return 'Rover', 'Copter'

    rover_parts = []
    copter_parts = []

    gamma = history.get('gamma')
    if gamma is not None:
        rover_parts.append(rf'\gamma={_format_math_float(gamma)}')

    tau_r = history.get('tau_r')
    if tau_r is not None:
        rover_parts.append(rf'\tau_r={_format_math_float(tau_r)}')

    lambda_ = history.get('lambda_', history.get('lambda'))
    if lambda_ is not None:
        copter_parts.append(rf'\lambda={_format_math_float(lambda_)}')

    D_c_budget = history.get('D_c_budget', history.get('D_c'))
    if D_c_budget is not None:
        copter_parts.append(rf'D_c={_format_math_float(D_c_budget)}')

    rover_label = 'Rover'
    if rover_parts:
        rover_label = rf"Rover (${' , '.join(rover_parts)}$)"

    copter_label = 'Copter'
    if copter_parts:
        copter_label = rf"Copter (${' , '.join(copter_parts)}$)"

    return rover_label, copter_label


def _animation_title(history: dict, *, phase_text: str, q_value: int) -> str:
    """Build one unified-animation title with phase and status."""
    title = f'[{phase_text}]   q={q_value}  —  {_run_status(history)}'
    return _title_with_scenario(history, title)


def _ap_overlay_text(ap: str) -> str:
    """Return one AP label."""
    return ap


def _annotate_true_labels(ax, *, fontsize: float,
                          color_override: str | None = None) -> None:
    """Annotate ground-truth AP cells at each cell's top-left corner."""
    for (r, c), labels in TRUE_L.items():
        for ap in labels:
            ax.text(
                c - 0.46,
                r - 0.43,
                _ap_overlay_text(ap),
                ha='left',
                va='top',
                fontsize=fontsize,
                color=color_override or AP_COLOURS.get(ap, 'yellow'),
                alpha=AP_OVERLAY_ALPHA,
                fontweight='bold',
                linespacing=AP_OVERLAY_LINESPACING,
            )


def _figure_header_layout(*, has_subtitle: bool,
                          has_metadata: bool) -> dict[str, float]:
    """Return local layout values for one figure-level header."""
    suptitle_y = 0.99
    subtitle_gap = 0.045
    metadata_gap = 0.045

    if has_subtitle and has_metadata:
        top_margin = 0.97
    elif has_metadata:
        top_margin = 0.97
    elif has_subtitle:
        top_margin = 0.96
    else:
        top_margin = 0.98

    return {
        'title_font_size': LEGEND_FONT_SIZE,
        'meta_font_size': LEGEND_FONT_SIZE,
        'suptitle_y': suptitle_y,
        'subtitle_gap': subtitle_gap,
        'metadata_gap': metadata_gap,
        'top_margin': top_margin,
    }


def _set_figure_header(fig, *, title: str, subtitle: str | None = None,
                       metadata: str | None = None) -> None:
    """Set a two- or three-line figure header with smaller secondary text."""
    layout = _figure_header_layout(
        has_subtitle=bool(subtitle),
        has_metadata=bool(metadata),
    )
    fig.suptitle(
        title,
        fontsize=layout['title_font_size'],
        fontfamily=PLOT_FONT_FAMILY,
        fontweight='bold',
        y=layout['suptitle_y'],
        va='top',
    )
    text_y = layout['suptitle_y'] - layout['subtitle_gap']
    if subtitle:
        fig.text(0.5, text_y, subtitle, ha='center', va='top',
                 fontsize=layout['meta_font_size'],
                 fontfamily=PLOT_FONT_FAMILY)
        text_y -= layout['metadata_gap']
    if metadata:
        fig.text(0.5, text_y, metadata, ha='center', va='top',
                 fontsize=layout['meta_font_size'],
                 fontfamily=PLOT_FONT_FAMILY)
    plt.tight_layout(rect=(0, 0, 1, layout['top_margin']))


def _set_square_grid_axes(ax) -> None:
    """Force one grid-world axes to render square cells."""
    ax.set_aspect('equal', adjustable='box')


def _set_grid_ticks(ax, tick_values, *, show_ticks: bool) -> None:
    """Show or hide one grid-world axes tick set."""
    if show_ticks:
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.tick_params(labelsize=AXIS_TICK_FONT_SIZE)
        return
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )


def _draw_trajectory_base(ax, *, show_ticks: bool, annotate_labels: bool = True) -> None:
    """Draw the static map background shared by trajectory images and videos."""
    for i in range(GRID + 1):
        ax.axhline(i - 0.5, color='#ddd', lw=0.6)
        ax.axvline(i - 0.5, color='#ddd', lw=0.6)
    for (r, c), labels in TRUE_L.items():
        for ap in labels:
            color = AP_COLOURS.get(ap)
            if color:
                ax.add_patch(mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=color, alpha=AP_GRID_ALPHA, edgecolor='none'))
    if annotate_labels:
        _annotate_true_labels(ax, fontsize=AP_OVERLAY_FONT_SIZE)
    _set_square_grid_axes(ax)
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(GRID - 0.5, -0.5)
    _set_grid_ticks(ax, range(GRID), show_ticks=show_ticks)


def plot_belief_map(ax, beliefs: dict, ap: str,
                    rover_pos=None, copter_pos=None,
                    rover_color=ROVER_COLOR,
                    copter_color=COPTER_COLOR,
                    show_ticks: bool = True) -> None:
    """Plot one belief heatmap for one atomic proposition."""
    grid = np.array([[beliefs[(r, c)][ap] for c in range(GRID)]
                     for r in range(GRID)])
    im = ax.imshow(grid, vmin=0, vmax=1, cmap='gray_r', origin='upper',
                   aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    _annotate_true_labels(ax, fontsize=AP_OVERLAY_FONT_SIZE)

    if rover_pos is not None:
        ax.plot(rover_pos[1], rover_pos[0], 'o', color=rover_color,
                markersize=9, markeredgecolor='white', markeredgewidth=1.2)
    if copter_pos is not None:
        ax.plot(copter_pos[1], copter_pos[0], 'o', color=copter_color,
                markersize=9, markeredgecolor='white', markeredgewidth=1.2)

    _set_square_grid_axes(ax)
    ax.set_title(f'Belief: {ap}', fontsize=LEGEND_FONT_SIZE)
    _set_grid_ticks(ax, range(0, GRID, 2), show_ticks=show_ticks)


# --- Static Plots ---
def make_belief_snapshots_plot(history: dict, filepath: str,
                               show_ticks: bool = True,
                               show_header: bool = True) -> None:
    """Plot the sparse belief snapshots saved during one run."""
    which = history.get('which_phi', '')
    palette = _agent_palette(history)
    APs = _branch_aps(which)
    fig_cols = len(APs)

    snaps   = history['beliefs_snapshot']
    snap_r  = history['snap_rover']
    snap_c  = history['snap_copter']
    n_snaps = len(snaps)

    cell_size = 4
    fig, axes = plt.subplots(n_snaps, fig_cols,
                             figsize=(cell_size * fig_cols, cell_size * n_snaps))
    if n_snaps == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_snaps):
        for j, ap in enumerate(APs):
            ax  = axes[i, j]
            rp  = snap_r[i] if i < len(snap_r) else None
            cp  = snap_c[i] if i < len(snap_c) else None
            plot_belief_map(ax, snaps[i], ap,
                            rover_pos  = rp if ap != 'O' else None,
                            copter_pos = cp if ap == 'O' else None,
                            rover_color=palette['rover'],
                            copter_color=palette['copter'],
                            show_ticks=show_ticks)

    status = _run_status(history, obstacle_detail=True)
    metadata = _format_run_metadata(history)
    subtitle = (
        '(Blue ● = rover  |  Red ● = copter  |  dark = high belief)'
        if _is_baseline_history(history)
        else '(● = rover  |  ● = copter  |  dark = high belief)'
    )
    if show_header:
        _set_figure_header(
            fig,
            title=_title_with_scenario(
                history,
                f'Environmental Belief Snapshots  —  {status}',
            ),
            subtitle=subtitle,
            metadata=metadata or None,
        )
    else:
        plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def make_trajectories_plot(history: dict, filepath: str,
                         use_substeps: bool = True,
                         show_ticks: bool = True,
                         show_header: bool = True) -> None:
    """Plot rover and copter trajectories on the ground-truth grid."""
    palette = _agent_palette(history)
    rover_label, copter_label = _agent_legend_labels(history)
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_trajectory_base(ax, show_ticks=show_ticks, annotate_labels=True)

    if use_substeps and 'rover_substeps' in history:
        rpath = np.array(history['rover_substeps'])
        cpath = np.array(history['copter_substeps'])
    else:
        rpath = np.array(history['rover_path'])
        cpath = np.array(history['copter_path'])

    rpath_rows = rpath[:, 0] + ROVER_PATH_ROW_OFFSET
    rpath_cols = rpath[:, 1] + ROVER_PATH_COL_OFFSET
    cpath_rows = cpath[:, 0] + COPTER_PATH_ROW_OFFSET
    cpath_cols = cpath[:, 1] + COPTER_PATH_COL_OFFSET

    ax.plot(rpath_cols, rpath_rows, '-', color=palette['rover_path'],
            lw=PATH_LINEWIDTH, alpha=PATH_ALPHA, label=rover_label)
    ax.plot(cpath_cols, cpath_rows, '-', color=palette['copter_path'],
            lw=PATH_LINEWIDTH, alpha=PATH_ALPHA, label=copter_label)
    ax.plot(rpath_cols[0], rpath_rows[0], 'o', color=palette['rover'],
            markersize=6)
    ax.plot(cpath_cols[0], cpath_rows[0], 'o', color=palette['copter'],
            markersize=6)
    ax.plot(rpath_cols[-1], rpath_rows[-1], 'x', color=palette['rover_path'],
            markersize=10, markeredgewidth=2.0)
    ax.plot(cpath_cols[-1], cpath_rows[-1], 'x', color=palette['copter_path'],
            markersize=10, markeredgewidth=2.0)

    status = _run_status(history)
    metadata = _format_run_metadata(history)
    if show_header:
        _set_figure_header(
            fig,
            title=_title_with_scenario(
                history,
                f'(k={history["k_final"]})  —  {status}',
            ),
            metadata=metadata or None,
        )
    else:
        plt.tight_layout()
    ax.legend(fontsize=TRAJECTORY_LEGEND_FONT_SIZE, loc='upper right')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def make_final_beliefs_plot(history: dict, filepath: str,
                            show_ticks: bool = True,
                            show_header: bool = True) -> None:
    """
    Final belief maps for all 5 APs side by side.
    """
    beliefs = history['beliefs_snapshot'][-1]
    palette = _agent_palette(history)
    metadata = _format_run_metadata(history)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, ap in enumerate(AP_LIST):
        plot_belief_map(axes[i], beliefs, ap,
                        rover_pos  = history['final_rover'],
                        copter_pos = history['final_copter'],
                        rover_color=palette['rover'],
                        copter_color=palette['copter'],
                        show_ticks=show_ticks)
    if show_header:
        _set_figure_header(
            fig,
            title=_title_with_scenario(history, 'Final Belief Maps (all APs)'),
            metadata=metadata or None,
        )
    else:
        plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def make_v_fxn_heatmap(beliefs: dict, filepath: str, rover_pos: tuple,
                       rover_q: int, V: dict | None = None) -> None:
    """
    Plot the rover value-function slice for one automaton state.

    If ``V`` is not provided, this computes one fresh D* Lite solution on the
    current belief map and visualizes its ``rover_q`` slice.
    """
    if V is None:
        from planning import DStarLitePlanner, grid_to_graph
        V, _, _ = DStarLitePlanner(grid_to_graph(), beliefs).solution()
    grid  = np.array([[V.get((r, c, rover_q), 0.0) for c in range(GRID)]
                      for r in range(GRID)])
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap='viridis', origin='upper',
                   aspect='equal')
    plt.colorbar(im, ax=ax, label='V (belief of mission completion)')
    _annotate_true_labels(
        ax,
        fontsize=AP_OVERLAY_COMPACT_FONT_SIZE,
        color_override='white',
    )
    ax.plot(rover_pos[1], rover_pos[0], 'r*', markersize=14, label='Rover')
    ax.set_title(f'Value Function V(·, q={rover_q})', fontsize=LEGEND_FONT_SIZE)
    _set_square_grid_axes(ax)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def make_convergence_plot(history: dict, filepath: str,
                          obstacle_cells=None, free_cells=None,
                          show_header: bool = True) -> None:
    """
    Plot belief convergence for obstacle and free cells.

    This is the compact 2-panel version:
        * Obstacle cells  — B(x|=O) -> 1
        * Free cells      — B(x|=O) -> 0
    """
    belief_history = history.get('belief_history')
    if belief_history is None:
        print("  make_convergence_plot: belief history missing from history object")
        return

    k_list = history['k_list']
    n = min(len(belief_history), len(k_list))
    ks = list(k_list[:n])

    if obstacle_cells is None:
        obstacle_cells = [rc for rc, lbl in TRUE_L.items() if 'O' in lbl][:3]
    if free_cells is None:
        free_cells = [rc for rc, lbl in TRUE_L.items() if not lbl and rc[0] < 5][:3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metadata = _format_run_metadata(history)
    if show_header:
        _set_figure_header(
            fig,
            title=_title_with_scenario(history, 'belief convergence'),
            metadata=metadata or None,
        )
    else:
        plt.tight_layout()

    colors_obs = ['#d62728', '#e377c2', '#ff7f0e']
    colors_free = ['#1f77b4', '#17becf', '#2ca02c']

    ax_obs, ax_free = axes

    for i, cell in enumerate(obstacle_cells):
        vals = [belief_history[t][cell]['O'] for t in range(n)]
        ax_obs.plot(ks, vals, color=colors_obs[i % len(colors_obs)], lw=1.8,
                    label=f'cell {cell}')
    ax_obs.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5, label='target = 1')
    ax_obs.set_ylim(-0.05, 1.05)
    ax_obs.set_xlabel('time step k', fontsize=10)
    ax_obs.set_ylabel('B(x |= O)', fontsize=10)
    ax_obs.set_title('obstacle cells: B(x|=O) -> 1',
                     fontsize=LEGEND_FONT_SIZE, fontweight='bold')
    ax_obs.grid(alpha=0.25)
    ax_obs.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')

    for i, cell in enumerate(free_cells):
        vals = [belief_history[t][cell]['O'] for t in range(n)]
        ax_free.plot(ks, vals, color=colors_free[i % len(colors_free)], lw=1.8,
                     label=f'cell {cell}')
    ax_free.axhline(0.0, color='black', lw=0.8, ls='--', alpha=0.5, label='target = 0')
    ax_free.set_ylim(-0.05, 1.05)
    ax_free.set_xlabel('time step k', fontsize=10)
    ax_free.set_ylabel('B(x |= O)', fontsize=10)
    ax_free.set_title('free cells: B(x|=O) -> 0',
                      fontsize=LEGEND_FONT_SIZE, fontweight='bold')
    ax_free.grid(alpha=0.25)
    ax_free.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')

    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# --- Animation Helpers ---
def _subsample_for_cycles(history: dict, belief_history: list) -> tuple:
    """Subsample belief history from per-step traces down to planning cycles."""
    rover_path  = history['rover_path']
    copter_path = history['copter_path']
    n_cycles    = len(rover_path)
    n_bh        = len(belief_history)
    step        = max(1, n_bh // n_cycles) if n_cycles else 1

    bh = [belief_history[min(i * step, n_bh - 1)] for i in range(n_cycles)]
    kl = [history['k_list'][min(i * step, len(history['k_list']) - 1)]
          for i in range(n_cycles)]

    if n_cycles > 0:
        bh[-1] = history['belief_history'][-1]
        kl[-1] = history['k_list'][-1]

    return rover_path, copter_path, bh, kl


def _save_animation(anim, filepath: str, fps: int) -> None:
    """Save an animation as mp4, falling back to gif when ffmpeg is unavailable."""
    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(filepath, writer=writer)
        print(f"  Saved animation: {filepath}")
    except Exception as e:
        gif_path = filepath.replace('.mp4', '.gif')
        print(f"  ffmpeg failed ({e}), saving as gif: {gif_path}")
        anim.save(gif_path, writer='pillow', fps=fps)


# --- Animations ---
def make_beliefs_animation(history: dict, filepath: str,
                   fps: int = 4, use_substeps: bool = True) -> None:
    """
    Animate per-AP belief panels.

    APs shown depend on which mission branch completed:
      φ1 -> [A, O]       φ2 -> [B, C, O]       φ3 -> [C, D, O]

    use_substeps=True  (default): one frame per timestep k - smooth.
    use_substeps=False: one frame per planning cycle - fast preview,
                        but last frame is guaranteed to show the true
                        final k and belief state.
    """
    APs = _branch_aps(history.get('which_phi', ''))
    palette = _agent_palette(history)

    belief_history = history.get('belief_history')
    if belief_history is None:
        print("  no belief history available in history object")
        return

    if use_substeps and 'rover_substeps' in history:
        rover_path  = history['rover_substeps']
        copter_path = history['copter_substeps']
        k_list      = history['k_list']
        bh          = belief_history
    else:
        rover_path, copter_path, bh, k_list = _subsample_for_cycles(
            history, belief_history)

    n_frames   = min(len(bh), len(rover_path), len(copter_path), len(k_list))
    phase_list = history.get('phase_list', None)
    metadata = _format_run_metadata(history)
    header_layout = _figure_header_layout(
        has_subtitle=False,
        has_metadata=bool(metadata),
    )
    cell_size = 4.5
    fig, axes = plt.subplots(1, len(APs), figsize=(cell_size * len(APs), cell_size))
    plt.subplots_adjust(wspace=0.35, top=header_layout['top_margin'])
    header_title = fig.suptitle(
        '',
        fontsize=header_layout['title_font_size'],
        fontfamily=PLOT_FONT_FAMILY,
        fontweight='bold',
        y=header_layout['suptitle_y'],
        va='top',
    )
    if metadata:
        fig.text(0.5,
                 header_layout['suptitle_y'] - header_layout['subtitle_gap'],
                 metadata,
                 ha='center', va='top',
                 fontsize=header_layout['meta_font_size'],
                 fontfamily=PLOT_FONT_FAMILY)

    def draw_frame(i):
        for ax, ap in zip(axes, APs):
            ax.clear()
            beliefs_i = bh[i]
            grid = np.array([[beliefs_i[(r, c)][ap] for c in range(GRID)]
                             for r in range(GRID)])
            ax.imshow(grid, vmin=0, vmax=1, cmap='gray_r',
                      origin='upper', aspect='equal')
            _annotate_true_labels(
                ax,
                fontsize=AP_OVERLAY_COMPACT_FONT_SIZE,
            )
            rp = rover_path[min(i, len(rover_path) - 1)]
            cp = copter_path[min(i, len(copter_path) - 1)]
            if ap != 'O':
                ax.plot(rp[1], rp[0], 'o', color=palette['rover'],
                        markersize=9, markeredgecolor='white')
                trail = rover_path[:i + 1]
                if len(trail) > 1:
                    tr = np.array(trail)
                    ax.plot(tr[:, 1] + ROVER_PATH_COL_OFFSET,
                            tr[:, 0] + ROVER_PATH_ROW_OFFSET, '-',
                            color=palette['rover_path'],
                            lw=PATH_LINEWIDTH, alpha=0.5)
            if ap == 'O':
                ax.plot(cp[1], cp[0], 'o', color=palette['copter'],
                        markersize=9, markeredgecolor='white')
                trail = copter_path[:i + 1]
                if len(trail) > 1:
                    tr = np.array(trail)
                    ax.plot(tr[:, 1] + COPTER_PATH_COL_OFFSET,
                            tr[:, 0] + COPTER_PATH_ROW_OFFSET, '-',
                            color=palette['copter_path'],
                            lw=PATH_LINEWIDTH, alpha=0.5)
            _set_square_grid_axes(ax)
            ax.set_title(f'Belief: {ap}', fontsize=LEGEND_FONT_SIZE)
            ax.set_xticks(range(0, GRID, 2))
            ax.set_yticks(range(0, GRID, 2))
            ax.tick_params(labelsize=AXIS_TICK_FONT_SIZE)

        phase  = phase_list[min(i, len(phase_list) - 1)] if phase_list else '?'
        pstr   = {'C': 'Copter exploring', 'R': 'Rover executing',
                  'W': 'Warmup', 'I': 'Init', 'E': 'End'}.get(phase, phase)
        header_title.set_text(
            _title_with_scenario(history, f'[{pstr}]')
        )

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=200)
    _save_animation(anim, filepath, fps)
    plt.close(fig)


def make_unified_animation(history: dict, filepath: str,
                           fps: int = 4, use_substeps: bool = True) -> None:
    """Animate obstacle belief, agent motion, and planner value overlays."""
    palette = _agent_palette(history)
    belief_history = history.get('belief_history')
    if belief_history is None:
        print("  no belief history available in history object")
        return
    if use_substeps and 'rover_substeps' in history:
        rover_path  = history['rover_substeps']
        copter_path = history['copter_substeps']
        k_list      = history['k_list']
        bh          = belief_history
    else:
        rover_path, copter_path, bh, k_list = _subsample_for_cycles(
            history, belief_history)
    phase_list = history.get('phase_list', None)
    n_frames   = min(len(bh), len(rover_path), len(copter_path), len(k_list))
    metadata = _format_run_metadata(history, include_seed=True)
    v_history    = history.get('v_history', None)  # list of V dicts, one per VI call
    fsa_states   = history.get('fsa_states', [0])  # rover FSA state per cycle
    epoch_span   = _epoch_span(history)

    # Build per-frame V lookup: map each substep frame to the most recent V.
    # v_history has one entry per planner call, while k_list has one entry per substep.
    def get_V_for_frame(frame_idx):
        if v_history is None or len(v_history) == 0:
            return None
        k_val = k_list[min(frame_idx, len(k_list) - 1)]
        cycle = min(k_val // epoch_span, len(v_history) - 1)
        return v_history[cycle]

    def get_q_for_frame(frame_idx):
        if not fsa_states:
            return 0
        k_val  = k_list[min(frame_idx, len(k_list) - 1)]
        cycle  = min(k_val // epoch_span, len(fsa_states) - 1)
        return fsa_states[cycle]

    header_layout = _figure_header_layout(
        has_subtitle=False,
        has_metadata=bool(metadata),
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_trajectory_base(ax, show_ticks=True, annotate_labels=True)

    im_obs = ax.imshow(
        np.full((GRID, GRID), 0.5), vmin=0, vmax=1,
        cmap='Greys', origin='upper', aspect='equal',
        alpha=0.6, zorder=3, interpolation='nearest')

    rover_dot,    = ax.plot([], [], 'o', color=palette['rover'], ms=6, zorder=6)
    copter_dot,   = ax.plot([], [], 'o', color=palette['copter'], ms=6, zorder=6)
    rover_trail,  = ax.plot([], [], '-', color=palette['rover_path'], lw=PATH_LINEWIDTH,
                            alpha=PATH_ALPHA, zorder=5)
    copter_trail, = ax.plot([], [], '-', color=palette['copter_path'],
                            lw=PATH_LINEWIDTH,
                            alpha=PATH_ALPHA, zorder=5)

    # V value texts - small number in top-left corner of each cell.
    v_texts = {
        (r, c): ax.text(
            c - 0.38, r - 0.35, '', ha='left', va='top',
            fontsize=ANIMATION_V_FONT_SIZE, color='#1a6b1a', alpha=0.85, zorder=8)
        for r in range(GRID)
        for c in range(GRID)
    }

    initial_phase = phase_list[0] if phase_list else '?'
    initial_pstr = {'C': 'Copter exploring', 'R': 'Rover executing',
                    'W': 'Warmup', 'I': 'Init', 'E': 'End'}.get(
                        initial_phase, initial_phase)
    header_title = fig.suptitle(
        _animation_title(
            history,
            phase_text=initial_pstr,
            q_value=get_q_for_frame(0),
        ),
        fontsize=ANIMATION_HEADER_FONT_SIZE,
        fontfamily=PLOT_FONT_FAMILY,
        fontweight='bold',
        y=header_layout['suptitle_y'],
        va='top',
    )
    if metadata:
        fig.text(
            0.5,
            header_layout['suptitle_y'] - header_layout['subtitle_gap'],
            metadata,
            ha='center',
            va='top',
            fontsize=ANIMATION_META_FONT_SIZE,
            fontfamily=PLOT_FONT_FAMILY,
        )
    plt.tight_layout(rect=(0, 0, 1, header_layout['top_margin']))

    def update(i):
        beliefs_i = bh[i]
        obs_grid  = np.array([[beliefs_i[(r, c)]['O'] for c in range(GRID)]
                              for r in range(GRID)])
        im_obs.set_data(obs_grid)

        # V value labels.
        V_now = get_V_for_frame(i)
        q_now = get_q_for_frame(i)
        for r in range(GRID):
            for c in range(GRID):
                if V_now is not None:
                    v_val = V_now.get((r, c, q_now), 0.0)
                    v_texts[(r, c)].set_text(f'{v_val:.2f}')
                else:
                    v_texts[(r, c)].set_text('')

        # Agent positions and trails.
        ri = min(i, len(rover_path)  - 1)
        ci = min(i, len(copter_path) - 1)
        if ri > 0:
            tr = np.array(rover_path[:ri + 1])
            rover_trail.set_data(
                tr[:, 1] + ROVER_PATH_COL_OFFSET,
                tr[:, 0] + ROVER_PATH_ROW_OFFSET,
            )
        rover_dot.set_data(
            [rover_path[ri][1] + ROVER_PATH_COL_OFFSET],
            [rover_path[ri][0] + ROVER_PATH_ROW_OFFSET],
        )
        if ci > 0:
            tc = np.array(copter_path[:ci + 1])
            copter_trail.set_data(
                tc[:, 1] + COPTER_PATH_COL_OFFSET,
                tc[:, 0] + COPTER_PATH_ROW_OFFSET,
            )
        copter_dot.set_data(
            [copter_path[ci][1] + COPTER_PATH_COL_OFFSET],
            [copter_path[ci][0] + COPTER_PATH_ROW_OFFSET],
        )

        phase = phase_list[min(i, len(phase_list) - 1)] if phase_list else '?'
        pstr  = {'C': 'Copter exploring', 'R': 'Rover executing',
                 'W': 'Warmup', 'I': 'Init', 'E': 'End'}.get(phase, phase)
        header_title.set_text(
            _animation_title(history, phase_text=pstr, q_value=q_now)
        )
        return ([im_obs, rover_dot, copter_dot, rover_trail, copter_trail, header_title]
                + list(v_texts.values()))

    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    _save_animation(anim, filepath, fps)
    plt.close(fig)
