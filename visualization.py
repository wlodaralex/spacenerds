"""
visualization.py 
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter

from environment      import GRID, AP_LIST, TRUE_L
from planning import rover_value_iteration

# Colour scheme for each atomic proposition
AP_COLORS = {
    'O': 'red',
    'A': 'blue',
    'B': 'limegreen',
    'C': 'darkorchid',
    'D': 'darkorange',
}




def plot_belief_map(ax, beliefs: dict, ap: str,
                    rover_pos=None, copter_pos=None) -> None:
    """

    """
    grid = np.array([[beliefs[(r, c)][ap] for c in range(GRID)]
                     for r in range(GRID)])
    im = ax.imshow(grid, vmin=0, vmax=1, cmap='gray_r', origin='upper',
                   aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    for (r, c), labels in TRUE_L.items():
        if ap in labels:
            ax.text(c, r, ap, ha='center', va='center', fontsize=7,
                    color=AP_COLORS.get(ap, 'yellow'), fontweight='bold')

    if rover_pos is not None:
        ax.plot(rover_pos[1], rover_pos[0], 'o', color='deepskyblue',
                markersize=9, markeredgecolor='white', markeredgewidth=1.2)
    if copter_pos is not None:
        ax.plot(copter_pos[1], copter_pos[0], 'o', color='tomato',
                markersize=9, markeredgecolor='white', markeredgewidth=1.2)

    ax.set_title(f'Belief: {ap}', fontsize=9)
    ax.set_xticks(range(0, GRID, 2))
    ax.set_yticks(range(0, GRID, 2))
    ax.tick_params(labelsize=6)




def make_belief_snapshots_plot(history: dict, filepath: str) -> None:
    """

    """
    which = history.get('which_phi', '')
    if 'φ1' in str(which):
        APs, fig_cols = ['A', 'O'], 2
    elif 'φ2' in str(which):
        APs, fig_cols = ['B', 'C', 'O'], 3
    else:
        APs, fig_cols = ['C', 'D', 'O'], 3

    snaps   = history['beliefs_snapshot']
    snap_k  = history['snap_k']
    snap_r  = history['snap_rover']
    snap_c  = history['snap_copter']
    n_snaps = len(snaps)

    fig, axes = plt.subplots(n_snaps, fig_cols,
                             figsize=(4 * fig_cols, 3.8 * n_snaps))
    if n_snaps == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_snaps):
        for j, ap in enumerate(APs):
            ax  = axes[i, j]
            rp  = snap_r[i] if i < len(snap_r) else None
            cp  = snap_c[i] if i < len(snap_c) else None
            plot_belief_map(ax, snaps[i], ap,
                            rover_pos  = rp if ap != 'O' else None,
                            copter_pos = cp if ap == 'O' else None)
            if j == 0:
                ax.set_ylabel(f'k = {snap_k[i]}', fontsize=10, fontweight='bold')

    status = (f"Complete via {which}" if history['complete'] else
              ("Failed (obstacle)" if history['fail'] else "Timeout"))
    fig.suptitle(
        f'Environmental Belief Snapshots  —  {status}\n'
        '(Blue ● = rover  |  Red ● = copter  |  dark = high belief)',
        fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def make_trajectories_plot(history: dict, filepath: str,
                         use_substeps: bool = True) -> None:
    """

    """
    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(GRID + 1):
        ax.axhline(i - 0.5, color='#ddd', lw=0.6)
        ax.axvline(i - 0.5, color='#ddd', lw=0.6)
    for (r, c), labels in TRUE_L.items():
        for ap in labels:
            ax.text(c, r, ap, ha='center', va='center', fontsize=9,
                    color=AP_COLORS.get(ap, 'gray'), fontweight='bold')

    if use_substeps and 'rover_substeps' in history:
        rpath = np.array(history['rover_substeps'])
        cpath = np.array(history['copter_substeps'])
    else:
        rpath = np.array(history['rover_path'])
        cpath = np.array(history['copter_path'])

    ax.plot(rpath[:, 1], rpath[:, 0], '-', color='steelblue',
            lw=1.5, alpha=0.7, label='Rover path')
    ax.plot(cpath[:, 1], cpath[:, 0], '-', color='tomato',
            lw=1.2, alpha=0.6, label='Copter path')
    ax.plot(rpath[0, 1],  rpath[0, 0],  '*', color='steelblue',
            markersize=14, label='Rover start')
    ax.plot(rpath[-1, 1], rpath[-1, 0], '^', color='steelblue',
            markersize=10, label='Rover end')
    ax.plot(cpath[0, 1],  cpath[0, 0],  's', color='tomato',
            markersize=9,  label='Copter start')

    which  = history.get('which_phi', '')
    status = (f"Complete via {which}" if history['complete'] else
              ("Failed" if history['fail'] else "Timeout"))
    ax.set_title(f'Trajectories  (k={history["k_final"]})  —  {status}',
                 fontsize=11)
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(GRID - 0.5, -0.5)
    ax.set_xticks(range(GRID))
    ax.set_yticks(range(GRID))
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def make_final_beliefs_plot(history: dict, filepath: str) -> None:
    """Final belief maps for all 5 APs side by side."""
    beliefs = history['beliefs_snapshot'][-1]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, ap in enumerate(AP_LIST):
        plot_belief_map(axes[i], beliefs, ap,
                        rover_pos  = history['final_rover'],
                        copter_pos = history['final_copter'])
    fig.suptitle('Final Belief Maps (all APs)', fontsize=12)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def make_v_fn_heatmap(beliefs: dict, filepath: str, rover_pos: tuple, rover_q: int) -> None:
    """

    """
    V, _, _ = rover_value_iteration(beliefs, vi_steps=80)
    grid  = np.array([[V.get((r, c, rover_q), 0.0) for c in range(GRID)]
                      for r in range(GRID)])
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap='viridis', origin='upper',
                   aspect='equal')
    plt.colorbar(im, ax=ax, label='V (belief of mission completion)')
    for (r, c), labels in TRUE_L.items():
        for ap in labels:
            ax.text(c, r, ap, ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
    ax.plot(rover_pos[1], rover_pos[0], 'r*', markersize=14, label='Rover')
    ax.set_title(f'Value Function V(·, q={rover_q})', fontsize=11)
    ax.legend()
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def make_convergence_plot(history: dict, filepath: str,
                          obstacle_cells=None, free_cells=None) -> None:
    """

    """
    belief_history = history.get('belief_history')
    if belief_history is None:
        print("  make_convergence_plot: need store_belief_history=True")
        return

    k_list = history['k_list']
    n      = min(len(belief_history), len(k_list))
    ks     = list(k_list[:n])

    if obstacle_cells is None:
        obstacle_cells = [rc for rc, lbl in TRUE_L.items() if 'O' in lbl][:3]
    if free_cells is None:
        free_cells = [rc for rc, lbl in TRUE_L.items()
                      if not lbl and rc[0] < 5][:3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        'Belief Convergence  (Theorem 1)\n'
        'Obstacle cells — B(x|=O) → 1   |   Free cells — B(x|=O) → 0',
        fontsize=11)

    colors_obs  = ['#d62728', '#e377c2', '#ff7f0e']
    colors_free = ['#1f77b4', '#17becf', '#2ca02c']

    ax_obs, ax_free = axes
    for i, cell in enumerate(obstacle_cells):
        vals = [belief_history[t][cell]['O'] for t in range(n)]
        ax_obs.plot(ks, vals, color=colors_obs[i % 3], lw=1.8,
                    label=f'cell {cell}')
    ax_obs.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5,
                   label='target = 1')
    ax_obs.set_ylim(-0.05, 1.05)
    ax_obs.set_xlabel('Time step k', fontsize=10)
    ax_obs.set_ylabel('B(x |= O)', fontsize=10)
    ax_obs.set_title('Obstacle cells  (truth: O = True)', fontsize=10)
    ax_obs.legend(fontsize=8)
    ax_obs.grid(True, alpha=0.3)

    for i, cell in enumerate(free_cells):
        vals = [belief_history[t][cell]['O'] for t in range(n)]
        ax_free.plot(ks, vals, color=colors_free[i % 3], lw=1.8,
                     label=f'cell {cell}')
    ax_free.axhline(0.0, color='black', lw=0.8, ls='--', alpha=0.5,
                    label='target = 0')
    ax_free.set_ylim(-0.05, 1.05)
    ax_free.set_xlabel('Time step k', fontsize=10)
    ax_free.set_ylabel('B(x |= O)', fontsize=10)
    ax_free.set_title('Free cells  (truth: O = False)', fontsize=10)
    ax_free.legend(fontsize=8)
    ax_free.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")




def _subsample_for_cycles(history: dict, belief_history: list) -> tuple:
    """

    """
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





def make_beliefs_animation(history: dict, filepath: str,
                   fps: int = 4, use_substeps: bool = True) -> None:
    """
    Animate per-AP belief panels.

    APs shown depend on which mission branch completed:
      φ1 → [A, O]       φ2 → [B, C, O]       φ3 → [C, D, O]

    use_substeps=True  (default): one frame per timestep k — smooth.
    use_substeps=False: one frame per planning cycle — fast preview,
                        but last frame is guaranteed to show the true
                        final k and belief state (FIX 9).
    """
    which = history.get('which_phi', '')
    APs   = (['A', 'O']      if 'φ1' in str(which) else
             ['B', 'C', 'O'] if 'φ2' in str(which) else
             ['C', 'D', 'O'])

    belief_history = history.get('belief_history')
    if belief_history is None:
        print("  No belief history. Run with store_belief_history=True.")
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

    fig, axes = plt.subplots(1, len(APs), figsize=(4 * len(APs), 4.5))
    plt.subplots_adjust(wspace=0.35)

    def draw_frame(i):
        for ax, ap in zip(axes, APs):
            ax.clear()
            beliefs_i = bh[i]
            grid = np.array([[beliefs_i[(r, c)][ap] for c in range(GRID)]
                             for r in range(GRID)])
            ax.imshow(grid, vmin=0, vmax=1, cmap='gray_r',
                      origin='upper', aspect='equal')
            for (r, c), labels in TRUE_L.items():
                if ap in labels:
                    ax.text(c, r, ap, ha='center', va='center', fontsize=8,
                            color=AP_COLORS.get(ap, 'yellow'), fontweight='bold')
            rp = rover_path[min(i, len(rover_path) - 1)]
            cp = copter_path[min(i, len(copter_path) - 1)]
            if ap != 'O':
                ax.plot(rp[1], rp[0], 'o', color='deepskyblue',
                        markersize=9, markeredgecolor='white')
                trail = rover_path[:i + 1]
                if len(trail) > 1:
                    tr = np.array(trail)
                    ax.plot(tr[:, 1], tr[:, 0], '-', color='deepskyblue',
                            lw=1, alpha=0.5)
            if ap == 'O':
                ax.plot(cp[1], cp[0], 'o', color='tomato',
                        markersize=9, markeredgecolor='white')
                trail = copter_path[:i + 1]
                if len(trail) > 1:
                    tr = np.array(trail)
                    ax.plot(tr[:, 1], tr[:, 0], '-', color='tomato',
                            lw=1, alpha=0.5)
            ax.set_title(f'Belief: {ap}', fontsize=9)
            ax.set_xticks(range(0, GRID, 2))
            ax.set_yticks(range(0, GRID, 2))
            ax.tick_params(labelsize=6)

        k_val  = k_list[min(i, len(k_list) - 1)]
        phase  = phase_list[min(i, len(phase_list) - 1)] if phase_list else '?'
        pstr   = {'C': 'Copter exploring', 'R': 'Rover executing',
                  'W': 'Warmup', 'I': 'Init', 'E': 'End'}.get(phase, phase)
        fig.suptitle(f'k = {k_val}   [{pstr}]', fontsize=12, fontweight='bold')

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=200)
    _save_animation(anim, filepath, fps)
    plt.close(fig)


def make_unified_animation(history: dict, filepath: str,
                           fps: int = 4, use_substeps: bool = True) -> None:
    """

    """
    belief_history = history.get('belief_history')
    if belief_history is None:
        print("  Need store_belief_history=True")
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

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(top=0.88)

    for i in range(GRID + 1):
        ax.axhline(i - 0.5, color='#aaa', lw=0.4, zorder=1)
        ax.axvline(i - 0.5, color='#aaa', lw=0.4, zorder=1)
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(GRID - 0.5, -0.5)
    ax.set_xticks(range(GRID))
    ax.set_yticks(range(GRID))

    im_obs = ax.imshow(
        np.full((GRID, GRID), 0.5), vmin=0, vmax=1,
        cmap='Greys', origin='upper', aspect='equal',
        alpha=0.6, zorder=3, interpolation='nearest')

    rover_dot,    = ax.plot([], [], 'o', color='deepskyblue', ms=11,
                            markeredgecolor='white', markeredgewidth=1.5, zorder=6)
    copter_dot,   = ax.plot([], [], 'o', color='tomato', ms=11,
                            markeredgecolor='white', markeredgewidth=1.5, zorder=6)
    rover_trail,  = ax.plot([], [], '-', color='deepskyblue', lw=1.5,
                            alpha=0.6, zorder=5)
    copter_trail, = ax.plot([], [], '-', color='tomato', lw=1.2,
                            alpha=0.5, zorder=5)


    revealed_texts = {
        (r, c, ap): ax.text(
            c, r, ap, ha='center', va='center', fontsize=9,
            color=AP_COLORS.get(ap, 'gray'), fontweight='bold',
            alpha=0.0, zorder=7)
        for r in range(GRID)
        for c in range(GRID)
        for ap in AP_LIST
    }

    title = ax.set_title('k = 0', fontsize=13, fontweight='bold', pad=10)
    ax.legend(handles=[
        mpatches.Patch(color='deepskyblue', label='Rover'),
        mpatches.Patch(color='tomato',      label='Copter'),
        mpatches.Patch(color='#555',        label='High obstacle belief'),
    ], loc='lower right', fontsize=8)

    def update(i):
        beliefs_i = bh[i]

        obs_grid = np.array([[beliefs_i[(r, c)]['O'] for c in range(GRID)]
                             for r in range(GRID)])
        im_obs.set_data(obs_grid)

        for r in range(GRID):
            for c in range(GRID):
                for ap in AP_LIST:
                    b = beliefs_i[(r, c)][ap]

                    alpha = max(0.0, min(1.0, (b - 0.55) * 6))
                    revealed_texts[(r, c, ap)].set_alpha(alpha)

        ri = min(i, len(rover_path)  - 1)
        ci = min(i, len(copter_path) - 1)
        if ri > 0:
            tr = np.array(rover_path[:ri + 1])
            rover_trail.set_data(tr[:, 1], tr[:, 0])
        rover_dot.set_data([rover_path[ri][1]], [rover_path[ri][0]])
        if ci > 0:
            tc = np.array(copter_path[:ci + 1])
            copter_trail.set_data(tc[:, 1], tc[:, 0])
        copter_dot.set_data([copter_path[ci][1]], [copter_path[ci][0]])

        k_val = k_list[min(i, len(k_list) - 1)]
        phase = phase_list[min(i, len(phase_list) - 1)] if phase_list else '?'
        pstr  = {'C': 'Copter exploring', 'R': 'Rover executing',
                 'W': 'Warmup', 'I': 'Init', 'E': 'End'}.get(phase, phase)
        title.set_text(f'k = {k_val}   [{pstr}]')
        return ([im_obs, rover_dot, copter_dot, rover_trail, copter_trail, title]
                + list(revealed_texts.values()))

    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    _save_animation(anim, filepath, fps)
    plt.close(fig)




def _save_animation(anim, filepath: str, fps: int) -> None:
    """
    
    """
    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(filepath, writer=writer)
        print(f"  Saved animation: {filepath}")
    except Exception as e:
        gif_path = filepath.replace('.mp4', '.gif')
        print(f"  ffmpeg failed ({e}), saving as gif: {gif_path}")
        anim.save(gif_path, writer='pillow', fps=fps)