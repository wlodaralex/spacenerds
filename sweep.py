"""
sweep.py - Parameter sweep for the game-theoretic rover-copter project.

Phase 0:
    - Evaluate five scenarios:
      1. baseline_local
      2. baseline_global
      3. game_copter_private
      4. game_rover_private
      5. game_full
    - Tune gamma and/or lambda only for the three game scenarios.

Phase 1:
    - Paired comparison at the tuned optima from Phase 0 on fresh starts.

Selection logic:
    - Baseline local/global are fixed.
    - Game copter private is ranked by completion rate, then median
      successful k, then runtime.
    - Game rover private and game full use a composite ranking:
      configs within 2 runs of the best completion count are compared by
      median successful k.

Output:
    - outputs/sweep_runs.csv
    - outputs/sweep_meta.json

Usage:
    - python sweep.py
    - python sweep.py --phase 0
    - python sweep.py --phase 1
    - python sweep.py --quick
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from environment import GRID, TRUE_L
from scenario import SimulationScenario
from simulation import run_simulation


# --- Configuration ---

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
RUNS_CSV = OUTPUT_DIR / 'sweep_runs.csv'
META_JSON = OUTPUT_DIR / 'sweep_meta.json'

GAMMAS_TUNE = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
LAMBDAS_TUNE = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
N_SEEDS_TUNE = 30
N_STARTS_PAIRED = 100

MAX_K = 400
META_SEED = 0
FIXED = dict(T_c=5, T_r=3, alpha=1.5, vi_steps=80)

RATE_TIE_MARGIN_RUNS = 2


# --- CSV Schema ---

FIELDNAMES = [
    'run_id', 'phase', 'scenario',
    'gamma', 'lambda', 'seed', 'copter_mode',
    'rover_start_r', 'rover_start_c',
    'copter_start_r', 'copter_start_c',
    'max_k', 'planner',
    'complete', 'fail', 'k_final', 'which_phi', 'epochs',
    'avg_expanded',
    'avg_Phi', 'avg_G', 'avg_C_r', 'avg_C_c', 'avg_I', 'avg_u_r', 'avg_u_c',
    'final_Phi', 'final_G',
    'rover_path_length', 'copter_path_length',
    'runtime_s',
]


# --- Shared Helpers ---

def _safe_mean(values):
    """Return the arithmetic mean or NaN for an empty list."""
    return sum(values) / len(values) if values else float('nan')


def _parse_bool(value):
    """Parse CSV bool-like values written by DictWriter."""
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def _generate_starts(n, meta_seed=META_SEED):
    """Generate obstacle-free rover and copter start pairs."""
    rng = np.random.RandomState(meta_seed)
    free_cells = [
        (r, c)
        for r in range(GRID)
        for c in range(GRID)
        if 'O' not in TRUE_L[(r, c)]
    ]
    starts = []
    for _ in range(n):
        rover_start = free_cells[rng.randint(len(free_cells))]
        copter_start = free_cells[rng.randint(len(free_cells))]
        starts.append((rover_start, copter_start))
    return starts


def _run_quiet(scenario_name, rover_start, copter_start, gamma, lambda_, copter_mode,
               seed, max_k=MAX_K):
    """Run one simulation with stdout suppressed."""
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
        start_time = time.perf_counter()
        scenario = SimulationScenario(
            name         = f'{scenario_name}_seed_{seed}',
            rover_start  = rover_start,
            copter_start = copter_start,
            T_c          = FIXED['T_c'],
            T_r          = FIXED['T_r'],
            alpha        = FIXED['alpha'],
            gamma        = gamma,
            lambda_      = lambda_,
            rho          = 1.0,
            tau_r        = 0.9,
            vi_steps     = FIXED['vi_steps'],
            D_c          = float('inf'),
            copter_mode  = copter_mode,
            seed         = seed,
            max_k        = max_k,
        )
        history = run_simulation(scenario = scenario)
        runtime_s = time.perf_counter() - start_time
    finally:
        sys.stdout = old_stdout
        devnull.close()
    return history, runtime_s


def _extract_row(history, runtime_s, *, run_id, phase, scenario,
                 gamma, lambda_, seed, copter_mode,
                 rover_start, copter_start, max_k):
    """Build one CSV row from a SimHistory."""
    epochs = history.epochs
    return {
        'run_id': run_id,
        'phase': phase,
        'scenario': scenario,
        'gamma': gamma,
        'lambda': lambda_,
        'seed': seed,
        'copter_mode': copter_mode,
        'rover_start_r': rover_start[0],
        'rover_start_c': rover_start[1],
        'copter_start_r': copter_start[0],
        'copter_start_c': copter_start[1],
        'max_k': max_k,
        'planner': history.meta.get('planner_name', ''),
        'complete': history.complete,
        'fail': history.fail,
        'k_final': history.k_final,
        'which_phi': history.which_phi or '',
        'epochs': len(epochs),
        'avg_expanded': _safe_mean([epoch.n_expanded for epoch in epochs]),
        'avg_Phi': _safe_mean([epoch.Phi for epoch in epochs]),
        'avg_G': _safe_mean([epoch.G for epoch in epochs]),
        'avg_C_r': _safe_mean([epoch.C_r for epoch in epochs]),
        'avg_C_c': _safe_mean([epoch.C_c for epoch in epochs]),
        'avg_I': _safe_mean([epoch.I for epoch in epochs]),
        'avg_u_r': _safe_mean([epoch.u_r for epoch in epochs]),
        'avg_u_c': _safe_mean([epoch.u_c for epoch in epochs]),
        'final_Phi': epochs[-1].Phi if epochs else float('nan'),
        'final_G': epochs[-1].G if epochs else float('nan'),
        'rover_path_length': history.rover_path_length(),
        'copter_path_length': history.copter_path_length(),
        'runtime_s': runtime_s,
    }


def _open_csv(path):
    """Open the sweep CSV for append and write a header when new."""
    exists = path.exists() and path.stat().st_size > 0
    file_handle = open(path, 'a', newline='')
    writer = csv.DictWriter(file_handle, fieldnames=FIELDNAMES)
    if not exists:
        writer.writeheader()
    return file_handle, writer


def _status_char(history):
    """Return a compact run status marker."""
    if history.complete:
        return '✓'
    if history.fail:
        return '✗'
    return '·'


def _group_stats(rows, key_fn):
    """Aggregate completion, speed, and runtime stats by config key."""
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[key_fn(row)].append(row)

    stats_by_key = {}
    for key, group_rows in grouped_rows.items():
        completion_flags = [_parse_bool(row['complete']) for row in group_rows]
        success_k = [
            float(row['k_final'])
            for row in group_rows
            if _parse_bool(row['complete'])
        ]
        runtimes = [float(row['runtime_s']) for row in group_rows]
        n_runs = len(group_rows)
        n_complete = sum(completion_flags)
        stats_by_key[key] = {
            'n': n_runs,
            'n_complete': n_complete,
            'completion_rate': n_complete / n_runs,
            'median_success_k': (
                float(np.median(success_k))
                if success_k else float('inf')
            ),
            'avg_runtime_s': _safe_mean(runtimes),
        }
    return stats_by_key


def _pick_best_simple(stats):
    """Pick by completion rate, then median successful k, then runtime."""
    ranked = sorted(
        stats.items(),
        key=lambda item: (
            -item[1]['completion_rate'],
            item[1]['median_success_k'],
            item[1]['avg_runtime_s'],
        ),
    )
    return ranked[0][0] if ranked else None


def _pick_best_composite(stats, tie_margin_runs=RATE_TIE_MARGIN_RUNS):
    """Pick by completion count, then prefer faster configs within the top tier."""
    if not stats:
        return None
    ranked = sorted(
        stats.items(),
        key=lambda item: -item[1]['n_complete'],
    )
    best_n_complete = ranked[0][1]['n_complete']
    top_tier = [
        (key, summary)
        for key, summary in ranked
        if best_n_complete - summary['n_complete'] <= tie_margin_runs
    ]
    top_tier.sort(
        key=lambda item: (
            item[1]['median_success_k'],
            -item[1]['n_complete'],
            item[1]['avg_runtime_s'],
        ),
    )
    return top_tier[0][0]


# --- Phase 0 ---

def phase0(csv_path, gammas, lambdas, n_seeds, max_k):
    """Run the Phase 0 sweep and write results incrementally."""
    starts = _generate_starts(max(n_seeds, 100))
    file_handle, writer = _open_csv(csv_path)

    def _run_and_write(scenario, gamma, lambda_, mode, run_id_prefix,
                       rover_start, copter_start, seed):
        history, runtime_s = _run_quiet(
            scenario,
            rover_start, copter_start,
            gamma, lambda_, mode, seed, max_k,
        )
        writer.writerow(_extract_row(
            history, runtime_s,
            run_id       = f"{run_id_prefix}_s{seed:03d}",
            phase        = 'phase0',
            scenario     = scenario,
            gamma        = gamma,
            lambda_      = lambda_,
            seed         = seed,
            copter_mode  = mode,
            rover_start  = rover_start,
            copter_start = copter_start,
            max_k        = max_k,
        ))
        file_handle.flush()
        return history, runtime_s

    try:
        for copter_mode in ('local', 'global'):
            print(f"\n{'=' * 60}")
            print(f"  Phase 0 - Baseline ({copter_mode})  [{n_seeds} seeds]")
            print(f"{'=' * 60}")
            for start_index in range(n_seeds):
                rover_start, copter_start = starts[start_index]
                seed = META_SEED + start_index + 1
                history, runtime_s = _run_and_write(
                    f'baseline_{copter_mode}',
                    0.0, 0.0, copter_mode, f'bl_{copter_mode}',
                    rover_start, copter_start, seed,
                )
                print(
                    f"  {copter_mode:6s} seed={seed:3d}  "
                    f"k={history.k_final:4d} {_status_char(history)}  "
                    f"{runtime_s:.1f}s"
                )

        print(f"\n{'=' * 60}")
        print(f"  Phase 0 - Game rover private  gamma x {n_seeds} seeds")
        print(f"{'=' * 60}")
        for gamma in gammas:
            if gamma == 0:
                continue
            complete_count = 0
            for start_index in range(n_seeds):
                rover_start, copter_start = starts[start_index]
                seed = META_SEED + start_index + 1
                history, _ = _run_and_write(
                    'game_rover_private',
                    gamma, 0.0, 'utility', f'grp_g{gamma}',
                    rover_start, copter_start, seed,
                )
                complete_count += int(history.complete)
            print(f"  gamma={gamma:<5.2f}  {complete_count}/{n_seeds} complete")

        print(f"\n{'=' * 60}")
        print(f"  Phase 0 - Game copter private  lambda x {n_seeds} seeds")
        print(f"{'=' * 60}")
        for lambda_ in lambdas:
            if lambda_ == 0:
                continue
            complete_count = 0
            for start_index in range(n_seeds):
                rover_start, copter_start = starts[start_index]
                seed = META_SEED + start_index + 1
                history, _ = _run_and_write(
                    'game_copter_private',
                    0.0, lambda_, 'utility', f'gcp_l{lambda_}',
                    rover_start, copter_start, seed,
                )
                complete_count += int(history.complete)
            print(f"  lambda={lambda_:<5.2f}  {complete_count}/{n_seeds} complete")

        grid_gammas = [gamma for gamma in gammas if gamma > 0]
        grid_lambdas = [lambda_ for lambda_ in lambdas if lambda_ > 0]
        n_configs = len(grid_gammas) * len(grid_lambdas)
        config_index = 0
        print(f"\n{'=' * 60}")
        print(f"  Phase 0 - Game full  {n_configs} configs x {n_seeds} seeds")
        print(f"{'=' * 60}")
        for gamma in grid_gammas:
            for lambda_ in grid_lambdas:
                config_index += 1
                complete_count = 0
                for start_index in range(n_seeds):
                    rover_start, copter_start = starts[start_index]
                    seed = META_SEED + start_index + 1
                    history, _ = _run_and_write(
                        'game_full',
                        gamma, lambda_, 'utility',
                        f'gf_g{gamma}_l{lambda_}',
                        rover_start, copter_start, seed,
                    )
                    complete_count += int(history.complete)
                print(
                    f"  [{config_index:3d}/{n_configs}] "
                    f"gamma={gamma:<5.2f} lambda={lambda_:<5.2f}  "
                    f"{complete_count}/{n_seeds} complete"
                )
    finally:
        file_handle.close()

    print(f"\nPhase 0 done. Results in {csv_path}")


# --- Phase 1 Parameter Selection ---

def _find_best_params(csv_path):
    """Select tuned parameters from Phase 0 rows in the sweep CSV."""
    with open(csv_path, newline='') as file_handle:
        rows = [
            row
            for row in csv.DictReader(file_handle)
            if row.get('phase') == 'phase0'
        ]
    if not rows:
        raise ValueError("No Phase 0 rows found.")

    best = {}

    baseline_local_rows = [
        row for row in rows
        if row['scenario'].strip() == 'baseline_local'
    ]
    if baseline_local_rows:
        baseline_local_stats = _group_stats(
            baseline_local_rows,
            key_fn=lambda row: 'baseline_local',
        )
        summary = baseline_local_stats['baseline_local']
        best['baseline_local'] = dict(
            gamma       = 0.0,
            lambda_     = 0.0,
            copter_mode = 'local',
        )
        print(
            f"  baseline_local     -> local    "
            f"({summary['completion_rate']:.0%}, "
            f"med_k={summary['median_success_k']:.0f})"
        )

    baseline_global_rows = [
        row for row in rows
        if row['scenario'].strip() == 'baseline_global'
    ]
    if baseline_global_rows:
        baseline_global_stats = _group_stats(
            baseline_global_rows,
            key_fn=lambda row: 'baseline_global',
        )
        summary = baseline_global_stats['baseline_global']
        best['baseline_global'] = dict(
            gamma       = 0.0,
            lambda_     = 0.0,
            copter_mode = 'global',
        )
        print(
            f"  baseline_global    -> global   "
            f"({summary['completion_rate']:.0%}, "
            f"med_k={summary['median_success_k']:.0f})"
        )

    rover_private_rows = [
        row for row in rows
        if row['scenario'].strip() == 'game_rover_private'
    ]
    if rover_private_rows:
        rover_private_stats = _group_stats(
            rover_private_rows,
            key_fn=lambda row: float(row['gamma']),
        )
        best_gamma = _pick_best_composite(rover_private_stats)
        summary = rover_private_stats[best_gamma]
        best['game_rover_private'] = dict(
            gamma       = best_gamma,
            lambda_     = 0.0,
            copter_mode = 'utility',
        )
        print(
            f"  game_rover_private -> gamma={best_gamma:<5.2f}  "
            f"({summary['completion_rate']:.0%}, "
            f"med_k={summary['median_success_k']:.0f})"
        )

    copter_private_rows = [
        row for row in rows
        if row['scenario'].strip() == 'game_copter_private'
    ]
    if copter_private_rows:
        copter_private_stats = _group_stats(
            copter_private_rows,
            key_fn=lambda row: float(row['lambda']),
        )
        best_lambda = _pick_best_simple(copter_private_stats)
        summary = copter_private_stats[best_lambda]
        best['game_copter_private'] = dict(
            gamma       = 0.0,
            lambda_     = best_lambda,
            copter_mode = 'utility',
        )
        print(
            f"  game_copter_private -> lambda={best_lambda:<5.2f}  "
            f"({summary['completion_rate']:.0%}, "
            f"med_k={summary['median_success_k']:.0f})"
        )

    game_full_rows = [
        row for row in rows
        if row['scenario'].strip() == 'game_full'
    ]
    if game_full_rows:
        game_full_stats = _group_stats(
            game_full_rows,
            key_fn=lambda row: (
                float(row['gamma']),
                float(row['lambda']),
            ),
        )
        best_gamma, best_lambda = _pick_best_composite(game_full_stats)
        summary = game_full_stats[(best_gamma, best_lambda)]
        best['game_full'] = dict(
            gamma       = best_gamma,
            lambda_     = best_lambda,
            copter_mode = 'utility',
        )
        print(
            f"  game_full          -> gamma={best_gamma:<5.2f} "
            f"lambda={best_lambda:<5.2f}  "
            f"({summary['completion_rate']:.0%}, "
            f"med_k={summary['median_success_k']:.0f})"
        )

    return best


# --- Phase 1 ---

def phase1(csv_path, n_starts, max_k):
    """Run the paired comparison using tuned Phase 0 parameters."""
    print(f"\n{'=' * 60}")
    print("  Phase 1 - Tuned param selection")
    print(f"{'=' * 60}")
    best = _find_best_params(csv_path)

    active_configs = {
        scenario_name: params
        for scenario_name, params in best.items()
        if params is not None
    }
    starts = _generate_starts(n_starts, meta_seed=42)

    print(f"\n{'=' * 60}")
    print(
        f"  Phase 1 - Paired comparison  "
        f"[{n_starts} starts x {len(active_configs)} configs]"
    )
    print(f"{'=' * 60}")

    file_handle, writer = _open_csv(csv_path)
    try:
        for start_index, (rover_start, copter_start) in enumerate(starts):
            seed = 1000 + start_index
            line = (
                f"  {start_index + 1:3d}/{n_starts}  "
                f"rover={rover_start} copter={copter_start} "
            )
            for scenario_name, params in active_configs.items():
                history, runtime_s = _run_quiet(
                    scenario_name,
                    rover_start, copter_start,
                    params['gamma'], params['lambda_'], params['copter_mode'],
                    seed, max_k,
                )
                writer.writerow(_extract_row(
                    history, runtime_s,
                    run_id       = f"p1_{scenario_name}_s{seed:04d}",
                    phase        = 'phase1',
                    scenario     = scenario_name,
                    gamma        = params['gamma'],
                    lambda_      = params['lambda_'],
                    seed         = seed,
                    copter_mode  = params['copter_mode'],
                    rover_start  = rover_start,
                    copter_start = copter_start,
                    max_k        = max_k,
                ))
                file_handle.flush()
                short_name = (
                    scenario_name
                    .replace('baseline_', 'bl_')
                    .replace('game_', 'g_')
                )
                line += f" {short_name[:12]:12s}={_status_char(history)}"
            print(line)
    finally:
        file_handle.close()

    print(f"\nPhase 1 done. Results appended to {csv_path}")


# --- Metadata ---

def _save_meta(csv_path, gammas, lambdas, n_seeds, n_starts, max_k):
    """Write one metadata snapshot for the sweep."""
    metadata = {
        'experiment': 'AER1516 Space Nerds parameter sweep v2',
        'date': time.strftime('%Y-%m-%d'),
        'adjustments': [
            (
                'game_rover_private and game_full composite ranking: within '
                f'{RATE_TIE_MARGIN_RUNS} runs of the best completion count '
                'prefer faster median k'
            ),
            'phase0 scenarios: baseline_local, baseline_global, '
            'game_copter_private, game_rover_private, game_full',
        ],
        'phase0': {
            'gammas': gammas,
            'lambdas': lambdas,
            'seeds_per_config': n_seeds,
            'max_k': max_k,
            'start_generation': (
                f'random free-space, shared, meta_seed={META_SEED}'
            ),
        },
        'phase1': {
            'n_starts': n_starts,
            'max_k': max_k,
            'start_generation': 'random free-space, meta_seed=42',
            'selection': {
                'tie_margin_runs': RATE_TIE_MARGIN_RUNS,
            },
        },
        'fixed_params': FIXED,
        'priors': {'target': 0.1, 'obstacle': 0.3},
        'csv': str(csv_path),
    }
    with open(META_JSON, 'w') as file_handle:
        json.dump(metadata, file_handle, indent=2)
    print(f"Meta saved to {META_JSON}")


# --- CLI ---

def main():
    """Run the requested sweep phases."""
    parser = argparse.ArgumentParser(description='Parameter sweep v2')
    parser.add_argument('--phase', type=int, default=None)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.quick:
        gammas = [0, 0.5, 2.0, 5.0]
        lambdas = [0, 0.5, 2.0, 5.0]
        n_seeds = 5
        n_starts = 10
        max_k = 200
    else:
        gammas = GAMMAS_TUNE
        lambdas = LAMBDAS_TUNE
        n_seeds = N_SEEDS_TUNE
        n_starts = N_STARTS_PAIRED
        max_k = MAX_K

    if args.phase is None and RUNS_CSV.exists():
        backup_path = RUNS_CSV.with_suffix('.csv.bak')
        RUNS_CSV.rename(backup_path)
        print(f"Backed up previous results to {backup_path}")

    if args.phase is None or args.phase == 0:
        phase0(RUNS_CSV, gammas, lambdas, n_seeds, max_k)

    if args.phase is None or args.phase == 1:
        phase1(RUNS_CSV, n_starts, max_k)

    _save_meta(RUNS_CSV, gammas, lambdas, n_seeds, n_starts, max_k)


if __name__ == '__main__':
    main()
