"""
exp1_baseline.py — Baseline comparison (Table 1).

6 configs × N random starts, parallel execution.
Same starts across all configs for fair comparison.

Usage:
    python exp1_baseline.py           # full run
    python exp1_baseline.py -j 14     # set workers
    python exp1_baseline.py --n-seeds 1000
    python exp1_baseline.py --dry-run # show plan
"""

import argparse, csv, os, sys, time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Configs ─────────────────────────────────────────────

CONFIGS = [
    # (name,            copter_mode,    gamma,  lambda)
    ('baseline_local',  'local',        0,      0),
    ('baseline_global', 'global',       0,      0),
    ('full_game',       'utility',      2,      7),   
    ('copter_only',     'utility',      0,      7),   
    ('rover_only',      'utility',      2,      0),
    ('altruistic',      'utility',      0,      0),
]

# ─── Fixed params ────────────────────────────────────────

N_SEEDS    = 100
MAX_K      = 300
META_SEED  = 0
TAU_R      = 0.5
D_C        = 30.0
N_WORKERS  = min(14, cpu_count() - 2)
CHECKPOINT = 50

FIXED = dict(
    T_c=5, T_r=3, alpha=1.5, rho=1.0, vi_steps=80,
    copter_top_k=5, copter_mc_samples=3,
)

# ─── Output ──────────────────────────────────────────────

CSV_PATH = ROOT / 'outputs' / 'exp1_baseline.csv'

FIELDS = [
    'config', 'copter_mode', 'gamma', 'lambda', 'tau_r', 'D_c', 'seed',
    'rover_start_r', 'rover_start_c',
    'copter_start_r', 'copter_start_c',
    'max_k',
    'complete', 'fail', 'k_final', 'which_phi', 'epochs',
    'avg_expanded',
    'avg_Phi', 'avg_G', 'avg_C_r', 'avg_C_c', 'avg_I',
    'avg_u_r', 'avg_u_c',
    'final_Phi', 'final_G',
    'rover_path_length', 'copter_path_length',
    'runtime_s',
]


# ─── Starts ──────────────────────────────────────────────

def _generate_starts(n):
    from environment import GRID, TRUE_L
    rng = np.random.RandomState(META_SEED)
    free = [(r, c) for r in range(GRID) for c in range(GRID)
            if 'O' not in TRUE_L[(r, c)]
            and all(ap not in TRUE_L[(r, c)] for ap in ['A','B','C','D'])]
    return [(free[rng.randint(len(free))], free[rng.randint(len(free))])
            for _ in range(n)]

# ─── Worker ──────────────────────────────────────────────

def _run_job(job):
    config_name, copter_mode, gamma, lambda_, seed, rover_start, copter_start = job

    devnull = open(os.devnull, 'w')
    old = sys.stdout
    sys.stdout = devnull
    try:
        from scenario import SimulationScenario
        from simulation import run_simulation

        t0 = time.perf_counter()
        sc = SimulationScenario(
            name=f'{config_name}_s{seed}',
            rover_start=rover_start, copter_start=copter_start,
            copter_mode=copter_mode,
            gamma=gamma, lambda_=lambda_,
            tau_r=TAU_R, D_c=D_C,
            seed=seed, max_k=MAX_K,
            T_c=FIXED['T_c'], T_r=FIXED['T_r'],
            alpha=FIXED['alpha'], rho=FIXED['rho'],
            vi_steps=FIXED['vi_steps'],
            copter_top_k=FIXED['copter_top_k'],
            copter_mc_samples=FIXED['copter_mc_samples'],
        )
        hist = run_simulation(scenario=sc)
        dt = time.perf_counter() - t0
    finally:
        sys.stdout = old
        devnull.close()

    eps = hist.epochs
    _m = lambda vals: sum(vals)/len(vals) if vals else float('nan')

    return {
        'config': config_name,
        'copter_mode': copter_mode,
        'gamma': gamma, 'lambda': lambda_,
        'tau_r': TAU_R, 'D_c': D_C,
        'seed': seed,
        'rover_start_r': rover_start[0],
        'rover_start_c': rover_start[1],
        'copter_start_r': copter_start[0],
        'copter_start_c': copter_start[1],
        'max_k': MAX_K,
        'complete': hist.complete,
        'fail': hist.fail,
        'k_final': hist.k_final,
        'which_phi': hist.which_phi or '',
        'epochs': len(eps),
        'avg_expanded': _m([e.n_expanded for e in eps]),
        'avg_Phi': _m([e.Phi for e in eps]),
        'avg_G': _m([e.G for e in eps]),
        'avg_C_r': _m([e.C_r for e in eps]),
        'avg_C_c': _m([e.C_c for e in eps]),
        'avg_I': _m([e.I for e in eps]),
        'avg_u_r': _m([e.u_r for e in eps]),
        'avg_u_c': _m([e.u_c for e in eps]),
        'final_Phi': eps[-1].Phi if eps else float('nan'),
        'final_G': eps[-1].G if eps else float('nan'),
        'rover_path_length': hist.rover_path_length(),
        'copter_path_length': hist.copter_path_length(),
        'runtime_s': dt,
    }


# ─── CSV I/O ─────────────────────────────────────────────

def _canon_key(config, seed):
    return f'{config}|{int(float(seed))}'

def _load_csv():
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return []
    with open(CSV_PATH, 'r', newline='') as f:
        return list(csv.DictReader(f))

def _save_csv(rows):
    rows.sort(key=lambda r: (r['config'], int(float(r['seed']))))
    CSV_PATH.parent.mkdir(exist_ok=True)
    with open(CSV_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


# ─── Main ────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-j', '--workers', type=int, default=N_WORKERS)
    p.add_argument('--n-seeds', type=int, default=N_SEEDS)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    all_starts = _generate_starts(args.n_seeds)
    seeds = [META_SEED + i + 1 for i in range(args.n_seeds)]

    # Build jobs
    all_jobs = []
    for name, mode, g, l in CONFIGS:
        for si, s in enumerate(seeds):
            rover_start, copter_start = all_starts[si]
            all_jobs.append((name, mode, g, l, s, rover_start, copter_start))

    # Dedup
    existing = _load_csv()
    done = {_canon_key(r['config'], r['seed']) for r in existing}
    jobs = [j for j in all_jobs if _canon_key(j[0], j[4]) not in done]

    total = len(all_jobs)
    n_skip = total - len(jobs)

    print(f"Exp 1: Baseline Comparison")
    print(f"  {len(CONFIGS)} configs × {args.n_seeds} seeds = {total} runs")
    print(f"  Already done: {n_skip}")
    print(f"  To run: {len(jobs)}")
    print(f"  Workers: {args.workers}")
    print(f"  τ_r={TAU_R}, D_c={D_C}, max_k={MAX_K}")
    est = len(jobs) * 2.0 / args.workers
    print(f"  Est: {est/60:.0f}min")

    if args.dry_run or not jobs:
        return

    print()
    completed = list(existing)
    n_done = n_comp = n_fail = 0
    t0 = time.perf_counter()

    with Pool(processes=args.workers) as pool:
        for row in pool.imap_unordered(_run_job, jobs, chunksize=4):
            completed.append(row)
            n_done += 1
            if row['complete']: n_comp += 1
            if row['fail']:     n_fail += 1

            elapsed = time.perf_counter() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(jobs) - n_done) / rate if rate > 0 else 0

            if n_done % CHECKPOINT == 0 or n_done == len(jobs):
                print(f"  [{n_done:>4d}/{len(jobs)}]  "
                      f"comp={n_comp}  fail={n_fail}  "
                      f"{rate:.1f}/s  ETA {eta/60:.0f}min")
                _save_csv(completed)

    _save_csv(completed)
    elapsed = time.perf_counter() - t0
    print(f"\nDone: {n_done} in {elapsed/60:.1f}min ({n_done/elapsed:.1f}/s)")
    print(f"CSV: {CSV_PATH} ({len(completed)} rows)")

    # Print summary table
    print(f"\n{'Config':<20} {'Comp%':>6} {'Fail%':>6} {'TO%':>6} {'Med k':>6}")
    print("-" * 50)
    for name, _, _, _ in CONFIGS:
        rows = [r for r in completed if r['config'] == name]
        n = len(rows)
        nc = sum(1 for r in rows if str(r['complete']) == 'True')
        nf = sum(1 for r in rows if str(r['fail']) == 'True')
        nt = n - nc - nf
        ks = [int(r['k_final']) for r in rows if str(r['complete']) == 'True']
        mk = sorted(ks)[len(ks)//2] if ks else -1
        print(f"{name:<20} {nc/n*100:5.0f}% {nf/n*100:5.0f}% "
              f"{nt/n*100:5.0f}% {mk:>6d}")


if __name__ == '__main__':
    main()
