"""
exp2_sweep.py — Adaptive multi-phase parameter sweep.

Phase 1: Coarse grid (fast exploration)
Phase 2: Refine around top-performing configs
Phase 3: Fine-tune best region

Loads existing CSV, skips completed runs, saves every 50 results.

Usage:
    python exp2_sweep.py                  # run all phases
    python exp2_sweep.py --phase 1        # coarse only
    python exp2_sweep.py --phase 2        # refine only (needs phase 1 data)
    python exp2_sweep.py --phase 3        # fine-tune (needs phase 2 data)
    python exp2_sweep.py --dry-run        # show plan
    python exp2_sweep.py -j 14            # set workers
"""

import argparse, csv, itertools, os, sys, time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Phase grids ──────────────────────────────────────────────

PHASE1 = dict(
    gammas  = [0, 1, 2, 3, 5, 8, 10],
    lambdas = [0, 1, 2, 3, 5, 8, 10],
    tau_rs  = [0.3, 0.5, 0.6, 0.8],
    d_cs    = [5, 10, 15, 30, float('inf')],
)

PHASE2_REFINE_RADIUS = dict(
    gamma  = [0, 1, -1],      # offsets from best
    lambda_ = [0, 1, -1, 2, -2],
)
PHASE2_TOP_N = 20             # refine around top N configs
PHASE2_COMP_THRESHOLD = 0.5   # only refine configs with comp > 50%

PHASE3 = dict(
    # fine grid around expected sweet spot
    gammas  = [1, 2, 3, 4, 5, 6],
    lambdas = [3, 4, 5, 6, 7, 8, 9, 10],
    tau_rs  = [0.5, 0.6],
    d_cs    = [15, 30],
)

# ─── Run settings ─────────────────────────────────────────────
N_SEEDS    = 30
MAX_K      = 300       
META_SEED  = 0
N_WORKERS  = min(14, cpu_count() - 2)
CHECKPOINT = 50        # save every 50 results

FIXED = dict(
    T_c=5, T_r=3, alpha=1.5, rho=1.0, vi_steps=80,
    copter_top_k=5, copter_mc_samples=3,
)

# ─── Output ───────────────────────────────────────────────────
CSV_PATH = ROOT / 'outputs' / 'exp2_sweep.csv'

FIELDS = [
    'gamma', 'lambda', 'tau_r', 'D_c', 'seed',
    'copter_mode',
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

SORT_KEYS = ['gamma', 'lambda', 'tau_r', 'D_c', 'seed']


# ─── Starts ───────────────────────────────────────────────────

def _generate_starts(n):
    from environment import GRID, TRUE_L
    rng = np.random.RandomState(META_SEED)
    free = [(r, c) for r in range(GRID) for c in range(GRID)
            if 'O' not in TRUE_L[(r, c)]]
    return [(free[rng.randint(len(free))], free[rng.randint(len(free))])
            for _ in range(n)]

_ALL_STARTS = _generate_starts(100)


# ─── Worker ───────────────────────────────────────────────────

def _run_job(job):
    gamma, lambda_, tau_r, D_c, seed = job
    si = seed - META_SEED - 1
    rover_start, copter_start = _ALL_STARTS[si]

    devnull = open(os.devnull, 'w')
    old = sys.stdout
    sys.stdout = devnull
    try:
        from scenario import SimulationScenario
        from simulation import run_simulation

        t0 = time.perf_counter()
        sc = SimulationScenario(
            name=f'g{gamma}_l{lambda_}_t{tau_r}_d{D_c}_s{seed}',
            rover_start=rover_start, copter_start=copter_start,
            copter_mode='utility',
            gamma=gamma, lambda_=lambda_,
            tau_r=tau_r, D_c=D_c,
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
        'gamma': gamma, 'lambda': lambda_,
        'tau_r': tau_r,
        'D_c': D_c if D_c != float('inf') else 'inf',
        'seed': seed,
        'copter_mode': 'utility',
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


# ─── CSV I/O ─────────────────────────────────────────────────

def _canon_key(g, l, t, d, s):
    dc_s = 'inf' if (d == float('inf') or str(d) == 'inf') else f'{float(d):.1f}'
    return f'{float(g):.1f}|{float(l):.1f}|{float(t):.2f}|{dc_s}|{int(float(s))}'


def _sort_key(row):
    def _f(v):
        try: return float(v) if v != 'inf' else 1e18
        except: return 0.0
    return tuple(_f(row.get(k, 0)) for k in SORT_KEYS)


def _load_csv():
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return []
    with open(CSV_PATH, 'r', newline='') as f:
        return list(csv.DictReader(f))


def _save_csv(rows):
    rows.sort(key=_sort_key)
    CSV_PATH.parent.mkdir(exist_ok=True)
    with open(CSV_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _build_done_set(rows):
    done = set()
    for r in rows:
        done.add(_canon_key(r['gamma'], r['lambda'], r['tau_r'], r['D_c'], r['seed']))
    return done


# ─── Job runner ───────────────────────────────────────────────

def _run_phase(phase_name, jobs_all, existing, done, n_workers):
    """Run jobs, skip done, save every CHECKPOINT results."""
    jobs = [j for j in jobs_all if _canon_key(j[0], j[1], j[2], j[3], j[4]) not in done]
    n_skip = len(jobs_all) - len(jobs)

    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"  Total: {len(jobs_all)}  Skip: {n_skip}  To run: {len(jobs)}")
    print(f"  Workers: {n_workers}  MAX_K: {MAX_K}")
    est = len(jobs) * 2.0 / n_workers
    print(f"  Est: {est/60:.0f}min")
    print(f"{'='*60}")

    if not jobs:
        print("  All done.")
        return existing, done

    completed = list(existing)
    n_done = n_comp = n_fail = 0
    t0 = time.perf_counter()

    with Pool(processes=n_workers) as pool:
        for row in pool.imap_unordered(_run_job, jobs, chunksize=4):
            completed.append(row)
            key = _canon_key(row['gamma'], row['lambda'],
                             row['tau_r'], row['D_c'], row['seed'])
            done.add(key)
            n_done += 1
            if row['complete']: n_comp += 1
            if row['fail']:     n_fail += 1

            elapsed = time.perf_counter() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(jobs) - n_done) / rate if rate > 0 else 0

            if n_done % 50 == 0 or n_done == len(jobs):
                print(f"  [{n_done:>6d}/{len(jobs)}]  "
                      f"comp={n_comp}  fail={n_fail}  "
                      f"{rate:.1f}/s  ETA {eta/60:.0f}min")

            if n_done % CHECKPOINT == 0:
                _save_csv(completed)

    _save_csv(completed)
    elapsed = time.perf_counter() - t0
    print(f"  Done: {n_done} in {elapsed/60:.1f}min ({n_done/elapsed:.1f}/s)")
    return completed, done


# ─── Phase 2: Adaptive refinement ─────────────────────────────

def _find_top_configs(rows):
    """Group by (gamma, lambda, tau_r, D_c) → compute completion rate → return top N."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (float(r['gamma']), float(r['lambda']),
               float(r['tau_r']),
               float('inf') if r['D_c'] == 'inf' else float(r['D_c']))
        groups[key].append(r)

    scored = []
    for key, rs in groups.items():
        n = len(rs)
        if n < 3:  # need enough seeds
            continue
        nc = sum(1 for r in rs if str(r['complete']) == 'True')
        comp_rate = nc / n
        k_comp = [int(r['k_final']) for r in rs if str(r['complete']) == 'True']
        med_k = sorted(k_comp)[len(k_comp)//2] if k_comp else 999
        score = comp_rate - 0.001 * med_k   # prefer high comp, low k
        scored.append((score, comp_rate, med_k, key))

    scored.sort(reverse=True)

    print(f"\n  Top configs (comp_rate, med_k):")
    for i, (sc, cr, mk, key) in enumerate(scored[:15]):
        print(f"    {i+1:2d}. γ={key[0]:.0f} λ={key[1]:.0f} "
              f"τ={key[2]:.1f} Dc={key[3]:.0f}  "
              f"comp={cr:.0%}  med_k={mk}")

    # Filter by threshold and take top N
    top = [(key, cr) for (sc, cr, mk, key) in scored
           if cr >= PHASE2_COMP_THRESHOLD][:PHASE2_TOP_N]
    return top


def _generate_phase2_jobs(top_configs, n_seeds):
    """Generate refinement jobs around top configs."""
    seeds = [META_SEED + i + 1 for i in range(n_seeds)]
    jobs = set()

    for (g, l, t, d), _ in top_configs:
        for dg in PHASE2_REFINE_RADIUS['gamma']:
            for dl in PHASE2_REFINE_RADIUS['lambda_']:
                g2 = g + dg
                l2 = l + dl
                if g2 < 0 or l2 < 0:
                    continue
                for s in seeds:
                    jobs.add((g2, l2, t, d, s))

    return list(jobs)


# ─── Main ────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-j', '--workers', type=int, default=N_WORKERS)
    p.add_argument('--phase', type=int, choices=[1, 2, 3], default=None,
                   help='Run only one phase (default: all)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    phases = [args.phase] if args.phase else [1, 2, 3]
    seeds = [META_SEED + i + 1 for i in range(N_SEEDS)]

    existing = _load_csv()
    done = _build_done_set(existing)
    print(f"Loaded {len(existing)} existing rows")

    t_total = time.perf_counter()

    # ── Phase 1: Coarse exploration ──
    if 1 in phases:
        p1 = PHASE1
        jobs = list(itertools.product(
            p1['gammas'], p1['lambdas'], p1['tau_rs'], p1['d_cs'], seeds
        ))
        print(f"\nPhase 1: {len(p1['gammas'])}γ × {len(p1['lambdas'])}λ "
              f"× {len(p1['tau_rs'])}τ × {len(p1['d_cs'])}Dc × {N_SEEDS} seeds "
              f"= {len(jobs)} total")

        if not args.dry_run:
            existing, done = _run_phase("Phase 1: Coarse", jobs,
                                        existing, done, args.workers)

    # ── Phase 2: Adaptive refinement ──
    if 2 in phases:
        top = _find_top_configs(existing)
        if not top:
            print("\nPhase 2: No configs above threshold. Run Phase 1 first.")
        else:
            jobs = _generate_phase2_jobs(top, N_SEEDS)
            print(f"\nPhase 2: {len(jobs)} refinement jobs "
                  f"around {len(top)} top configs")

            if not args.dry_run:
                existing, done = _run_phase("Phase 2: Refine", jobs,
                                            existing, done, args.workers)

    # ── Phase 3: Fine-tune sweet spot ──
    if 3 in phases:
        p3 = PHASE3
        jobs = list(itertools.product(
            p3['gammas'], p3['lambdas'], p3['tau_rs'], p3['d_cs'], seeds
        ))
        print(f"\nPhase 3: {len(p3['gammas'])}γ × {len(p3['lambdas'])}λ "
              f"× {len(p3['tau_rs'])}τ × {len(p3['d_cs'])}Dc × {N_SEEDS} seeds "
              f"= {len(jobs)} total")

        if not args.dry_run:
            existing, done = _run_phase("Phase 3: Fine-tune", jobs,
                                        existing, done, args.workers)

    elapsed = time.perf_counter() - t_total
    print(f"\nAll phases done in {elapsed/60:.0f}min. "
          f"CSV: {CSV_PATH} ({len(existing)} rows)")


if __name__ == '__main__':
    main()