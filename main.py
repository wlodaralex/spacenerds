"""run scenarios and generate output artifacts."""

import json
import os
import time
import numpy as np
from simulation import run_simulation
from visualization import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
outpath = os.path.join(BASE_DIR, "outputs")

def _load_scenarios(path: str) -> list:
    """load scenarios and normalize tuple fields."""
    with open(path) as f:
        raw = json.load(f)
    for s in raw:
        s['rover_start']  = tuple(s['rover_start'])
        s['copter_start'] = tuple(s['copter_start'])
    return raw


def main():
    os.makedirs(outpath, exist_ok=True)

    scenarios = _load_scenarios(os.path.join(BASE_DIR, 'scenarios.json'))

    for scenario in scenarios:
        print(f"SCENARIO {scenario['name']}  |  rover_start={scenario['rover_start']}, copter_start={scenario['copter_start']}")

        # step 1 run one scenario
        start_time = time.perf_counter()
        hist = run_simulation(
            rover_start=scenario['rover_start'],
            copter_start=scenario['copter_start'],
            T_c=scenario['T_c'],
            T_r=scenario['T_r'],
            alpha=scenario['alpha'],
            gamma=scenario.get('gamma', scenario.get('gamma', 0.0)),
            lambda_=scenario.get('lambda', 0.0),
            tau_r=scenario.get('tau_r', 1.0),
            vi_steps=scenario['vi_steps'],
            max_k=scenario['max_k'],
            copter_mode=scenario.get('copter_mode', 'global'),
            seed=scenario['seed'],
            copter_energy=scenario.get('copter_energy') or float('inf'),
        )
        solve_time_s = time.perf_counter() - start_time
        hist['scenario_name'] = scenario['name']
        hist['solve_time_s'] = solve_time_s
        hist['gamma'] = scenario.get('gamma', 0.0)
        hist['lambda'] = scenario.get('lambda', 0.0)

        # step 2 save outputs
        prefix = os.path.join(outpath, f"scen{scenario['name']}")
        hist.save(f"scen{scenario['name']}", outdir=outpath)
        legacy = hist.to_legacy_dict()

        # optional plots
        # make_belief_snapshots_plot(legacy, filepath=prefix + "_belief_snapshots.png")
        make_trajectories_plot(legacy, filepath=prefix + "_trajectories.png", use_substeps=True)
        # make_final_beliefs_plot(legacy, filepath=prefix + "_final_beliefs.png")
        # make_v_fxn_heatmap(legacy['beliefs_snapshot'][0], filepath=prefix + "_value_function_q0.png", rover_pos=scenario['rover_start'], rover_q=0)
        make_convergence_plot(legacy, filepath=prefix + "_convergence.png")
        # make_beliefs_animation(legacy, filepath=prefix + "_beliefs_animation.mp4", fps=4, use_substeps=True)
        # make_unified_animation(legacy, filepath=prefix + "_unified_animation.mp4", fps=4, use_substeps=True)
    
    print("\n Outputs saved to:", outpath)


def table1_experiment(n_trials=100, max_k=300, meta_seed=0):
    """reproduce the local/global baseline table."""
    from environment import GRID, TRUE_L

    rng = np.random.RandomState(meta_seed)

    free_space = [(r, c) for r in range(GRID) for c in range(GRID)
                  if 'O' not in TRUE_L[(r, c)]]

    results = {
        'local':  {'complete': 0, 'times': []},
        'global': {'complete': 0, 'times': []},
    }

    starts = []
    for i in range(n_trials):
        rover  = free_space[rng.randint(len(free_space))]
        copter = free_space[rng.randint(len(free_space))]
        starts.append((rover, copter))

    for i, (rover, copter) in enumerate(starts):
        seed_i = meta_seed + i + 1

        trial_ok = {}
        for mode in ('local', 'global'):
            t0 = time.time()
            hist = run_simulation(
                rover_start=rover,
                copter_start=copter,
                T_c=5, T_r=3, alpha=1.5,
                gamma=0.0, lambda_=0.0,
                vi_steps=80,
                max_k=max_k,
                copter_mode=mode,
                seed=seed_i,
            )
            elapsed = time.time() - t0
            results[mode]['times'].append(elapsed)
            trial_ok[mode] = hist['complete']
            if hist['complete']:
                results[mode]['complete'] += 1

        local_s  = 'complete' if trial_ok['local']  else '--'
        global_s = 'complete' if trial_ok['global'] else '--'
        print(f"  Trial {i+1:3d}/{n_trials}  "
              f"rover={str(rover):>7s} copter={str(copter):>7s}  "
              f"local={local_s}  global={global_s}")

    print("Table 1 Reproduction (Section 6.1.2)")
    header = f"  {'':30} {'Completions':>14}   {'Avg time (s)':>13}"
    sep    = f"  {'-'*30} {'-'*14}   {'-'*13}"
    print(header)
    print(sep)
    for mode in ('local', 'global'):
        alg = 'Algorithm 2' if mode == 'local' else 'Algorithm 3'
        label = f"{mode.capitalize()} ({alg})"
        comp  = results[mode]['complete']
        avg_t = np.mean(results[mode]['times'])
        print(f"  {label:30s} {comp:>7d}/{n_trials}     {avg_t:>10.1f}")
    print(f"  Paper reference:  Local  62/100, 3.0 s")
    print(f"                    Global 71/100, 31  s")


if __name__ == '__main__':
    main()
    # table1_experiment(n_trials=100, max_k=300, meta_seed=0)
