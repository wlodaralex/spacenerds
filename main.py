"""
main.py - Entry point for collaborative rover-copter simulation.

Reference:
    [1] K. Hashimoto, N. Tsumagari, and T. Ushio, 'Collaborative Rover-copter 
        Path Planning and Exploration with Temporal Logic Specifications Based 
        on Bayesian Update Under Uncertain Environments', ACM Transactions on 
        Cyber-Physical Systems, vol. 6, no. 2, pp. 1-24, Apr. 2022.

Usage:
    - Run `python main.py` from directory (where all module files exist).
    - Visualizations written to outpath defined.

Scenario(s):
    - By default, only one scenario is displayed (seen below).
    - To simulate other scenarios in same run, modify the relevant section.

Model parameters (Section 6.1.1):
    - T_c = 5:
        * Copter exploration phase length. Each cycle the copter takes T_c steps, sensing obstacle labels (AP_c = {O}) and updating beliefs.
    - T_r = 3:
        * Rover execution phase length. Each cycle the rover follows its optimal policy (synthesized via value iteration) for T_r steps before replanning.
        * NOTE: T_r > 5 may cause rover to appear stuck due to a stay-action V-peak trap (see CHANGELOG.md).
    - alpha = 1.5:
        * Eq. 10: Acquisition weight in W(x) = H(B(x|=O)) + alpha * b_max(x).
        * Balances entropy-driven exploration versus rover-path-biased exploration.
        * Higher values pull the copter toward cells the rover is likely to visit.

Implementation parameters:
    - vi_steps: Number of value-iteration sweeps (Eqs. 20-21). The paper assumes convergence to the optimal policy μ* 
                but does not specify a sweep count, in practice the algorithm is run to convergence. 
    - max_k: Hard time limit (total timesteps) added for computational practicality. Prevents the simulation 
             from running indefinitely if beliefs do not converge or the rover cannot find a path.
    - copter_mode: Selects the copter exploration algorithm.
        * 'local' -> Algorithm 2 (local selection-based exploration, Eq. 11).
        * 'global' -> Algorithm 3 (global selection-based exploration, Eqs. 12-13).
    - seed: Fixes the pseudo-random state for stochastic transitions and sensor noise, ensuring results are reproducible across runs.
            Added for scientific reproducibility and presentation consistency (the same seed produces the same trajectory every run).
    - store_belief_history: When 'True', records a full deepcopy of the 10x10 belief grid at every timestep k (required for some visualizations).
                            Can be disabled for large-scale testing to limit memory use and reduce runtime.
"""

import os
import time
import numpy as np
from simulation import run_simulation
from visualization import *

# Create output directory in user's home folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # spacenerds directory
outpath = os.path.join(BASE_DIR, "outputs")

def main():
    os.makedirs(outpath, exist_ok=True)

    scenarios = [
        {
            'name': '_global',
            'rover_start': (0, 0),
            'copter_start': (0, 0),
            'T_c': 5,
            'T_r': 3,
            'alpha': 1.5,
            'vi_steps': 80,
            'max_k': 600,
            'copter_mode': 'global',
            'seed': 12,
            'store_belief_history': True,
        },
            {
            'name': '_local',
            'rover_start': (0, 0),
            'copter_start': (0, 0),
            'T_c': 5,
            'T_r': 3,
            'alpha': 1.5,
            'vi_steps': 80,
            'max_k': 600,
            'copter_mode': 'local',
            'seed': 12,
            'store_belief_history': True,
        },
    ]

    for scenario in scenarios:
        print("\n" + "=" * 60)
        print(f"SCENARIO {scenario['name']}  |  rover_start={scenario['rover_start']}, copter_start={scenario['copter_start']}")
        print("=" * 60)

        hist = run_simulation(
            rover_start=scenario['rover_start'],
            copter_start=scenario['copter_start'],
            T_c=scenario['T_c'],
            T_r=scenario['T_r'],
            alpha=scenario['alpha'],
            vi_steps=scenario['vi_steps'],
            max_k=scenario['max_k'],
            copter_mode=scenario.get('copter_mode', 'global'),
            seed=scenario['seed'],
            store_belief_history=scenario['store_belief_history'],
        )

        prefix = os.path.join(outpath, f"scen{scenario['name']}")

        make_belief_snapshots_plot(hist, filepath=prefix + "_belief_snapshots.png")
        make_trajectories_plot(hist, filepath=prefix + "_trajectories.png", use_substeps=True)
        make_final_beliefs_plot(hist, filepath=prefix + "_final_beliefs.png")
        make_v_fxn_heatmap(hist['beliefs_snapshot'][0], filepath=prefix + "_value_function_q0.png", rover_pos=scenario['rover_start'], rover_q=0)
        make_convergence_plot(hist, filepath=prefix + "_convergence.png")
        make_beliefs_animation(hist, filepath=prefix + "_beliefs_animation.mp4", fps=4, use_substeps=True)
        make_unified_animation(hist, filepath=prefix + "_unified_animation.mp4", fps=4, use_substeps=True)
    
    print("\n Outputs saved to:", outpath)


def table1_experiment(n_trials=100, max_k=300, meta_seed=0):
    """
    Reproduce Table 1 from the paper (Section 6.1.2).

    For each trial:
        1. Randomly sample rover and copter start positions from X.
        2. Run Algorithm 1 with local exploration  (Algorithm 2).
        3. Run Algorithm 1 with global exploration (Algorithm 3).
    Both use the same start positions and per-trial seed.

    Report:
        * Number of missions completed before k = max_k.
        * Average running time (s) per simulation.
    """
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
                T_c=5, T_r=3, alpha=1.5, vi_steps=80,
                max_k=max_k,
                copter_mode=mode,
                seed=seed_i,
                store_belief_history=False,
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
    # main()
    table1_experiment(n_trials=100, max_k=300, meta_seed=0)