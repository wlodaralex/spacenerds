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
    - warmup: Practical heuristic that maintains strict 'algorithm 1' adherence when warmup=0 (see CHANGELOG.md).
        * 0 -> copter senses once from start at k=0, matching the the non-unform belief patch visible in right-most plot of Fig. 3.
        * N -> N copter-only exploration prior to rover movement, uses a rover-centred b_max so pre-exploration covers the rover's upcoming area rather than wandering randomly.
    - seed: Fixes the pseudo-random state for stochastic transitions and sensor noise, ensuring results are reproducible across runs.
            Added for scientific reproducibility and presentation consistency (the same seed produces the same trajectory every run).
    - store_belief_history: When 'True', records a full deepcopy of the 10x10 belief grid at every timestep k (required for some visualizations).
                            Can be disabled for large-scale testing to limit memory use and reduce runtime.
"""

import os
from simulation import run_simulation
from visualization import *

# Create output directory in user's home folder
HOME = os.path.expanduser('~') # Get user's home directory
outpath = os.path.join(HOME, "aer1516", "project", "outputs")


def main():
    os.makedirs(outpath, exist_ok=True)

    scenarios = [
        {
            'name': 'global',
            'exploration_policy': 'global',
            'rover_start': (0, 0),
            'copter_start': (0, 0),
            'T_c': 5,
            'T_r': 3,
            'alpha': 1.5,
            'vi_steps': 200,
            'max_k': 600,
            'warmup': 0, 
            'seed': 12,
            'store_belief_history': True,
        },
        # {
        #     'name': 'local',
        #     'exploration_policy': 'local',
        #     'rover_start': (0, 0),
        #     'copter_start': (0, 0),
        #     'T_c': 5,
        #     'T_r': 3,
        #     'alpha': 1.5,
        #     'vi_steps': 200,
        #     'max_k': 600,
        #     'warmup': 0, 
        #     'seed': 12,
        #     'store_belief_history': True,
        # },
    ]

    results = {}
    
    for scenario in scenarios:
        name = scenario['name']
        print("\n" + "=" * 60)
        print(f"SCENARIO {name}  |  rover_start={scenario['rover_start']}, copter_start={scenario['copter_start']}")
        print("=" * 60)

        hist = run_simulation(
            rover_start=scenario['rover_start'],
            copter_start=scenario['copter_start'],
            T_c=scenario['T_c'],
            T_r=scenario['T_r'],
            alpha=scenario['alpha'],
            vi_steps=scenario['vi_steps'],
            max_k=scenario['max_k'],
            warmup=scenario['warmup'],
            seed=scenario['seed'],
            store_belief_history=scenario['store_belief_history'],
            exploration_policy=scenario['exploration_policy'],
        )
        results[name] = hist

        prefix = os.path.join(outpath, f"scen{name}")

        make_belief_snapshots_plot(hist, filepath=prefix + "_belief_snapshots.png")
        make_trajectories_plot(hist, filepath=prefix + "_trajectories.png", use_substeps=True)
        make_final_beliefs_plot(hist, filepath=prefix + "_final_beliefs.png")
        make_v_fn_heatmap(hist['beliefs_snapshot'][0], filepath=prefix + "_value_function_q0.png", rover_pos=scenario['rover_start'], rover_q=0)
        make_convergence_plot(hist, filepath=prefix + "_convergence.png")
        make_beliefs_animation(hist, filepath=prefix + "_beliefs_animation.mp4", fps=4, use_substeps=True)
        make_unified_animation(hist, filepath=prefix + "_unified_animation.mp4", fps=4, use_substeps=True)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for name, hist in results.items():
        status = ("COMPLETE" if hist['complete'] else
                  ("FAILED"  if hist['fail']     else "TIMEOUT"))
        print(f"  {name.upper():6s}:  k_final={hist['k_final']:4d}  "
              f"branch={hist['which_phi']}  status={status}")
    print("=" * 60)
    print(f"\nOutputs saved to: {outpath}")


if __name__ == '__main__':
    main()