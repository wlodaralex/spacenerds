"""
main.py

- load scenarios, run simulations, and save outputs
- provide an optional baseline table reproduction entry point

function
- `_load_scenarios`: read scenario json into typed configs
- `main`: run the configured scenarios
- `table1_experiment`: reproduce the baseline comparison table
"""

import json
import time
from pathlib import Path

import numpy as np

from scenario import SimulationScenario
from simulation import run_simulation
from visualization import (
    make_belief_snapshots_plot,
    make_beliefs_animation,
    make_convergence_plot,
    make_final_beliefs_plot,
    make_trajectories_plot,
    make_unified_animation,
    make_v_fxn_heatmap,
)

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


# --- Scenario Loading ---
def _load_scenarios(path: Path) -> list[SimulationScenario]:
    """Load scenario JSON into typed ``SimulationScenario`` objects."""
    with path.open() as f:
        raw_scenarios = json.load(f)
    return [SimulationScenario.from_dict(raw) for raw in raw_scenarios]


# --- Main Entry Point ---
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    scenarios = _load_scenarios(BASE_DIR / 'scenarios.json')

    for scenario in scenarios:
        print(
            f"\nscenario: {scenario.name}  |  "
            f"rover_start={scenario.rover_start}, "
            f"copter_start={scenario.copter_start}"
        )

        # step 1 run one scenario
        start_time = time.perf_counter()
        hist = run_simulation(scenario = scenario)
        solve_time_s = time.perf_counter() - start_time
        hist.meta['scenario_name'] = scenario.name
        hist.meta['solve_time_s'] = solve_time_s

        # step 2 save outputs
        prefix = OUTPUT_DIR / f"scen{scenario.name}"
        hist.save(
            f"scen{scenario.name}",
            outdir = str(OUTPUT_DIR),
        )
        legacy = hist.to_legacy_dict()

        # step 3 make plots
        # make_belief_snapshots_plot(legacy, filepath=f"{prefix}_belief_snapshots.png")
        make_trajectories_plot(
            legacy,
            filepath     = f"{prefix}_trajectories.png",
            use_substeps = True,
        )
        # make_final_beliefs_plot(legacy, filepath=f"{prefix}_final_beliefs.png")
        # make_v_fxn_heatmap(
        #     legacy['beliefs_snapshot'][0],
        #     filepath=f"{prefix}_value_function_q0.png",
        #     rover_pos=scenario.rover_start,
        #     rover_q=0,
        # )
        make_convergence_plot(
            legacy,
            filepath = f"{prefix}_convergence.png",
        )
        # make_beliefs_animation(
        #     legacy,
        #     filepath=f"{prefix}_beliefs_animation.mp4",
        #     fps=4,
        #     use_substeps=True,
        # )
        # make_unified_animation(
        #     legacy,
        #     filepath=f"{prefix}_unified_animation.mp4",
        #     fps=4,
        #     use_substeps=True,
        # )

    print("\nOutputs saved to:", OUTPUT_DIR)


# --- Table 1 Reproduction ---
def table1_experiment(n_trials=100, max_k=300, meta_seed=0):
    """reproduce the local/global baseline table."""
    from environment import GRID, TRUE_L

    rng = np.random.default_rng(meta_seed)

    free_space = [(r, c) for r in range(GRID) for c in range(GRID)
                  if 'O' not in TRUE_L[(r, c)]]

    results = {
        'local':  {'complete': 0, 'times': []},
        'global': {'complete': 0, 'times': []},
    }

    starts = []
    for i in range(n_trials):
        rover  = free_space[rng.integers(len(free_space))]
        copter = free_space[rng.integers(len(free_space))]
        starts.append((rover, copter))

    for i, (rover, copter) in enumerate(starts):
        seed_i = meta_seed + i + 1

        trial_ok = {}
        for mode in ('local', 'global'):
            t0 = time.time()
            scenario = SimulationScenario(
                name         = f'table1_{mode}_{i+1}',
                rover_start  = rover,
                copter_start = copter,
                T_c          = 5,
                T_r          = 3,
                alpha        = 1.5,
                gamma        = 0.0,
                lambda_      = 0.0,
                rho          = 1.0,
                tau_r        = 0.9,
                vi_steps     = 80,
                max_k        = max_k,
                seed         = seed_i,
                copter_mode  = mode,
                D_c          = float('inf'),
            )
            hist = run_simulation(scenario = scenario)
            elapsed = time.time() - t0
            results[mode]['times'].append(elapsed)
            trial_ok[mode] = hist.complete
            if hist.complete:
                results[mode]['complete'] += 1

        local_s  = 'complete' if trial_ok['local']  else '--'
        global_s = 'complete' if trial_ok['global'] else '--'
        print(f"  Trial {i+1:3d}/{n_trials}  "
              f"rover={str(rover):>7s} copter={str(copter):>7s}  "
              f"local={local_s}  global={global_s}")

    print("Table 1 Reproduction (Section 6.1.2)")
    print(f"  {'':30} {'Completions':>14}   {'Avg time (s)':>13}")
    print(f"  {'-'*30} {'-'*14}   {'-'*13}")
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
