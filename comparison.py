"""
comparison.py - 

setups tried: 
* 0.5 -> rover/copter: 1e-4 tie, 1e-3 tol (only global completes main.py)
* 0.5 -> rover/copter: 1e-6 tie, 1e-4 tol (only global completes main.py)
* 0.3 -> rover/copter: 1e-5/1e-4 tie, 1e-3 tol (global + local completes)
"""

import time
# import random
import numpy as np
import os
import sys
from io import StringIO

from simulation import run_simulation
from environment import GRID, TRUE_L, CARDINALS, clip


def run_comparison_study(n_trials=100, output_dir=None, measure_runtime=True, rng_starts=None):
    """
    Run local vs. global exploration comparison over n_trials.
    
    Each trial:
    1. Randomly samples rover_start and copter_start
    2. Runs BOTH local and global with the same seed (for fair comparison)
    """
    if output_dir is None:
        HOME = os.path.expanduser('~')
        output_dir = os.path.join(HOME, "aer1516", "project", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"COMPARISON STUDY: {n_trials} trials (local vs. global exploration)")
    print("=" * 70)
    print(f"  Max timesteps: k=300")
    print(f"  Random rover/copter starts: uniform from (0,0) to ({GRID},{GRID})")
    print(f"  Seed: 12 (same as main.py, for fair comparison)")
    print(f"  Metrics: Success rate, avg k, {'avg runtime' if measure_runtime else ''}")
    print("=" * 70 + "\n")
    
    results = {
        'local': {'completions': [], 'k_finals': [], 'runtimes': []},
        'global': {'completions': [], 'k_finals': [], 'runtimes': []}
    }
    
    # CSV logging
    csv_path = os.path.join(output_dir, "comparison_trials.csv")
    header = "trial,rover_start,copter_start,local_completed,local_k"
    if measure_runtime:
        header += ",local_runtime"
    header += ",global_completed,global_k"
    if measure_runtime:
        header += ",global_runtime"
    header += "\n"
    
    with open(csv_path, 'w') as f:
        f.write(header)
    
    for trial in range(n_trials):
        # Random starting positions (excluding obstacles)
        while True:
            rover_start = (rng_starts.randint(0, GRID), rng_starts.randint(0, GRID))
            copter_start = (rng_starts.randint(0, GRID), rng_starts.randint(0, GRID))
            
            # def is_bad_start(pos):
            #     if 'O' in TRUE_L[pos]:
            #         return True
            #     for dr, dc in CARDINALS:
            #         nb = clip(pos[0] + dr, pos[1] + dc)
            #         if 'O' in TRUE_L[nb]:
            #             return True
            #     return False
            # if not is_bad_start(rover_start):
            #     break
            
            # if ('O' not in TRUE_L[rover_start] and 'O' not in TRUE_L[copter_start]):
            #     break

            if 'O' not in TRUE_L[rover_start]:
                break
            
            # if TRUE_L[rover_start] or TRUE_L[copter_start]:
            #     continue
            # break



        # # Use same seed for both policies (changed from 1000+trial)
        # seed = 1000 + trial
        # seed = 42  #  Use same seed as main.py for consistency
        seed = trial
        
        print(f"Trial {trial+1:3d}/{n_trials}  "
              f"rover={rover_start}  copter={copter_start}", 
              end="", flush=True)
        
        # Run local
        start_time = time.time() if measure_runtime else 0
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            hist_local = run_simulation(
                rover_start=rover_start,
                copter_start=copter_start,
                T_c=5, T_r=3, alpha=1.5, vi_steps=200,
                max_k=300, warmup=0, seed=seed,  # ← Same seed
                store_belief_history=False,
                use_substeps=False,
                exploration_policy='local',
                verbose=False,
            )
        finally:
            sys.stdout = old_stdout
        
        local_time = time.time() - start_time if measure_runtime else 0
        local_completed = hist_local['complete'] and hist_local['k_final'] <= 300
        local_k = hist_local['k_final']
        
        results['local']['completions'].append(local_completed)
        results['local']['k_finals'].append(local_k if local_completed else 300)
        results['local']['runtimes'].append(local_time)
        
        # Run global
        start_time = time.time() if measure_runtime else 0
        
        sys.stdout = StringIO()
        try:
            hist_global = run_simulation(
                rover_start=rover_start,
                copter_start=copter_start,
                T_c=5, T_r=3, alpha=1.5, vi_steps=200,
                max_k=300, warmup=0, seed=seed,  # ← Same seed as local
                store_belief_history=False,
                use_substeps=False,
                exploration_policy='global',
                verbose=False,
            )
        finally:
            sys.stdout = old_stdout
        
        global_time = time.time() - start_time if measure_runtime else 0
        global_completed = hist_global['complete'] and hist_global['k_final'] <= 300
        global_k = hist_global['k_final']
        
        results['global']['completions'].append(global_completed)
        results['global']['k_finals'].append(global_k if global_completed else 300)
        results['global']['runtimes'].append(global_time)
        
        # # DEBUG (after both k values are computed)
        # if local_k == 8 or global_k == 8:
        #     print(f"\n  [DEBUG k=8: Trial {trial+1}, rover={rover_start}, copter={copter_start}]")
        #     print(f"    Local: k={local_k}, fail={hist_local['fail']}, complete={hist_local['complete']}")
        #     print(f"    Global: k={global_k}, fail={hist_global['fail']}, complete={hist_global['complete']}")
            
        #     # Check initial labels
        #     print(f"    Rover start labels: {TRUE_L[rover_start]}")
        #     print(f"    Copter start labels: {TRUE_L[copter_start]}")
            
        #     # Check what caused the failure
        #     if hist_local['fail']:
        #         print(f"    Local trajectory length: {len(hist_local.get('states', []))}")
        #         print(f"    Local FSA history length: {len(hist_local.get('fsa_history', []))}")
        #         if len(hist_local.get('fsa_history', [])) > 0:
        #             print(f"    Initial FSA state: {hist_local['fsa_history'][0]}")
            
        #     if hist_global['fail']:
        #         print(f"    Global trajectory length: {len(hist_global.get('states', []))}")
        #         print(f"    Global FSA history length: {len(hist_global.get('fsa_history', []))}")
        #         if len(hist_global.get('fsa_history', [])) > 0:
        #             print(f"    Initial FSA state: {hist_global['fsa_history'][0]}")        
        
        #     # Check if simulation actually ran
        #     print(f"    Local rover_path length: {len(hist_local.get('rover_path', []))}")
        #     print(f"    Local copter_path length: {len(hist_local.get('copter_path', []))}")
        #     print(f"    Global rover_path length: {len(hist_global.get('rover_path', []))}")
        #     print(f"    Global copter_path length: {len(hist_global.get('copter_path', []))}")

        #     print(f"\n  [DEBUG k=8: Trial {trial+1}, rover={rover_start}, copter={copter_start}]")
        #     print(f"    Local: k={local_k}, fail={hist_local['fail']}, complete={hist_local['complete']}")
        #     print(f"    Global: k={global_k}, fail={hist_global['fail']}, complete={hist_global['complete']}")
        #     print(f"    Rover start labels: {TRUE_L[rover_start]}")
        #     print(f"    Local final_q: {hist_local.get('final_q', '?')}")
        #     print(f"    Global final_q: {hist_global.get('final_q', '?')}")
            
        #     # Show rover's first move
        #     if len(hist_local['rover_path']) >= 2:
        #         print(f"    Local rover moved: {hist_local['rover_path'][0]} → {hist_local['rover_path'][1]}")
        #         print(f"    Labels at new pos: {TRUE_L[hist_local['rover_path'][1]]}")
        #     if len(hist_global['rover_path']) >= 2:
        #         print(f"    Global rover moved: {hist_global['rover_path'][0]} → {hist_global['rover_path'][1]}")
        #         print(f"    Labels at new pos: {TRUE_L[hist_global['rover_path'][1]]}")

        # Compact progress
        progress = f"  | LOCAL: k={local_k:3d} {'✓' if local_completed else '✗'}"
        if measure_runtime:
            progress += f" {local_time:5.2f}s"
        progress += f"  | GLOBAL: k={global_k:3d} {'✓' if global_completed else '✗'}"
        if measure_runtime:
            progress += f" {global_time:5.2f}s"
        print(progress)
        
        # CSV logging
        row = f"{trial+1},{rover_start},{copter_start}," \
              f"{int(local_completed)},{local_k}"
        if measure_runtime:
            row += f",{local_time:.3f}"
        row += f",{int(global_completed)},{global_k}"
        if measure_runtime:
            row += f",{global_time:.3f}"
        row += "\n"
        
        with open(csv_path, 'a') as f:
            f.write(row)
    

    # Summary Statistics
    local_success = sum(results['local']['completions'])
    global_success = sum(results['global']['completions'])
    
    # Average k (only for successful trials)
    local_k_successful = [k for k, c in zip(results['local']['k_finals'], 
                                             results['local']['completions']) if c]
    global_k_successful = [k for k, c in zip(results['global']['k_finals'], 
                                              results['global']['completions']) if c]
    
    # Calculate avg k, or None if no completions
    local_avg_k = np.mean(local_k_successful) if local_k_successful else None
    global_avg_k = np.mean(global_k_successful) if global_k_successful else None
    
    # Average runtime over ALL trials (including failures)
    local_avg_time = np.mean(results['local']['runtimes']) if measure_runtime else 0
    global_avg_time = np.mean(results['global']['runtimes']) if measure_runtime else 0
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (Table 1 Reproduction)")
    print("=" * 70)
    
    if measure_runtime:
        print(f"{'Policy':<22} {'Completions':<15} {'Avg k':<12} {'Avg Runtime (s)':<15}")
        print("-" * 70)
        
        # Format avg k with N/A if None
        local_k_str = f"{local_avg_k:.1f}" if local_avg_k is not None else "N/A"
        global_k_str = f"{global_avg_k:.1f}" if global_avg_k is not None else "N/A"
        
        print(f"{'Local (Algorithm 2)':<22} {local_success}/{n_trials:<12} "
              f"{local_k_str:<12} {local_avg_time:<15.2f}")
        print(f"{'Global (Algorithm 3)':<22} {global_success}/{n_trials:<12} "
              f"{global_k_str:<12} {global_avg_time:<15.2f}")
    else:
        print(f"{'Policy':<22} {'Completions':<15} {'Avg k':<12}")
        print("-" * 70)
        
        local_k_str = f"{local_avg_k:.1f}" if local_avg_k is not None else "N/A"
        global_k_str = f"{global_avg_k:.1f}" if global_avg_k is not None else "N/A"
        
        print(f"{'Local (Algorithm 2)':<22} {local_success}/{n_trials:<12} {local_k_str:<12}")
        print(f"{'Global (Algorithm 3)':<22} {global_success}/{n_trials:<12} {global_k_str:<12}")
    
    print("=" * 70)
    print(f"\nPaper's results (for reference):")
    print(f"  Local:  62/100 completions (3.0s avg)")
    print(f"  Global: 71/100 completions (31s avg)")
    print(f"\nInterpretation:")
    
    # Handle N/A in interpretation
    if local_avg_k is not None and global_avg_k is not None:
        print(f"  - Global completes missions in fewer timesteps ({global_avg_k:.0f} vs {local_avg_k:.0f})")
    elif global_avg_k is not None:
        print(f"  - Global completes missions in {global_avg_k:.0f} timesteps (Local had no completions)")
    elif local_avg_k is not None:
        print(f"  - Local completes missions in {local_avg_k:.0f} timesteps (Global had no completions)")
    else:
        print(f"  - Neither policy completed any missions")
    
    if measure_runtime:
        print(f"  - Runtime comparison: {global_avg_time:.1f}s vs {local_avg_time:.1f}s per trial")
    print(f"\nPer-trial data: {csv_path}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    # np.random.seed(42)

    rng_starts = np.random.RandomState(42)
    
    # Run the study (set measure_runtime=False to skip timing)
    results = run_comparison_study(n_trials=100, measure_runtime=True, rng_starts=rng_starts)
    
    # Optional: Generate summary plots
    try:
        import matplotlib
        matplotlib.use('Agg')  # Suppress plot windows
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Completion counts
        ax1 = axes[0]
        policies = ['Local', 'Global']
        completions = [
            sum(results['local']['completions']),
            sum(results['global']['completions'])
        ]
        ax1.bar(policies, completions, color=['steelblue', 'tomato'], alpha=0.8)
        ax1.set_ylabel('Mission Completions (out of 100)', fontsize=11)
        ax1.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(completions):
            ax1.text(i, v + 2, f'{v}/100', ha='center', fontweight='bold', fontsize=10)
        
        # Plot 2: k distribution
        ax2 = axes[1]
        local_k_successful = [k for k, c in zip(results['local']['k_finals'], 
                                                 results['local']['completions']) if c]
        global_k_successful = [k for k, c in zip(results['global']['k_finals'], 
                                                  results['global']['completions']) if c]
        
        bp = ax2.boxplot([local_k_successful, global_k_successful],
                         labels=policies, patch_artist=True)
        colors = ['steelblue', 'tomato']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.set_ylabel('Timesteps to Completion (k)', fontsize=11)
        ax2.set_title('Mission Completion Speed', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        HOME = os.path.expanduser('~')
        outpath = os.path.join(HOME, "aer1516", "project", "outputs")
        fig_path = os.path.join(outpath, "comparison_summary.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSummary plots saved to: {fig_path}")
    except Exception as e:
        print(f"\nSkipping plots: {e}")