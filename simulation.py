"""
simulation.py
"""

import copy
import random
import numpy as np

from environment import GRID
from sensor import init_beliefs
from fsa import FSA_ACCEPT, FSA_DEAD
from planning import value_iteration, compute_b_max
from agents import copter_sense, copter_explore, rover_execute


def run_simulation(
    rover_start  = (0, 0),
    copter_start = (0, 0),
    T_c      = 5,      
    T_r      = 3,     
    alpha    = 1.5,    
    vi_steps = 80,     
    max_k    = 600,   
    warmup   = 0,     
    seed     = 42,
    store_belief_history = False,
) -> dict:
    """

    """
    np.random.seed(seed)
    random.seed(seed)

    # 1 initialise the timestep, copter and rover positions
    k          = 0                  
    copter_pos = copter_start
    rover_pos  = rover_start
    rover_q    = 0  

    # 2 initialise the beliefs using prior knowledge
    beliefs    = init_beliefs() 
    
    history = {
        'rover_path':      [rover_pos],
        'copter_path':     [copter_pos],
        'rover_substeps':  [rover_pos],
        'copter_substeps': [copter_pos],
        'fsa_states':      [rover_q],
        'k_list':          [0],
        'phase_list':      ['I'],  
        'beliefs_snapshot':    [copy.deepcopy(beliefs)],
        'snap_k':          [0],
        'snap_rover':      [rover_pos],
        'snap_copter':     [copter_pos],
        'complete':        False,
        'fail':            False,
        'k_final':         0,
        'which_phi':       None,
        'belief_history':  [] if store_belief_history else None,
    }
    SNAP_TARGETS = 3        # number of snapshots to take

    if store_belief_history:
        history['belief_history'].append(copy.deepcopy(beliefs))

    complete = False
    
    copter_sense(beliefs, copter_pos)
    history['beliefs_snapshot'][0] = copy.deepcopy(beliefs)
    if store_belief_history:
        history['belief_history'][-1] = copy.deepcopy(beliefs)

    # 3 rover computes the optimal policy and b_max 
    print("  Initial value iteration...")
    V, policy, vi_sweeps = value_iteration(beliefs, vi_steps=vi_steps, T_r=T_r)
    print(f"  Initial value iteration converged in {vi_sweeps}/{vi_steps} sweeps.")
    b_max     = compute_b_max(beliefs, policy, rover_pos, rover_q, T_r)

    # 4 repeat exploration phase (algorithm 2 or 3) and execution phase (algorithm 4)
    while k < max_k and not complete:

        # 4.1 exploration phase (algorithm 2 or 3)
        k_before_c = k
        copter_pos, c_substeps, c_bsnaps = copter_explore(
            beliefs, copter_pos, b_max, T_c, alpha,
            record_beliefs=store_belief_history)
        k += T_c

        for step_i, cpos in enumerate(c_substeps):
            history['copter_substeps'].append(cpos)
            history['rover_substeps'].append(rover_pos)
            history['k_list'].append(k_before_c + step_i + 1)
            history['phase_list'].append('C')
            if store_belief_history:
                snap = (c_bsnaps[step_i] if step_i < len(c_bsnaps)
                        else copy.deepcopy(beliefs))
                history['belief_history'].append(snap)

        V, policy, vi_sweeps = value_iteration(beliefs, vi_steps=vi_steps, T_r=T_r)
        b_max     = compute_b_max(beliefs, policy, rover_pos, rover_q, T_r)
        
        # 4.2 execution phase (algorithm 4)
        k_before_r = k
        rover_pos, rover_q, substeps, r_bsnaps = rover_execute(
            beliefs, rover_pos, rover_q, policy, T_r,
            record_beliefs=store_belief_history)
        k += T_r

        for step_i, (spos, sq) in enumerate(substeps[1:]):
            history['rover_substeps'].append(spos)
            history['copter_substeps'].append(copter_pos)
            history['k_list'].append(k_before_r + step_i + 1)
            history['phase_list'].append('R')
            if store_belief_history:
                snap_i = step_i + 1
                snap = (r_bsnaps[snap_i] if snap_i < len(r_bsnaps)
                        else copy.deepcopy(beliefs))
                history['belief_history'].append(snap)

        history['rover_path'].append(rover_pos)
        history['copter_path'].append(copter_pos)
        history['fsa_states'].append(rover_q)

        n_snaps     = len(history['snap_k'])
        next_snap_k = history['snap_k'][0] + (max_k // SNAP_TARGETS) * n_snaps
        if (rover_q in FSA_ACCEPT or rover_q == FSA_DEAD
                or k >= next_snap_k):
            if n_snaps < SNAP_TARGETS + 1:
                history['beliefs_snapshot'].append(copy.deepcopy(beliefs))
                history['snap_k'].append(k)
                history['snap_rover'].append(rover_pos)
                history['snap_copter'].append(copter_pos)

        complete = rover_q in FSA_ACCEPT
        if complete:
            which = {3: 'φ1 (found A)', 4: 'φ2 (B→C)',
                     5: 'φ3 (C→D)'}.get(rover_q, '?')
            history['which_phi'] = which
            print(f"   MISSION COMPLETE at k={k}, branch={which}, pos={rover_pos}"
                  f"[VI={vi_sweeps}/{vi_steps} sweeps]")
        elif rover_q == FSA_DEAD:
            print(f"   MISSION FAILED (obstacle) at k={k}, pos={rover_pos}")
            history['fail'] = True
            break
        elif k % 200 == 0:
            V_here = V.get((rover_pos[0], rover_pos[1], rover_q), 0.0)
            print(f"  k={k:4d}  rover={rover_pos}  q={rover_q}  "
                  f"V={V_here:.3f}  copter={copter_pos}"
                  f"VI={vi_sweeps}/{vi_steps} sweeps")

    if not complete and not history['fail']:
        print(f"  Time limit reached at k={max_k}")

   
    if history['snap_k'][-1] < k:
        history['beliefs_snapshot'].append(copy.deepcopy(beliefs))
        history['snap_k'].append(k)
        history['snap_rover'].append(rover_pos)
        history['snap_copter'].append(copter_pos)

    if store_belief_history:
        if not history['k_list'] or history['k_list'][-1] != k:
            history['belief_history'].append(copy.deepcopy(beliefs))
            history['rover_substeps'].append(rover_pos)
            history['copter_substeps'].append(copter_pos)
            history['k_list'].append(k)
            history['phase_list'].append('E')

    history['complete']     = complete
    history['k_final']      = k
    history['final_rover']  = rover_pos
    history['final_copter'] = copter_pos
    history['final_q']      = rover_q
    return history