"""
agents.py - Rover mission execution
          - Copter exploration

Setup:
    - Acquisition function for exploration (Section 4.2.2: Eq. 9 & 10).
        - Copter: local selection-based exploration (Section 4.2.3: Algorithm 2).
        - Copter: global selection-based exploration (Section 4.2.3: Algorithm 3).
    - Rover: mission execution (Section 4.3.2: Algorithm 4).
"""

import copy
import random
import numpy as np

from environment import GRID, AP_LIST, ACTIONS, CARDINALS, clip, TRUE_L, transition_probabilities
from sensor import sensor_beta, bayes_update, observe
from fsa import FSA_ACCEPT, FSA_DEAD, fsa_step
from planning import copter_value_iteration



# Sensing #
def rover_sense(beliefs: dict, pos: tuple) -> None:
    """
    Update beliefs for all cells (in-place) within rover sensor range.

    * AP_r = {A,B,C,D,O}
    * Rover's maximum accuracy is 100% (Section 6.1.1):
        * R = 2
        * M = 0.5
    """
    R, M = 2.0, 0.5
    for r in range(GRID):
        for c in range(GRID):
            beta = sensor_beta(pos, (r, c), R, M)
            if beta <= 0.5:
                continue
            for ap in AP_LIST:
                z = observe(TRUE_L[(r, c)], ap, beta)
                beliefs[(r, c)][ap] = bayes_update(beliefs[(r, c)][ap], z, beta)


def copter_sense(beliefs: dict, pos: tuple) -> None:
    """
    Update obstacle beliefs for all cells (in-place) within copter sensor range.

    * AP_c = {O}
    * Copter's maximum accuracy is 90% (Section 6.1.1):
        * R = 4
        * M = 0.4
    """
    R, M = 4.0, 0.4
    for r in range(GRID):
        for c in range(GRID):
            beta = sensor_beta(pos, (r, c), R, M)
            if beta <= 0.5:
                continue
            z = observe(TRUE_L[(r, c)], 'O', beta)
            beliefs[(r, c)]['O'] = bayes_update(beliefs[(r, c)]['O'], z, beta)



# Acquisition Function #
def entropy_H(b: float) -> float:
    """
    Binary Shannon entropy (Section 4.2.2: Eq. 9):
        * H(b) = -b log2(b) - (1-b) log2(1-b)
            * H = 1 when b = 0.5 (maximum uncertainty).
            * H = 0 when b = 0 or b = 1 (certain).
        - Clipping prevents log(0).
    """
    b = np.clip(b, 1e-9, 1 - 1e-9)
    return -b * np.log2(b) - (1 - b) * np.log2(1 - b)


def acquisition_fn_W(beliefs: dict, b_max: dict, alpha: float) -> dict:
    """
    Acquisition function (Section 4.2.2: Eq. 10):
        * W(x) = H(B(x|=O)) + α * b_max(x)

    Return: 
        * W : dict. keyed by (r, c).
    """
    W = {}
    for r in range(GRID):
        for c in range (GRID):
            W[(r, c)] = entropy_H(beliefs[(r, c)]['O']) + alpha * b_max.get((r, c), 0.0)
    return W



# Copter Exploration (Local) #
def copter_explore_local(beliefs: dict, copter_pos: tuple, b_max: dict, T_c: int,
                         alpha: float = 1.5, record_beliefs: bool = False, vi_steps: int = 80) -> tuple:
    """
    Run the copter's local selection-based exploration for T_c steps (algorithm 2).

    In the local approach, the copter is 'nearsighted'. It re-evaluates its best
    immediate move at every step rather than committing to a distant target.

    Structure:
        1. for ℓ = 0 : T_c - 1 (every step for T_c steps):
            a. Compute W(x) for all neighbours based on latest observations (Eq. 10).
            b. Compute optimal policy μ*_c1 by maximizing the expected acquisition at the next state (Eq. 11).
            c. Apply μ*_c1(x) and move stochastically.
            d. Sense from new position and update beliefs (Eq. 8).
            
    Parameter:
        * beliefs        : environmental belief dict. (mutated in-place).
        * copter_pos     : current copter position.
        * b_max          : rover reachability belief map for acquisition weighting (see planning.py).
        * T_c            : copter exploration phase length.
        * alpha          : acquisition weight.
        * record_beliefs : if True, appends a deepcopy of beliefs at every substep for animation.
        * vi_steps       : maximum number of sweeps.

    Return:
        *
        *
        *
    """
    pos = list(copter_pos)
    substeps = []
    belief_snapshots = []

    for _ in range(T_c):
        W = acquisition_fn_W(beliefs, b_max, alpha)
        
        best_a, best_ew = 0, -np.inf  # 'best_a = 0' prevents crash if no action found
        for a in range(len(ACTIONS)):
            trans = transition_probabilities(pos[0], pos[1], a, p_intended=0.90)
            ew = 0.0
            for cell, p in trans.items():
                ew += p * W.get(cell, 0.0)
            if ew > best_ew:
                best_a, best_ew = a, ew

        dr, dc = ACTIONS[best_a]
        nr, nc = clip(pos[0] + dr, pos[1] + dc)
        if np.random.rand() < 0.90:
            pos = [nr, nc]
        else:
            slips = [list(clip(nr + ddr, nc + ddc))
                    for ddr, ddc in CARDINALS
                    if clip(nr + ddr, nc + ddc) != (nr, nc)]
            pos = random.choice(slips) if slips else [nr, nc]
        
        copter_sense(beliefs, tuple(pos))
        substeps.append(tuple(pos))
        if record_beliefs:
            belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), substeps, belief_snapshots



# Copter Exploration (Global) #
def copter_explore_global(beliefs: dict, copter_pos: tuple, b_max: dict, T_c: int,
                   alpha: float = 1.5, record_beliefs: bool = False, vi_steps: int = 80) -> tuple:
    """    
    Run the copter's global selection-based exploration for T_c steps (algorithm 3).

    In the global approach, the copter commits to a high-value target (x*) and
    navigates there using an optimal policy before re-evaluating the map.

    Structure:
        - NOTE: The outline of Algorithm-3 uses 'ℓ < T_c - 1' but states "iterate the same procedure 
                as [Algorithm-2] for time period T_c", so we also employ 'ℓ < T_c'.
        
        1. OUTER while ℓ < T_c:
            a. Compute W(x) globally over all cells (Eq. 10).
            b. Pick x* = argmax W(x) (Eq. 12).
                - NOTE: Excluding the current cell and converged cells (H < 0.05) to prevent deadlock 
                        where the copter is co-located with the rover (b_max peak) and never moves (i.e., stalls).
            c. Compute optimal policy μ*_c2 to reach x* via value iteration (Eq. 13).
        2. INNER while x ≠ x* AND ℓ < T_c:
            a. Apply μ*_c2(x) and move stochastically until x* is reached or T_c steps are reached.
            b. Sense from new position and update beliefs (Eq. 8).

    Parameter:
        * beliefs        : environmental belief dict. (mutated in-place).
        * copter_pos     : current copter position.
        * b_max          : rover reachability belief map for acquisition weighting (see planning.py).
        * T_c            : copter exploration phase length.
        * alpha          : acquisition weight.
        * record_beliefs : if True, appends a deepcopy of beliefs at every substep for animation.
        * vi_steps       : maximum number of sweeps.

    Return:
        *
        *
        *
    """
    pos = list(copter_pos)
    substeps = []
    belief_snapshots = []
    l = 0  # step counter
    max_copter_vi_sweeps = 0

    while l < T_c:
        W = acquisition_fn_W(beliefs, b_max, alpha)

        # candidates = {k: v for k, v in W.items()
        #               if k != tuple(pos) and entropy_H(beliefs[k]['O']) > 0.05}
        # if not candidates:
        #     candidates = {k: v for k, v in W.items() if k != tuple(pos)}
        # if not candidates:
        #     candidates = W  
        # x_star = max(candidates, key=candidates.get)

        candidates = {k: v for k, v in W.items() if k != tuple(pos)}
        if not candidates:
            break
        else:
            x_star = max(candidates, key=candidates.get)

        # x_star = max(W, key=W.get)  # Using only this results in infinite loop

        # Compute optimal policy to reach x* via value iteration
        copter_policy, copter_vi_sweeps = copter_value_iteration(x_star, vi_steps=vi_steps)
        
        max_copter_vi_sweeps = max(max_copter_vi_sweeps, copter_vi_sweeps)

        # Inner loop: follow policy until x* or T_c steps reached
        while tuple(pos) != x_star and l < T_c:
            if tuple(pos) in copter_policy:
                a = copter_policy[tuple(pos)]
            else:
                print(f"DEBUG: Copter at {pos} has no policy for target {x_star}")
                a = 0
            # a = copter_policy.get(tuple(pos), 0)

            dr, dc = ACTIONS[a]
            nr, nc = clip(pos[0] + dr, pos[1] + dc)

            if np.random.rand() < 0.90:
                pos = [nr, nc]
            else:
                slips = [list(clip(nr + ddr, nc + ddc))
                         for ddr, ddc in CARDINALS
                         if clip(nr + ddr, nc + ddc) != (nr, nc)]
                pos = random.choice(slips) if slips else [nr, nc]

            copter_sense(beliefs, tuple(pos))
            substeps.append(tuple(pos))
            if record_beliefs:
                belief_snapshots.append(copy.deepcopy(beliefs))

            l += 1
    print(f"  Copter value iteration maximum sweeps used {max_copter_vi_sweeps}/{vi_steps}.")

    return tuple(pos), substeps, belief_snapshots



# Rover Mission Execution #
def rover_execute(beliefs: dict, rover_pos: tuple, rover_q: int,
                  rover_policy: dict, T_r: int,
                  record_beliefs: bool = False) -> tuple:
    """

    """
    pos = list(rover_pos)
    q = rover_q
    substeps = []
    belief_snapshots = []

    q_prev = q
    q = fsa_step(q, TRUE_L[tuple(pos)])
    if q != q_prev:
        print(f"    FSA transition: q={q_prev} → q={q} at pos={tuple(pos)}, labels={TRUE_L[tuple(pos)]}")


    rover_sense(beliefs, tuple(pos))
    substeps.append((tuple(pos), q))
    if record_beliefs:
        belief_snapshots.append(copy.deepcopy(beliefs))
    if q in FSA_ACCEPT or q == FSA_DEAD:
        return tuple(pos), q, substeps, belief_snapshots

    for _ in range(T_r):
        if q in FSA_ACCEPT or q == FSA_DEAD:
            break

        a = rover_policy.get((pos[0], pos[1], q), 1)
        dr, dc = ACTIONS[a]
        nr, nc = clip(pos[0] + dr, pos[1] + dc)

        if np.random.rand() < 0.95:
            pos = [nr, nc]
            # Debug: Log when rover visits cells near D
            if tuple(pos) in [(7,5), (7,6), (6,6), (8,6)]:
                print(f"      Rover at {tuple(pos)}, q={q}, looking for={'D' if q==2 else 'other'}")
        else:
            slips = [list(clip(nr + ddr, nc + ddc))
                     for ddr, ddc in CARDINALS
                     if clip(nr + ddr, nc + ddc) != (nr, nc)]
            pos = random.choice(slips) if slips else [nr, nc]

        q_prev = q
        q = fsa_step(q, TRUE_L[tuple(pos)])
        if q != q_prev:
            print(f"    FSA transition: q={q_prev} → q={q} at pos={tuple(pos)}, labels={TRUE_L[tuple(pos)]}")

        rover_sense(beliefs, tuple(pos))
        substeps.append((tuple(pos), q))
        if record_beliefs:
            belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), q, substeps, belief_snapshots