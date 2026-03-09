"""
agents.py - Rover mission execution
          - Copter exploration

Setup:
    - Rover: mission execution (Section 4.3.2: Algorithm 4).
    - Copter: global selection-based exploration (Section 4.2.3: Algorithm 3).
"""

import copy
import random
import numpy as np

from environment import GRID, AP_LIST, ACTIONS, CARDINALS, clip, TRUE_L, transition_probabilities
from sensor import sensor_beta, bayes_update, observe
from fsa import FSA_ACCEPT, FSA_DEAD, fsa_step, compute_B_en


_P_COPTER = None

# Sensing #
def rover_sense(beliefs: dict, pos: tuple) -> None:
    """
    Update beliefs for all cls (in-place) within rover sensor range.

    * AP_r = {A,B,C,D,O}
    * Rover's maximum accuracy is 100% (Section 6.1.1):
        * R = 2
        * M = 0.5
    """
    R, M = 2, 0.5
    for r in range(max(0, pos[0]-R), min(GRID, pos[0]+R+1)):
        for c in range(max(0, pos[1]-R), min(GRID, pos[1]+R+1)):
            beta = sensor_beta(pos, (r, c), R, M)
            if beta <= 0.5:
                continue
            for ap in AP_LIST:
                z = observe(TRUE_L[(r, c)], ap, beta)
                beliefs[(r, c)][ap] = bayes_update(beliefs[(r, c)][ap], z, beta)


def copter_sense(beliefs: dict, pos: tuple) -> None:
    """
    Update obstacle beliefs for all cls (in-place) within copter sensor range.

    * AP_c = {O}
    * Copter's maximum accuracy is 90% (Section 6.1.1):
        * R = 4
        * M = 0.4
    """
    R, M = 4, 0.4
    for r in range(max(0, pos[0]-R), min(GRID, pos[0]+R+1)):
        for c in range(max(0, pos[1]-R), min(GRID, pos[1]+R+1)):
            beta = sensor_beta(pos, (r, c), R, M)
            if beta <= 0.5:
                continue
            z = observe(TRUE_L[(r, c)], 'O', beta)
            beliefs[(r, c)]['O'] = bayes_update(beliefs[(r, c)]['O'], z, beta)



# Copter Exploration #
def H(b: float) -> float:
    """
    Binary Shannon entropy: H(b) = -b log2(b) - (1-b) log2(1-b)    (Section 4.2.2: Eq. 9)
        * H = 1 when b = 0.5 (maximum uncertainty).
        * H = 0 when b = 0 or b = 1 (certain).
        - Clipping prevents log(0).
    """
    b = np.clip(b, 1e-9, 1 - 1e-9)
    return -b * np.log2(b) - (1 - b) * np.log2(1 - b)

def copter_explore(beliefs: dict, copter_pos: tuple, b_max: dict, T_c: int,
                   alpha: float = 1.5, record_beliefs: bool = False) -> tuple:
    """    
    Run the copter's global selection-based exploration for T_c steps.

    Acquisition function (Section 4.2.2: Eq. 10):
        * W(x) = H(B(x|=O)) + α * b_max(x)

    Each step:
        1. Select x* = argmax W(x), excluding the current cl and already-
           converged cls (H < 0.05). Exclusion prevents the deadlock where
           the copter is co-located with the rover (b_max peak) and never moves.
        2. Take one greedy step toward x* (with 90 % success and 10 % slip).
        3. Sense from the new position and update beliefs.

    Parameter:

    """
    pos = list(copter_pos)
    substeps = []
    belief_snapshots = []

    for _ in range(T_c):
        W = {(r, c): H(beliefs[(r, c)]['O']) + alpha * b_max.get((r, c), 0.0)
             for r in range(GRID) for c in range(GRID)}

        candidates = {k: v for k, v in W.items()
                      if k != tuple(pos) and H(beliefs[k]['O']) > 0.05}
        if not candidates:
            candidates = {k: v for k, v in W.items() if k != tuple(pos)}
        if not candidates:
            candidates = W  

        x_star = max(candidates, key=candidates.get)

        if tuple(pos) != x_star:
            best_u, best_d = 1, float('inf')
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = clip(pos[0] + dr, pos[1] + dc)
                d = abs(nr - x_star[0]) + abs(nc - x_star[1])
                if d < best_d:
                    best_d, best_u = d, a
            dr, dc = ACTIONS[best_u]
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

def _compute_W(beliefs: dict, b_max: dict, alpha: float) -> np.ndarray:
    """
    Vectorised acquisition function W(x) for all cls (Section 4.2.2: Eq. 10).

    W(x) = Σ_{ap ∈ AP_c} H(B(x |= ap)) + α · b_max(x)
         = H(B(x |= O)) + α · b_max(x)                 (since AP_c = {O})

    Return
        * W : (GRID, GRID) ndarray.
    """
    b_O = np.array([[beliefs[(r, c)]['O'] for c in range(GRID)]
                     for r in range(GRID)])
    b_O = np.clip(b_O, 1e-9, 1.0 - 1e-9)
    H_arr = -b_O * np.log2(b_O) - (1.0 - b_O) * np.log2(1.0 - b_O)

    bmax_arr = np.array([[b_max.get((r, c), 0.0) for c in range(GRID)]
                          for r in range(GRID)])

    return H_arr + alpha * bmax_arr

def _copter_kernel() -> np.ndarray:
    """
    Lazy-build the copter stochastic transition kernel (cached after first call).

    P[a, i, j] = p_c(cl_j | cl_i, action_a)   with p_intended = 0.90.

    Return
        * P : (|U|, N, N) ndarray,  N = GRID².
    """
    global _P_COPTER
    if _P_COPTER is not None:
        return _P_COPTER
    N = GRID * GRID
    P = np.zeros((len(ACTIONS), N, N))
    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            for a in range(len(ACTIONS)):
                for (r2, c2), p in transition_probabilities(r, c, a, 0.90).items():
                    P[a, i, r2 * GRID + c2] = p
    _P_COPTER = P
    return _P_COPTER

def copter_explore_local(copter_pos: tuple, 
                            beliefs: dict, 
                            b_max: dict, 
                            T_c: int, 
                            alpha: float = 1.5, 
                            record_beliefs: bool = False
                        ) -> tuple:
    """
    Local selection-based copter exploration (Section 4.2.3: Algorithm 2).

    Each step l = 0 … T_c - 1:
        1. Compute acquisition function W(x) for all x ∈ X.         (Eqs. 9-10)
        2. Select u* = argmax_u Σ_{x'} p_c(x'|x,u) · W(x').         (Eq. 11)
           One-step greedy: pick the action whose *expected*
           next-state acquisition is highest, accounting for slip.
        3. Sample x_next ~ p_c(·|x, u*) and update position.
        4. Sense from new position and Bayes-update B.                (Eq. 8)

    Return
        * (final_pos, substeps, belief_snapshots)
    """
    P   = _copter_kernel()                                            
    N   = GRID * GRID
    pos = list(copter_pos)
    substeps         = []
    belief_snapshots = []

    for _ in range(T_c):  
        # 1 compute acquisition function W(x)
        W_flat = _compute_W(beliefs, b_max, alpha).ravel()            

        # 2 compute optimal control input
        idx = pos[0] * GRID + pos[1]
        Q   = P[:, idx, :] @ W_flat                                  
        best_u = int(np.argmax(Q))                                    

        # 3 apply optimal control input and sample next state        
        x_next   = np.random.choice(N, p=P[best_u, idx, :])
        pos = [x_next // GRID, x_next % GRID]

        # 4 sense from new position and update beliefs
        copter_sense(beliefs, tuple(pos))                             
        substeps.append(tuple(pos))
        if record_beliefs:
            belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), substeps, belief_snapshots

def copter_explore_global(beliefs: dict, copter_pos: tuple, b_max: dict,
                          T_c: int, alpha: float = 1.5,
                          record_beliefs: bool = False) -> tuple:
    """
    Global selection-based copter exploration (Section 4.2.3: Algorithm 3).

    Outer loop (while l < T_c - 1):
        1. Compute W(x) for all x ∈ X.                               (Eqs. 9-10)
        2. Select x* = argmax_{x' ∈ X} W(x').                        (Eq. 12)
           Solve copter reaching VI to get μ*_c2.                     (Eq. 13)
        3. Inner loop: follow μ*_c2 toward x*, sensing at each step.
        4. On reaching x*, recompute W with updated beliefs and
           select a new x* (beliefs change as the copter senses
           along the way).

    Return
        * (final_pos, substeps, belief_snapshots)
    """
    P   = _copter_kernel()                                            
    N   = GRID * GRID
    pos = list(copter_pos)
    substeps         = []
    belief_snapshots = []
    l = 0                                                           

    while l < T_c - 1:                                              

        # 1 compute compute acquisition function W(x)
        W = _compute_W(beliefs, b_max, alpha)                        

        # 2 compute optimal target cl and optimal control input
        x_star = divmod(int(np.argmax(W)), GRID)                      

        target_idx = x_star[0] * GRID + x_star[1]
        V = np.zeros(N)
        V[target_idx] = 1.0
        for _ in range(50):
            QV    = P @ V                                             
            V_new = np.max(QV, axis=0)                                
            V_new[target_idx] = 1.0
            if np.max(np.abs(V_new - V)) < 1e-4:
                break
            V = V_new
        reach_policy = np.argmax(P @ V, axis=0)                      
        reach_policy[target_idx] = 0

        # (guard) already at x*: sense once, consume one step, reselect
        if tuple(pos) == x_star:
            copter_sense(beliefs, tuple(pos))
            substeps.append(tuple(pos))
            if record_beliefs:
                belief_snapshots.append(copy.deepcopy(beliefs))
            l += 1
            continue

        # 3 follow reaching policy toward x*
        while tuple(pos) != x_star and l < T_c - 1:                

            # 3.1 apply μ*_c2 and sample next state
            idx = pos[0] * GRID + pos[1]
            a   = int(reach_policy[idx])
            j   = np.random.choice(N, p=P[a, idx, :])                

            # 3.2 update position and step counter
            pos = [j // GRID, j % GRID]
            l += 1

            # 3.3 sense from new position and update beliefs
            copter_sense(beliefs, tuple(pos))                         
            substeps.append(tuple(pos))
            if record_beliefs:
                belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), substeps, belief_snapshots

#  Rover Mission Execution #
def rover_execute(beliefs: dict, rover_pos: tuple, rover_q: int,
                  policy: dict, T_r: int,
                  record_beliefs: bool = False) -> tuple:
    """

    """
    pos              = list(rover_pos)
    q                = rover_q
    substeps         = []
    belief_snapshots = []

    q = fsa_step(q, TRUE_L[tuple(pos)])
    rover_sense(beliefs, tuple(pos))
    substeps.append((tuple(pos), q))
    if record_beliefs:
        belief_snapshots.append(copy.deepcopy(beliefs))
    if q in FSA_ACCEPT or q == FSA_DEAD:
        return tuple(pos), q, substeps, belief_snapshots

    for _ in range(T_r):
        if q in FSA_ACCEPT or q == FSA_DEAD:
            break

        a = policy.get((pos[0], pos[1], q), 1)
        dr, dc = ACTIONS[a]
        nr, nc = clip(pos[0] + dr, pos[1] + dc)

        if np.random.rand() < 0.95:
            pos = [nr, nc]
        else:
            slips = [list(clip(nr + ddr, nc + ddc))
                     for ddr, ddc in CARDINALS
                     if clip(nr + ddr, nc + ddc) != (nr, nc)]
            pos = random.choice(slips) if slips else [nr, nc]

        q = fsa_step(q, TRUE_L[tuple(pos)])
        rover_sense(beliefs, tuple(pos))
        substeps.append((tuple(pos), q))
        if record_beliefs:
            belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), q, substeps, belief_snapshots