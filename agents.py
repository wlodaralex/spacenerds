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

from environment import GRID, AP_LIST, ACTIONS, CARDINALS, clip, TRUE_L
from sensor import sensor_beta, bayes_update, observe
from fsa import FSA_ACCEPT, FSA_DEAD, fsa_step, compute_B_en



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
        1. Select x* = argmax W(x), excluding the current cell and already-
           converged cells (H < 0.05). Exclusion prevents the deadlock where
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
            best_a, best_d = 1, float('inf')
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = clip(pos[0] + dr, pos[1] + dc)
                d = abs(nr - x_star[0]) + abs(nc - x_star[1])
                if d < best_d:
                    best_d, best_a = d, a
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



# Rover Mission Execution #
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