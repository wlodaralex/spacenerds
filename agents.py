<<<<<<< HEAD
"""rover execution, sensing, and copter exploration policies."""

import copy
import math
import random

import numpy as np

from environment import GRID, AP_LIST, ACTIONS, CARDINALS, clip, TRUE_L, transition_probabilities
from sensor import sensor_beta, bayes_update, observe
from fsa import FSA_ACCEPT, FSA_DEAD, fsa_step, compute_B_en


_P_COPTER = None
_UTILITY_ROLLOUT_CACHE = {}

def rover_sense(beliefs: dict, pos: tuple) -> None:
    """update all local beliefs visible to the rover."""
    R, M = 2, 0.5
    for r in range(max(0, pos[0]-R), min(GRID, pos[0]+R+1)):
        for c in range(max(0, pos[1]-R), min(GRID, pos[1]+R+1)):
            # formula: quartic sensor dome
            beta = sensor_beta(pos, (r, c), R, M)
            if beta <= 0.5:
                continue
            for ap in AP_LIST:
                # formula: bernoulli bayes flip
                z = observe(TRUE_L[(r, c)], ap, beta)
                beliefs[(r, c)][ap] = bayes_update(beliefs[(r, c)][ap], z, beta)

def copter_sense(beliefs: dict, pos: tuple) -> None:
    """update local obstacle beliefs visible to the copter."""
    R, M = 4, 0.4
    for r in range(max(0, pos[0]-R), min(GRID, pos[0]+R+1)):
        for c in range(max(0, pos[1]-R), min(GRID, pos[1]+R+1)):
            # formula: quartic sensor dome
            beta = sensor_beta(pos, (r, c), R, M)
            if beta <= 0.5:
                continue
            # step 1 the copter only updates o, not a/b/c/d
            z = observe(TRUE_L[(r, c)], 'O', beta)
            beliefs[(r, c)]['O'] = bayes_update(beliefs[(r, c)]['O'], z, beta)

def H(b: float) -> float:
    """binary shannon entropy."""
    # formula: binary fog score
    b = np.clip(b, 1e-9, 1 - 1e-9)
    return -b * np.log2(b) - (1 - b) * np.log2(1 - b)

def copter_explore(beliefs: dict, copter_pos: tuple, b_max: dict, T_c: int,
                   alpha: float = 1.5, record_beliefs: bool = False) -> tuple:
    """legacy single-target global helper kept for reference."""
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
    """return the Hashimoto acquisition score over the whole grid."""
    # formula: hashimoto scout field
    # w(x) = h(b_o(x)) + alpha * b_max(x)
    b_O = np.array([[beliefs[(r, c)]['O'] for c in range(GRID)]
                     for r in range(GRID)])
    b_O = np.clip(b_O, 1e-9, 1.0 - 1e-9)
    H_arr = -b_O * np.log2(b_O) - (1.0 - b_O) * np.log2(1.0 - b_O)

    bmax_arr = np.array([[b_max.get((r, c), 0.0) for c in range(GRID)]
                          for r in range(GRID)])

    return H_arr + alpha * bmax_arr

def _copter_kernel() -> np.ndarray:
    """build and cache the stochastic copter kernel for local/global modes."""
    global _P_COPTER
    if _P_COPTER is not None:
        return _P_COPTER
    N = GRID * GRID
    P = np.zeros((len(ACTIONS), N, N))
    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            for a in range(len(ACTIONS)):
                # formula: stochastic slip kernel
                for (r2, c2), p in transition_probabilities(r, c, a, 0.90).items():
                    P[a, i, r2 * GRID + c2] = p
    _P_COPTER = P
    return _P_COPTER

def copter_explore_local(beliefs: dict, copter_pos: tuple, b_max: dict,
                         T_c: int, alpha: float = 1.5,
                         record_beliefs: bool = False) -> tuple:
    """one-step greedy copter policy with stochastic motion."""
    P   = _copter_kernel()                                            
    N   = GRID * GRID
    pos = list(copter_pos)
    substeps         = []
    belief_snapshots = []

    for _ in range(T_c):  
        # step 1 score all cells with W(x)
        W_flat = _compute_W(beliefs, b_max, alpha).ravel()            

        # step 2 choose the best expected next action
        # formula: one-step lookahead
        # q(a) = sum_x' p(x' | x, a) w(x')
        idx = pos[0] * GRID + pos[1]
        Q   = P[:, idx, :] @ W_flat                                  
        best_u = int(np.argmax(Q))                                    

        # step 3 sample the next state under the copter kernel
        x_next   = np.random.choice(N, p=P[best_u, idx, :])
        pos = [x_next // GRID, x_next % GRID]

        # step 4 sense from the realized position
        copter_sense(beliefs, tuple(pos))                             
        substeps.append(tuple(pos))
        if record_beliefs:
            belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), substeps, belief_snapshots

def copter_explore_global(beliefs: dict, copter_pos: tuple, b_max: dict,
                          T_c: int, alpha: float = 1.5,
                          record_beliefs: bool = False) -> tuple:
    """global target-selection copter policy with stochastic motion."""
    P   = _copter_kernel()                                            
    N   = GRID * GRID
    pos = list(copter_pos)
    substeps         = []
    belief_snapshots = []
    l = 0                                                           

    while l < T_c - 1:                                              
        # step 1 choose the current best target cell
        # formula: greedy beacon pick
        W = _compute_W(beliefs, b_max, alpha)                        
        x_star = divmod(int(np.argmax(W)), GRID)                      

        # step 1b solve the finite-horizon reaching value iteration
        # formula: reach-the-beacon bellman
        target_idx = x_star[0] * GRID + x_star[1]
        V = np.zeros(N)
        V[target_idx] = 1.0
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            for _ in range(50):
                QV    = P @ V
                V_new = np.clip(np.max(QV, axis=0), 0.0, 1.0)
                V_new[target_idx] = 1.0
                if np.max(np.abs(V_new - V)) < 1e-4:
                    break
                V = V_new
            reach_policy = np.argmax(P @ V, axis=0)
        reach_policy[target_idx] = 0

        # step 1a if already at the target, sense and reselect
        if tuple(pos) == x_star:
            copter_sense(beliefs, tuple(pos))
            substeps.append(tuple(pos))
            if record_beliefs:
                belief_snapshots.append(copy.deepcopy(beliefs))
            l += 1
            continue

        # step 2 follow the reaching policy toward that target
        while tuple(pos) != x_star and l < T_c - 1:                

            idx = pos[0] * GRID + pos[1]
            a   = int(reach_policy[idx])
            # formula: stochastic slip kernel
            j   = np.random.choice(N, p=P[a, idx, :])                

            pos = [j // GRID, j % GRID]
            l += 1

            # step 3 sense from the realized position
            copter_sense(beliefs, tuple(pos))                         
            substeps.append(tuple(pos))
            if record_beliefs:
                belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), substeps, belief_snapshots

def _copter_move(pos: list, action: int) -> list:
    """Execute copter action: deterministic motion (formulation §2/§5)."""
    # step 1 utility mode uses the clipped deterministic motion map
    dr, dc = ACTIONS[action]
    nr, nc = clip(pos[0] + dr, pos[1] + dc)
    return [nr, nc]

def _utility_rollout_cache(remaining: int):
    """cache all action sequences and motion deltas for one horizon.

    for a remaining horizon H, the exhaustive copter solver evaluates M = 8^H
    discrete action sequences. this cache materializes that action space once:

    - sequence_actions[m, t] is the t-th action in sequence m
    - row_delta_by_sequence and col_delta_by_sequence store the per-step motion
    - step_length_by_sequence stores the euclidean move length per step

    the vectorized rollout then applies the same recurrence to every candidate
    sequence in parallel:

        r_{m,t+1} = clip(r_{m,t} + delta_r[m,t])
        c_{m,t+1} = clip(c_{m,t} + delta_c[m,t])

    so the later scoring step can batch all M candidates without changing the
    underlying exhaustive search.
    """
    cached_rollout = _UTILITY_ROLLOUT_CACHE.get(remaining)
    if cached_rollout is not None:
        return cached_rollout

    from itertools import product as iproduct

    sequence_actions = np.array(
        list(iproduct(range(1, len(ACTIONS)), repeat=remaining)),
        dtype=np.int8,
    )
    action_row_delta = np.array(
        [dr for dr, _ in ACTIONS],
        dtype=np.int8,
    )
    action_col_delta = np.array(
        [dc for _, dc in ACTIONS],
        dtype=np.int8,
    )
    action_step_length = np.hypot(
        action_row_delta.astype(np.float64),
        action_col_delta.astype(np.float64),
    )

    cached_rollout = {
        'sequence_actions': sequence_actions,
        'row_delta_by_sequence': action_row_delta[sequence_actions],
        'col_delta_by_sequence': action_col_delta[sequence_actions],
        'step_length_by_sequence': action_step_length[sequence_actions],
        'sequence_index': np.arange(len(sequence_actions)),
    }
    _UTILITY_ROLLOUT_CACHE[remaining] = cached_rollout
    return cached_rollout

def copter_explore_utility(beliefs: dict, copter_pos: tuple, b_max: dict,
                           T_c: int, alpha: float = 1.5,
                           record_beliefs: bool = False,
                           lambda_: float = 0.0,
                           rover_pos: tuple = (0, 0),
                           energy: float = float('inf')) -> tuple:
    """deterministic exhaustive copter search over the remaining horizon.

    the optimized implementation is still the same exact maximization over the
    discrete action space. for the remaining horizon H, it scores all
    M = 8^H candidate sequences by

        score_m = alpha I_m - lambda C_c,m

    in parallel. if sequence m visits cells x_{m,1}, ..., x_{m,H}, then

        C_c,m = sum_t 1[moved_{m,t}] * ||delta_{m,t}||_2
                + rho * ||x_{m,H} - x_r||_2

    and

        I_m = sum_x 1[x is newly seen by sequence m] * H(B_O(x))

    the boolean matrix seen_by_sequence[m, x] preserves the exact old semantics
    that repeated visits within one sequence only count once. the solver still
    returns the first action of the best feasible sequence, but the rollout and
    scoring are vectorized across all candidates.
    """
    pos = list(copter_pos)
    visited_global = {tuple(pos)}
    substeps = []
    belief_snapshots = []
    RHO = 1.0

    for step in range(T_c):
        remaining = T_c - step

        # step 1 cache the entropy reward of unseen cells
        # formula: entropy scout bonus
        # only cells not yet visited in this epoch can contribute new i(a_c)
        obstacle_belief_flat = np.array(
            [beliefs[(r, c)]['O'] for r in range(GRID) for c in range(GRID)],
            dtype=np.float64,
        )
        unseen_cell_mask = np.ones(GRID * GRID, dtype=bool)
        for r, c in visited_global:
            unseen_cell_mask[r * GRID + c] = False
        entropy_flat = np.zeros(GRID * GRID, dtype=np.float64)
        unseen_obstacle_belief = np.clip(
            obstacle_belief_flat[unseen_cell_mask],
            1e-9,
            1.0 - 1e-9,
        )
        entropy_flat[unseen_cell_mask] = (
            -unseen_obstacle_belief * np.log2(unseen_obstacle_belief)
            - (1.0 - unseen_obstacle_belief) * np.log2(1.0 - unseen_obstacle_belief)
        )

        # step 2 score staying for the rest of the epoch
        # formula: stay-put baseline
        d_return_now = abs(pos[0] - rover_pos[0]) + abs(pos[1] - rover_pos[1])
        best_score = -lambda_ * RHO * d_return_now
        best_action = 0

        # step 3 score every remaining action sequence
        # formula: batch rollout over m = 8^h candidates
        rollout_cache = _utility_rollout_cache(remaining)
        sequence_actions = rollout_cache['sequence_actions']
        row_delta_by_sequence = rollout_cache['row_delta_by_sequence']
        col_delta_by_sequence = rollout_cache['col_delta_by_sequence']
        step_length_by_sequence = rollout_cache['step_length_by_sequence']
        sequence_index = rollout_cache['sequence_index']

        row_position = np.full(len(sequence_actions), pos[0], dtype=np.int16)
        col_position = np.full(len(sequence_actions), pos[1], dtype=np.int16)
        flight_cost = np.zeros(len(sequence_actions), dtype=np.float64)
        information_gain = np.zeros(len(sequence_actions), dtype=np.float64)
        seen_by_sequence = np.zeros(
            (len(sequence_actions), GRID * GRID),
            dtype=bool,
        )

        for horizon_index in range(remaining):
            # step 3a apply the deterministic motion recurrence to every sequence
            next_row_position = np.clip(
                row_position + row_delta_by_sequence[:, horizon_index],
                0,
                GRID - 1,
            )
            next_col_position = np.clip(
                col_position + col_delta_by_sequence[:, horizon_index],
                0,
                GRID - 1,
            )
            moved = (
                (next_row_position != row_position)
                | (next_col_position != col_position)
            )

            # formula: copter trip ledger
            flight_cost += moved * step_length_by_sequence[:, horizon_index]

            row_position = next_row_position
            col_position = next_col_position
            cell_index = row_position * GRID + col_position

            # formula: set-once scout bonus
            # repeated visits in one sequence count once, matching the old set
            newly_seen = (
                unseen_cell_mask[cell_index]
                & ~seen_by_sequence[sequence_index, cell_index]
            )
            information_gain += newly_seen * entropy_flat[cell_index]
            seen_by_sequence[sequence_index, cell_index] = True

        # formula: frozen-return leash
        return_cost = RHO * np.hypot(
            row_position - rover_pos[0],
            col_position - rover_pos[1],
        )
        total_cost = flight_cost + return_cost
        feasible_sequence = total_cost <= energy
        if np.any(feasible_sequence):
            # formula: copter surrogate utility
            # score = alpha * i - lambda * c_c
            score_by_sequence = alpha * information_gain - lambda_ * total_cost
            feasible_score = np.where(
                feasible_sequence,
                score_by_sequence,
                -np.inf,
            )
            best_sequence_index = int(np.argmax(feasible_score))
            candidate_score = float(feasible_score[best_sequence_index])
            if candidate_score > best_score:
                best_score = candidate_score
                best_action = int(sequence_actions[best_sequence_index, 0])

        # step 4 staying means the epoch is done
        if best_action == 0:
            for _ in range(remaining):
                copter_sense(beliefs, tuple(pos))
                substeps.append(tuple(pos))
                if record_beliefs:
                    belief_snapshots.append(copy.deepcopy(beliefs))
            break

        # step 5 execute the first move of the best sequence
        pos = _copter_move(pos, best_action)
        visited_global.add(tuple(pos))
        copter_sense(beliefs, tuple(pos))
        substeps.append(tuple(pos))
        if record_beliefs:
            belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), substeps, belief_snapshots

def rover_execute(beliefs: dict, rover_pos: tuple, rover_q: int,
                  policy: dict, T_r: int,
                  record_beliefs: bool = False) -> tuple:
    """execute the rover policy for at most t_r steps."""
    pos              = list(rover_pos)
    q                = rover_q
    substeps         = [(tuple(pos), q)]
    belief_snapshots = []

    for _ in range(T_r):
        if q in FSA_ACCEPT or q == FSA_DEAD:
            break

        # step 1 read the committed action from the current rover policy
        a = policy.get((pos[0], pos[1], q), 1)
        dr, dc = ACTIONS[a]
        nr, nc = clip(pos[0] + dr, pos[1] + dc)

        # step 2 apply deterministic rover motion
        pos = [nr, nc]

        # step 3 update the realized automaton state with true labels
        q = fsa_step(q, TRUE_L[tuple(pos)])

        # step 4 fuse the rover's local observation into the shared belief map
        rover_sense(beliefs, tuple(pos))
        substeps.append((tuple(pos), q))
        if record_beliefs:
            belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), q, substeps, belief_snapshots


def compute_u_c(V: dict, beliefs: dict, rover_pos: tuple, rover_q: int,
                copter_substeps: list, alpha: float, lambda_: float,
                rho: float = 1.0) -> float:
    """compute the logged copter utility used by the acceptance gate."""
    # approximation note: this uses the planner value v at the current rover
    # state as a proxy for the shared mission term g. that matches the shipped
    # surrogate gate, but is not a full re-evaluation of the counterfactual
    # rover best response for each copter trajectory.

    # formula: mission logscore proxy
    V_phi = V.get((rover_pos[0], rover_pos[1], rover_q), 0.0)

    # formula: entropy scout bonus
    observed = set(copter_substeps)
    I = sum(H(beliefs[pos]['O']) for pos in observed)

    # formula: copter trip ledger
    C_c = 0.0
    for k in range(len(copter_substeps) - 1):
        dr = copter_substeps[k][0] - copter_substeps[k+1][0]
        dc = copter_substeps[k][1] - copter_substeps[k+1][1]
        C_c += math.sqrt(dr*dr + dc*dc)
    if copter_substeps:
        dr = copter_substeps[-1][0] - rover_pos[0]
        dc = copter_substeps[-1][1] - rover_pos[1]
        C_c += rho * math.sqrt(dr*dr + dc*dc)

    # formula: mission logscore
    G = np.log(np.clip(V_phi, 1e-9, 1.0))

    # formula: copter utility ledger
    return G + alpha * I - lambda_ * C_c
=======
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
import math
# import random

import numpy as np

from environment import GRID, AP_LIST, ACTIONS, CARDINALS, clip, TRUE_L, transition_probabilities
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
                         alpha: float = 1.5, record_beliefs: bool = False, vi_steps: int = 80,
                         verbose: bool = True) -> tuple:
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
def copter_explore(beliefs: dict, copter_pos: tuple, b_max: dict, T_c: int,
                   alpha: float = 1.5, record_beliefs: bool = False) -> tuple:
    """legacy single-target global helper kept for reference."""
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
            pos = slips[np.random.randint(len(slips))] if slips else [nr, nc]
        
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



# Copter Exploration (Global) #
def copter_explore_global(beliefs: dict, copter_pos: tuple, b_max: dict, T_c: int,
                   alpha: float = 1.5, record_beliefs: bool = False, vi_steps: int = 80,
                   verbose: bool = True) -> tuple:
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
                pos = slips[np.random.randint(len(slips))] if slips else [nr, nc]

            copter_sense(beliefs, tuple(pos))
            substeps.append(tuple(pos))
            if record_beliefs:
                belief_snapshots.append(copy.deepcopy(beliefs))

            l += 1
    if verbose:
        print(f"  Copter value iteration maximum sweeps used {max_copter_vi_sweeps}/{vi_steps}.")

    return tuple(pos), substeps, belief_snapshots



# Rover Mission Execution #
def _compute_W(beliefs: dict, b_max: dict, alpha: float) -> np.ndarray:
    """return the Hashimoto acquisition score over the whole grid."""
    # formula: hashimoto scout field
    # w(x) = h(b_o(x)) + alpha * b_max(x)
    b_O = np.array([[beliefs[(r, c)]['O'] for c in range(GRID)]
                     for r in range(GRID)])
    b_O = np.clip(b_O, 1e-9, 1.0 - 1e-9)
    H_arr = -b_O * np.log2(b_O) - (1.0 - b_O) * np.log2(1.0 - b_O)

    bmax_arr = np.array([[b_max.get((r, c), 0.0) for c in range(GRID)]
                          for r in range(GRID)])

    return H_arr + alpha * bmax_arr

def _copter_kernel() -> np.ndarray:
    """build and cache the stochastic copter kernel for local/global modes."""
    global _P_COPTER
    if _P_COPTER is not None:
        return _P_COPTER
    N = GRID * GRID
    P = np.zeros((len(ACTIONS), N, N))
    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            for a in range(len(ACTIONS)):
                # formula: stochastic slip kernel
                for (r2, c2), p in transition_probabilities(r, c, a, 0.90).items():
                    P[a, i, r2 * GRID + c2] = p
    _P_COPTER = P
    return _P_COPTER

def copter_explore_local(beliefs: dict, copter_pos: tuple, b_max: dict,
                         T_c: int, alpha: float = 1.5,
                         record_beliefs: bool = False) -> tuple:
    """one-step greedy copter policy with stochastic motion."""
    P   = _copter_kernel()                                            
    N   = GRID * GRID
    pos = list(copter_pos)
    substeps         = []
    belief_snapshots = []

    for _ in range(T_c):  
        # step 1 score all cells with W(x)
        W_flat = _compute_W(beliefs, b_max, alpha).ravel()            

        # step 2 choose the best expected next action
        # formula: one-step lookahead
        # q(a) = sum_x' p(x' | x, a) w(x')
        idx = pos[0] * GRID + pos[1]
        Q   = P[:, idx, :] @ W_flat                                  
        best_u = int(np.argmax(Q))                                    

        # step 3 sample the next state under the copter kernel
        x_next   = np.random.choice(N, p=P[best_u, idx, :])
        pos = [x_next // GRID, x_next % GRID]

        # step 4 sense from the realized position
        copter_sense(beliefs, tuple(pos))                             
        substeps.append(tuple(pos))
        if record_beliefs:
            belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), substeps, belief_snapshots

def copter_explore_global(beliefs: dict, copter_pos: tuple, b_max: dict,
                          T_c: int, alpha: float = 1.5,
                          record_beliefs: bool = False) -> tuple:
    """global target-selection copter policy with stochastic motion."""
    P   = _copter_kernel()                                            
    N   = GRID * GRID
    pos = list(copter_pos)
    substeps         = []
    belief_snapshots = []
    l = 0                                                           

    while l < T_c - 1:                                              
        # step 1 choose the current best target cell
        # formula: greedy beacon pick
        W = _compute_W(beliefs, b_max, alpha)                        
        x_star = divmod(int(np.argmax(W)), GRID)                      

        # step 1b solve the finite-horizon reaching value iteration
        # formula: reach-the-beacon bellman
        target_idx = x_star[0] * GRID + x_star[1]
        V = np.zeros(N)
        V[target_idx] = 1.0
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            for _ in range(50):
                QV    = P @ V
                V_new = np.clip(np.max(QV, axis=0), 0.0, 1.0)
                V_new[target_idx] = 1.0
                if np.max(np.abs(V_new - V)) < 1e-4:
                    break
                V = V_new
            reach_policy = np.argmax(P @ V, axis=0)
        reach_policy[target_idx] = 0

        # step 1a if already at the target, sense and reselect
        if tuple(pos) == x_star:
            copter_sense(beliefs, tuple(pos))
            substeps.append(tuple(pos))
            if record_beliefs:
                belief_snapshots.append(copy.deepcopy(beliefs))
            l += 1
            continue

        # step 2 follow the reaching policy toward that target
        while tuple(pos) != x_star and l < T_c - 1:                

            idx = pos[0] * GRID + pos[1]
            a   = int(reach_policy[idx])
            # formula: stochastic slip kernel
            j   = np.random.choice(N, p=P[a, idx, :])                

            pos = [j // GRID, j % GRID]
            l += 1

            # step 3 sense from the realized position
            copter_sense(beliefs, tuple(pos))                         
            substeps.append(tuple(pos))
            if record_beliefs:
                belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), substeps, belief_snapshots

def _copter_move(pos: list, action: int) -> list:
    """Execute copter action: deterministic motion (formulation §2/§5)."""
    # step 1 utility mode uses the clipped deterministic motion map
    dr, dc = ACTIONS[action]
    nr, nc = clip(pos[0] + dr, pos[1] + dc)
    return [nr, nc]

def _utility_rollout_cache(remaining: int):
    """cache all action sequences and motion deltas for one horizon.

    for a remaining horizon H, the exhaustive copter solver evaluates M = 8^H
    discrete action sequences. this cache materializes that action space once:

    - sequence_actions[m, t] is the t-th action in sequence m
    - row_delta_by_sequence and col_delta_by_sequence store the per-step motion
    - step_length_by_sequence stores the euclidean move length per step

    the vectorized rollout then applies the same recurrence to every candidate
    sequence in parallel:

        r_{m,t+1} = clip(r_{m,t} + delta_r[m,t])
        c_{m,t+1} = clip(c_{m,t} + delta_c[m,t])

    so the later scoring step can batch all M candidates without changing the
    underlying exhaustive search.
    """
    cached_rollout = _UTILITY_ROLLOUT_CACHE.get(remaining)
    if cached_rollout is not None:
        return cached_rollout

    from itertools import product as iproduct

    sequence_actions = np.array(
        list(iproduct(range(1, len(ACTIONS)), repeat=remaining)),
        dtype=np.int8,
    )
    action_row_delta = np.array(
        [dr for dr, _ in ACTIONS],
        dtype=np.int8,
    )
    action_col_delta = np.array(
        [dc for _, dc in ACTIONS],
        dtype=np.int8,
    )
    action_step_length = np.hypot(
        action_row_delta.astype(np.float64),
        action_col_delta.astype(np.float64),
    )

    cached_rollout = {
        'sequence_actions': sequence_actions,
        'row_delta_by_sequence': action_row_delta[sequence_actions],
        'col_delta_by_sequence': action_col_delta[sequence_actions],
        'step_length_by_sequence': action_step_length[sequence_actions],
        'sequence_index': np.arange(len(sequence_actions)),
    }
    _UTILITY_ROLLOUT_CACHE[remaining] = cached_rollout
    return cached_rollout

def copter_explore_utility(beliefs: dict, copter_pos: tuple, b_max: dict,
                           T_c: int, alpha: float = 1.5,
                           record_beliefs: bool = False,
                           lambda_: float = 0.0,
                           rover_pos: tuple = (0, 0),
                           energy: float = float('inf')) -> tuple:
    """deterministic exhaustive copter search over the remaining horizon.

    the optimized implementation is still the same exact maximization over the
    discrete action space. for the remaining horizon H, it scores all
    M = 8^H candidate sequences by

        score_m = alpha I_m - lambda C_c,m

    in parallel. if sequence m visits cells x_{m,1}, ..., x_{m,H}, then

        C_c,m = sum_t 1[moved_{m,t}] * ||delta_{m,t}||_2
                + rho * ||x_{m,H} - x_r||_2

    and

        I_m = sum_x 1[x is newly seen by sequence m] * H(B_O(x))

    the boolean matrix seen_by_sequence[m, x] preserves the exact old semantics
    that repeated visits within one sequence only count once. the solver still
    returns the first action of the best feasible sequence, but the rollout and
    scoring are vectorized across all candidates.
    """
    pos = list(copter_pos)
    visited_global = {tuple(pos)}
    substeps = []
    belief_snapshots = []
    RHO = 1.0

    for step in range(T_c):
        remaining = T_c - step

        # step 1 cache the entropy reward of unseen cells
        # formula: entropy scout bonus
        # only cells not yet visited in this epoch can contribute new i(a_c)
        obstacle_belief_flat = np.array(
            [beliefs[(r, c)]['O'] for r in range(GRID) for c in range(GRID)],
            dtype=np.float64,
        )
        unseen_cell_mask = np.ones(GRID * GRID, dtype=bool)
        for r, c in visited_global:
            unseen_cell_mask[r * GRID + c] = False
        entropy_flat = np.zeros(GRID * GRID, dtype=np.float64)
        unseen_obstacle_belief = np.clip(
            obstacle_belief_flat[unseen_cell_mask],
            1e-9,
            1.0 - 1e-9,
        )
        entropy_flat[unseen_cell_mask] = (
            -unseen_obstacle_belief * np.log2(unseen_obstacle_belief)
            - (1.0 - unseen_obstacle_belief) * np.log2(1.0 - unseen_obstacle_belief)
        )

        # step 2 score staying for the rest of the epoch
        # formula: stay-put baseline
        d_return_now = abs(pos[0] - rover_pos[0]) + abs(pos[1] - rover_pos[1])
        best_score = -lambda_ * RHO * d_return_now
        best_action = 0

        # step 3 score every remaining action sequence
        # formula: batch rollout over m = 8^h candidates
        rollout_cache = _utility_rollout_cache(remaining)
        sequence_actions = rollout_cache['sequence_actions']
        row_delta_by_sequence = rollout_cache['row_delta_by_sequence']
        col_delta_by_sequence = rollout_cache['col_delta_by_sequence']
        step_length_by_sequence = rollout_cache['step_length_by_sequence']
        sequence_index = rollout_cache['sequence_index']

        row_position = np.full(len(sequence_actions), pos[0], dtype=np.int16)
        col_position = np.full(len(sequence_actions), pos[1], dtype=np.int16)
        flight_cost = np.zeros(len(sequence_actions), dtype=np.float64)
        information_gain = np.zeros(len(sequence_actions), dtype=np.float64)
        seen_by_sequence = np.zeros(
            (len(sequence_actions), GRID * GRID),
            dtype=bool,
        )

        for horizon_index in range(remaining):
            # step 3a apply the deterministic motion recurrence to every sequence
            next_row_position = np.clip(
                row_position + row_delta_by_sequence[:, horizon_index],
                0,
                GRID - 1,
            )
            next_col_position = np.clip(
                col_position + col_delta_by_sequence[:, horizon_index],
                0,
                GRID - 1,
            )
            moved = (
                (next_row_position != row_position)
                | (next_col_position != col_position)
            )

            # formula: copter trip ledger
            flight_cost += moved * step_length_by_sequence[:, horizon_index]

            row_position = next_row_position
            col_position = next_col_position
            cell_index = row_position * GRID + col_position

            # formula: set-once scout bonus
            # repeated visits in one sequence count once, matching the old set
            newly_seen = (
                unseen_cell_mask[cell_index]
                & ~seen_by_sequence[sequence_index, cell_index]
            )
            information_gain += newly_seen * entropy_flat[cell_index]
            seen_by_sequence[sequence_index, cell_index] = True

        # formula: frozen-return leash
        return_cost = RHO * np.hypot(
            row_position - rover_pos[0],
            col_position - rover_pos[1],
        )
        total_cost = flight_cost + return_cost
        feasible_sequence = total_cost <= energy
        if np.any(feasible_sequence):
            # formula: copter surrogate utility
            # score = alpha * i - lambda * c_c
            score_by_sequence = alpha * information_gain - lambda_ * total_cost
            feasible_score = np.where(
                feasible_sequence,
                score_by_sequence,
                -np.inf,
            )
            best_sequence_index = int(np.argmax(feasible_score))
            candidate_score = float(feasible_score[best_sequence_index])
            if candidate_score > best_score:
                best_score = candidate_score
                best_action = int(sequence_actions[best_sequence_index, 0])

        # step 4 staying means the epoch is done
        if best_action == 0:
            for _ in range(remaining):
                copter_sense(beliefs, tuple(pos))
                substeps.append(tuple(pos))
                if record_beliefs:
                    belief_snapshots.append(copy.deepcopy(beliefs))
            break

        # step 5 execute the first move of the best sequence
        pos = _copter_move(pos, best_action)
        visited_global.add(tuple(pos))
        copter_sense(beliefs, tuple(pos))
        substeps.append(tuple(pos))
        if record_beliefs:
            belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), substeps, belief_snapshots

def rover_execute(beliefs: dict, rover_pos: tuple, rover_q: int,
                  rover_policy: dict, T_r: int,
                  record_beliefs: bool = False, verbose: bool = True) -> tuple:
    """

    """
    pos = list(rover_pos)
    q = rover_q
    substeps = []
                  policy: dict, T_r: int,
                  record_beliefs: bool = False) -> tuple:
    """execute the rover policy for at most t_r steps."""
    pos              = list(rover_pos)
    q                = rover_q
    substeps         = [(tuple(pos), q)]
    belief_snapshots = []

    q_prev = q
    q = fsa_step(q, TRUE_L[tuple(pos)])
    if q != q_prev and verbose:
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
        # step 1 read the committed action from the current rover policy
        a = policy.get((pos[0], pos[1], q), 1)
        dr, dc = ACTIONS[a]
        nr, nc = clip(pos[0] + dr, pos[1] + dc)

        if np.random.rand() < 0.95:
            pos = [nr, nc]
            # Debug: Log when rover visits cells near D
            if tuple(pos) in [(7,5), (7,6), (6,6), (8,6)] and verbose:
                print(f"      Rover at {tuple(pos)}, q={q}, looking for={'D' if q==2 else 'other'}")
        else:
            slips = [list(clip(nr + ddr, nc + ddc))
                     for ddr, ddc in CARDINALS
                     if clip(nr + ddr, nc + ddc) != (nr, nc)]
            pos = slips[np.random.randint(len(slips))] if slips else [nr, nc]
        # step 2 apply deterministic rover motion
        pos = [nr, nc]

        q_prev = q
        # step 3 update the realized automaton state with true labels
        q = fsa_step(q, TRUE_L[tuple(pos)])
        if q != q_prev and verbose:
            print(f"    FSA transition: q={q_prev} → q={q} at pos={tuple(pos)}, labels={TRUE_L[tuple(pos)]}")


        # step 4 fuse the rover's local observation into the shared belief map
        rover_sense(beliefs, tuple(pos))
        substeps.append((tuple(pos), q))
        if record_beliefs:
            belief_snapshots.append(copy.deepcopy(beliefs))

    return tuple(pos), q, substeps, belief_snapshots
>>>>>>> origin/main
