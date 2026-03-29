"""
planning.py - Value iteration (VI) and reachability belief (b_max).

Setup:
    - Rover planning pipeline:
        - Value iteration over the product belief MDP (Section 4.3.2: Eqs. 20-21 & Algorithm 4).
        - Reachability belief b_max used by the copter's acquisition function to bias 
          exploration toward the rover's likely path (Section 4.3.3).

    - Product MDP state space - (cell, FSA_state) (Section 4.3.2 Remark 1):
        - Beliefs enter only through B_en weights (not part of the state space).
        - Reduced complexity O(|X|^2 |Q|^2 |U|) with polynomial in |X| versus 
          the exponential complexity in |X| of prior work that includes belief states explicitly.
    
    - Design decision:
        - Implemented convergence check as the paper assumes VI runs to convergence but does not specify a sweep count.
            - This implementation runs at most 'T_steps' sweeps and stops early if the maximum 
              change in V across all states falls below a set tolerance.
        - Optional: stay-action guard for large T_r as algorithm can 'trap' the rover at a high-belief cell where V_stay > V_move.
            - This belief-induced local maximum for small T_r is self-corrected at the next replanning cycle, but for larger
              the freeze is enough to look like a hang.
            - See CHANGELOG.md.
"""

import numpy as np
from environment import GRID, ACTIONS, transition_probabilities
from fsa import FSA_ACCEPT, FSA_DEAD, FSA_ALL, compute_B_en

_Q_IDX = {q: i for i, q in enumerate(FSA_ALL)}
_NQ = len(FSA_ALL)
_N = GRID * GRID
_P_ROVER = None

def _rover_kernel():
    """Build and cache rover transition kernel P[a, i, j] with p_intended=0.95."""
    global _P_ROVER
    if _P_ROVER is not None:
        return _P_ROVER
    nA = len(ACTIONS)
    P = np.zeros((nA, _N, _N))
    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            for a in range(nA):
                for (r2, c2), p in transition_probabilities(r, c, a, 0.95).items():
                    P[a, i, r2 * GRID + c2] = p
    _P_ROVER = P
    return P


# Value Iteration #
def value_iteration(beliefs: dict, vi_steps: int = 80, T_r: int = 3, tol: float = 1e-4) \
    -> tuple[dict, dict, int]:
    """
    Solve the finite-horizon reachability problem on the product belief MDP.

    Value function (Eq. 20):
        * V^{l+1}(r,c,q) = max_{a} sum_{r',c',q'} p_S((r',c',q')|(r,c,q),a) * V^l(r',c',q')

    Transition belief (Eq. 18):
        * p_S = p_r(r'|r,a) * B_en(r,c;q->q')
    
    Parameter:
        * beliefs  : environmental belief dict. B[(r,c)][ap] that is fixed for this VI call.
        * vi_steps : maximum number of sweeps.
        * tol      : convergence threshold.

    Return:
        * V           : dict keyed by (r, c, q) -> value in [0, 1].
        * policy      : dict keyed by (r, c, q) -> action index.
        * sweeps_used : number of sweeps performed.
    """
    P = _rover_kernel()
    nA = len(ACTIONS)

    # B_en[i, qi, qj] = belief-weighted FSA transition probability at cell i from q_qi to q_qj
    B_en = np.zeros((_N, _NQ, _NQ))
    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            for q in FSA_ALL:
                qi = _Q_IDX[q]
                if q in FSA_ACCEPT or q == FSA_DEAD:
                    continue
                ben = compute_B_en(beliefs[(r, c)], q)
                for q_next, p_fsa in ben.items():
                    B_en[i, qi, _Q_IDX[q_next]] = p_fsa

    # V[i, qi] — value at (cell i, FSA state index qi)
    V = np.zeros((_N, _NQ))
    for q in FSA_ACCEPT:
        V[:, _Q_IDX[q]] = 1.0

    accept_mask = np.array([q in FSA_ACCEPT for q in FSA_ALL])
    dead_idx = _Q_IDX[FSA_DEAD]
    active_qi = [_Q_IDX[q] for q in FSA_ALL if q not in FSA_ACCEPT and q != FSA_DEAD]

    pol = np.ones((_N, _NQ), dtype=int)

    for sweep in range(vi_steps):
        # For each action a: Q_a[i, qi] = Σ_j P[a,i,j] * Σ_qj B_en[i,qi,qj] * V[j,qj]
        # Rewrite: weighted_V[i, qi] = Σ_qj B_en[i,qi,qj] * V_sum_over_j[a,i,qj]
        #   where V_sum_over_j[a,i,qj] = Σ_j P[a,i,j]*V[j,qj] = (P[a] @ V[:,qj])[i]

        # PV[a, i, qj] = Σ_j P[a,i,j] * V[j,qj]
        PV = np.einsum('aij,jq->aiq', P, V)

        # Q_a[a, i, qi] = Σ_qj B_en[i,qi,qj] * PV[a,i,qj]
        Q_all = np.einsum('iqj,aij->aiq', B_en, PV)

        # cardinal actions only (exclude stay = action 0)
        Q_card = Q_all[1:]
        best_a_offset = np.argmax(Q_card, axis=0)
        best_a = best_a_offset + 1
        V_new_vals = np.max(Q_card, axis=0)

        # Preserve absorbing states
        V_new = V_new_vals.copy()
        for q in FSA_ACCEPT:
            V_new[:, _Q_IDX[q]] = 1.0
        V_new[:, dead_idx] = 0.0

        delta = np.max(np.abs(V_new - V))
        V = V_new
        pol = best_a

        if delta < tol:
            return _unpack(V, pol, sweep + 1)

    return _unpack(V, pol, vi_steps)


def _unpack(V_arr, pol_arr, sweeps):
    """Convert numpy arrays back to dicts for compatibility."""
    V = {}
    policy = {}
    for r in range(GRID):
        for c in range(GRID):
            i = r * GRID + c
            for q in FSA_ALL:
                qi = _Q_IDX[q]
                V[(r, c, q)] = float(V_arr[i, qi])
                if q in FSA_ACCEPT:
                    policy[(r, c, q)] = 0
                elif q == FSA_DEAD:
                    policy[(r, c, q)] = 0
                else:
                    policy[(r, c, q)] = int(pol_arr[i, qi])
    return V, policy, sweeps



# Reachability Belief #
def compute_b_max(beliefs: dict, policy: dict, rover_pos: tuple, rover_q: int, T_r: int) -> dict:
    """
    Compute b_max(x) = max_{l=0..T_r} Pr[rover reaches cell x in l steps].

    Forward-propagates the rover's probability distribution through the
    product belief MDP under the current optimal policy, accumulating the
    maximum per-cell visitation probability over all horizons l ∈ {0, ..., T_r}.

    Return:
        * b_max: dict. {(r,c): probability} for all grid cells.
            - Tells the copter where the rover is likely to go next, so it can 'weight' 
              its exploration of obstacles along that path (Section 4.2.2: Eq. 10).  
    """
    # Initialize with all probability mass at rover's current product-MDP state
    dist  = {(rover_pos[0], rover_pos[1], rover_q): 1.0}
    b_max = {}
    for r in range(GRID):
        for c in range(GRID):
            total = 0.0
            for q in FSA_ALL:
                total += dist.get((r, c, q), 0.0)
            b_max[(r, c)] = total

    for _ in range(T_r):
        dist_new = {}
        for (r, c, q), prob in dist.items():
            if prob < 1e-9:
                continue
            if q in FSA_ACCEPT or q == FSA_DEAD:
                # Absorbing states so probability mass stays at current cell
                dist_new[(r, c, q)] = dist_new.get((r, c, q), 0.0) + prob
                continue
            a = policy.get((r, c, q), 1)
            ben = compute_B_en(beliefs[(r, c)], q)
            for (r2, c2), p_move in transition_probabilities(r, c, a, 0.95).items():
                for q_next, p_fsa in ben.items():
                    key = (r2, c2, q_next)
                    dist_new[key] = dist_new.get(key, 0.0) + prob * p_move * p_fsa
        dist = dist_new

        # Update b_max and keep the maximum visitation probability seen so far
        for r in range(GRID):
            for c in range(GRID):
                v = 0.0
                for q in FSA_ALL:
                    v += dist.get((r, c, q), 0.0)
                if v > b_max[(r, c)]:
                    b_max[(r, c)] = v

    return b_max    