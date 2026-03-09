"""
planning.py - Value iteration (VI) and reachability belief (b_max).

Setup:
    - Copter planning pipeline:
        - Value iteration

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
from numpy import random

from environment import GRID, ACTIONS, transition_probabilities
from fsa import FSA_ACCEPT, FSA_DEAD, FSA_ALL, compute_B_en



# Copter Value Iteration #
def copter_value_iteration(target: tuple, vi_steps: int = 80, tol: float = 1e-3) -> tuple[dict, int]:
    """
    Solve the point-to-point reachability problem to reach 'target' x*.

    Optimal policy (Eq. 13):
        * μ*_c2 = argmax_{μ_c} Pr[trajectory reaches x* in finite time]

    Parameter:
        * target   : (row, column) of the cell x* to navigate toward.
        * vi_steps : maximum number of sweeps.
        * tol      : convergence threshold (default of 1e-6).

    Return:
        * copter_policy : dict. keyed by (r, c) -> action index.
        * sweeps_used   :
            * sweeps_used < T_steps  : indicates early convergence.
            * sweeps_used == T_steps : performed all allowed sweeps.
                - VI did not meet the early stopping criterion, but the value function may still be stable.
    """
    # Initialize V
    V = {}
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) == target:
                V[(r, c)] = 1.0  
            else:
                V[(r, c)] = 0.0

    copter_policy = {}

    for sweep in range(vi_steps):
        V_new = {}
        delta = 0.0  # track max change this sweep (for convergence check)

        for r in range(GRID):
            for c in range(GRID):
                if (r, c) == target:
                    V_new[(r, c)] = 1.0
                    copter_policy[(r, c)] = 0
                    continue

                # best_val, best_a = -1.0, 1
                
                # # Bellman maximization over actions
                # for a in range(1, len(ACTIONS)):
                #     trans = transition_probabilities(r, c, a, p_intended=0.90)
                #     val = 0.0
                #     for cell, p in trans.items():
                #         val += p * V.get(cell, 0.0)
                #     if val > best_val:
                #         best_val, best_a = val, a

                # V_new[(r, c)] = best_val
                # copter_policy[(r, c)] = best_a

                # collect all actions within equally optimal tolerance, pick randomly among them to avoid bias and prevent picking same direction
                best_val = -1.0
                candidates = []

                # Bellman maximization over actions
                for a in range(1, len(ACTIONS)):
                    trans = transition_probabilities(r, c, a, p_intended=0.90)
                    val = 0.0
                    for cell, p in trans.items():
                        val += p * V.get(cell, 0.0)
                    if val > best_val + 1e-5:
                        best_val = val
                        candidates = [a]
                    elif abs(val - best_val) < 1e-5:
                        candidates.append(a)

                V_new[(r, c)] = best_val
                copter_policy[(r, c)] = random.choice(candidates)

                # Track the largest change across non-terminal states                
                delta = max(delta, abs(best_val - V.get((r, c), 0.0)))

        V = V_new

        # Early exit if value function has stabilized
        if delta < tol:
            return copter_policy, sweep + 1  # converged before budget exhausted

    return copter_policy, vi_steps  # budget exhausted (V may still be practically converged)



# Rover Value Iteration #
def rover_value_iteration(beliefs: dict, vi_steps: int = 80, T_r: int = 3, tol: float = 1e-3) \
    -> tuple[dict, dict, int]:
    """
    Solve the finite-horizon reachability problem on the product belief MDP.

    Value function (Eq. 20):
        * V^{l+1}(r,c,q) = max_{a} sum_{r',c',q'} p_S((r',c',q')|(r,c,q),a) * V^l(r',c',q')

    Transition belief (Eq. 18):
        * p_S = p_r(r'|r,a) * B_en(r,c;q->q')
    
    Parameter:
        * beliefs  : environmental belief dict. B[(r,c)][ap] that is fixed for this VI call.
        * T_r      : rover execution phase length that controls stay-action inclusion.
                        * T_r <= 5 : all 5 actions (paper's implementation)
                        * T_r > 5  : 4 cardinal actions (stay excluded)
        * vi_steps : maximum number of sweeps.
        * tol      : convergence threshold (default of 1e-6).

    Return:
        * V            : dict. keyed by (r, c, q) -> value in [0, 1].
        * rover_policy : dict. keyed by (r, c, q) -> action index.
        * sweeps_used  :
            * sweeps_used < T_steps  : indicates early convergence.
            * sweeps_used == T_steps : performed all allowed sweeps.
                - VI did not meet the early stopping criterion, but the value function may still be stable.
    """
    cells = []
    for r in range(GRID):
        for c in range(GRID):
            cells.append((r, c))

    # Initialize V
    V = {}
    for (r, c) in cells:
        for q in FSA_ALL:
            if q in FSA_ACCEPT:
                V[(r, c, q)] = 1.0  # mission success
            else:
                V[(r, c, q)] = 0.0

    # Cache B_en for all non-terminal (cell, q) pairs
    B_en = {}
    for (r, c) in cells:
        for q in FSA_ALL:
            if q not in FSA_ACCEPT and q != FSA_DEAD:
                # Beliefs are fixed for the duration of one VI call
                B_en[(r, c, q)] = compute_B_en(beliefs[(r, c)], q) 

    rover_policy = {}
    action_range = range(1, len(ACTIONS)) if T_r > 5 else range(len(ACTIONS))

    for sweep in range(vi_steps):
        V_new = {}
        delta = 0.0  # track max change this sweep (for convergence check)

        for (r, c) in cells:
            for q in FSA_ALL:

                # Absorbing states
                if q in FSA_ACCEPT:
                    V_new[(r, c, q)] = 1.0
                    rover_policy[(r, c, q)] = 0  # stay is correct at accepting state
                    continue
                if q == FSA_DEAD:
                    V_new[(r, c, q)] = 0.0
                    rover_policy[(r, c, q)] = 0
                    continue

                ben = B_en.get((r, c, q), {})
                # best_val = -1.0
                # best_a = 1  # default: move up

                # # Bellman maximization over actions
                # for a in action_range:
                #     trans = transition_probabilities(r, c, a, p_intended=0.95)
                #     val = 0.0
                #     for (r2, c2), p_move in trans.items():
                #         for q_next, p_fsa in ben.items():
                #             val += p_move * p_fsa * V.get((r2, c2, q_next), 0.0)
                #     if val > best_val:
                #         best_val = val
                #         best_a = a

                # V_new[(r, c, q)] = best_val
                # rover_policy[(r, c, q)] = best_a

                # collect all actions within equally optimal tolerance, pick randomly among them to avoid bias and prevent picking same direction
                best_val = -1.0
                candidates = []

                # Bellman maximization over actions
                for a in action_range:
                    trans = transition_probabilities(r, c, a, p_intended=0.95)
                    val = 0.0
                    for (r2, c2), p_move in trans.items():
                        for q_next, p_fsa in ben.items():
                            val += p_move * p_fsa * V.get((r2, c2, q_next), 0.0)
                    if val > best_val + 1e-6:
                        best_val = val
                        candidates = [a]
                    elif abs(val - best_val) < 1e-6:
                        candidates.append(a)

                V_new[(r, c, q)] = best_val
                rover_policy[(r, c, q)] = random.choice(candidates)

                # Track the largest change across non-terminal states
                delta = max(delta, abs(best_val - V.get((r, c, q), 0.0)))

        V = V_new

        # Early exit if value function has stabilized
        if delta < tol:
            return V, rover_policy, sweep + 1  # converged before budget exhausted

    return V, rover_policy, vi_steps  # budget exhausted (V may still be practically converged)



# Reachability Belief #
def compute_b_max(beliefs: dict, rover_policy: dict, rover_pos: tuple, rover_q: int, T_r: int) -> dict:
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
    # Cache B_en for all (cell, q) pairs
    B_en_cache = {}
    for r in range(GRID):
        for c in range(GRID):
            for q in FSA_ALL:
                B_en_cache[(r, c, q)] = compute_B_en(beliefs[(r, c)], q)
    
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
            a = rover_policy.get((r, c, q), 1)
            ben = B_en_cache[(r, c, q)]  # Use cached value
            # ben = compute_B_en(beliefs[(r, c)], q)
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