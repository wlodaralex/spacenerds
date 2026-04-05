"""
planning.py

- build rover planners and copter path selectors
- provide reachability, entropy, and automaton helpers used during planning

class
- `DStarLitePlanner`: incremental rover planner on the product graph

function
- `grid_to_graph`: build the spatial graph
- `plan_copter_action_local`: choose one local baseline copter action
- `plan_copter_action_global`: choose one global baseline target and policy
- `plan_copter_action_game`: choose the best feasible game copter sequence
"""

import heapq
import math

import numpy as np

from environment import GRID, ACTIONS, clip, transition_probabilities
from fsa import FSA_ACCEPT, FSA_DEAD, FSA_ALL, compute_B_en
from sensor import sensor_beta


# --- Shared Planner State ---
_COPTER_FOOTPRINT_MASK = None
_UTILITY_ROLLOUT_CACHE = {}


# --- Public Helpers ---
def grid_to_graph() -> dict:
    """build the spatial adjacency graph for the grid."""
    adjacency_by_cell = {}
    for r in range(GRID):
        for c in range(GRID):
            neighbor_cells = []
            for action_index in range(1, len(ACTIONS)):   # skip stay (action 0)
                # step 1 encode the 8-connected spatial graph used by both planners
                # design note: the spatial graph excludes stay, so rover planning
                # always reasons over moving actions rather than waiting in place
                dr, dc = ACTIONS[action_index]
                nr, nc = clip(r + dr, c + dc)
                if (nr, nc) != (r, c):
                    neighbor_cells.append(((nr, nc), action_index))
            adjacency_by_cell[(r, c)] = neighbor_cells
    return adjacency_by_cell


def binary_entropy(belief: float) -> float:
    """Return binary Shannon entropy for one Bernoulli belief."""
    belief = np.clip(belief, 1e-9, 1 - 1e-9)
    return -belief * np.log2(belief) - (1 - belief) * np.log2(1 - belief)


def _baseline_b_en_cache(beliefs):
    """Cache baseline ``B_en`` values for one fixed belief snapshot."""
    return {
        (r, c, q): compute_B_en(beliefs[(r, c)], q)
        for r in range(GRID)
        for c in range(GRID)
        for q in FSA_ALL
        if q not in FSA_ACCEPT and q != FSA_DEAD
    }


# --- Baseline Copter Planning ---
def acquisition_function(beliefs, b_max, alpha):
    """Return the baseline copter acquisition function W(x)."""
    acquisition_values = {}
    for r in range(GRID):
        for c in range(GRID):
            acquisition_values[(r, c)] = (
                binary_entropy(beliefs[(r, c)]['O'])
                + alpha * b_max.get((r, c), 0.0)
            )
    return acquisition_values


def plan_copter_action_local(beliefs, copter_pos, b_max, alpha=1.5):
    """Return the next baseline local copter action."""
    acquisition_values = acquisition_function(beliefs, b_max, alpha)

    best_action, best_expected_value = 0, -np.inf
    for action_index in range(len(ACTIONS)):
        transition_distribution = transition_probabilities(
            copter_pos[0], copter_pos[1], action_index, p_intended=1.0
        )
        expected_value = sum(
            probability * acquisition_values.get(destination_cell, 0.0)
            for destination_cell, probability in transition_distribution.items()
        )
        if expected_value > best_expected_value:
            best_action, best_expected_value = action_index, expected_value
    return best_action


def plan_copter_action_global(beliefs, copter_pos, b_max,
                              alpha=1.5, vi_steps=80):
    """Return the current baseline global copter target and policy."""
    acquisition_values = acquisition_function(beliefs, b_max, alpha)
    candidate_values = {
        cell: value
        for cell, value in acquisition_values.items()
        if cell != tuple(copter_pos)
    }
    if not candidate_values:
        return None, {}, 0

    target_cell = max(candidate_values, key=candidate_values.get)
    copter_policy, copter_vi_sweeps = copter_value_iteration(
        target_cell, vi_steps=vi_steps
    )
    return target_cell, copter_policy, copter_vi_sweeps


def copter_value_iteration(target, vi_steps=80, tol=1e-3):
    """Copter value iteration.

    Solve the point-to-point reachability problem to reach ``target``.

    This keeps the baseline control flow but uses the current deterministic motion:
        - deterministic transition kernel with ``p_intended = 1.0``
        - randomized tie-breaking among equally optimal actions
        - convergence tolerance ``1e-3``
    """
    V = {}
    for r in range(GRID):
        for c in range(GRID):
            V[(r, c)] = 1.0 if (r, c) == target else 0.0

    copter_policy = {}
    for sweep in range(vi_steps):
        V_new = {}
        delta = 0.0
        for r in range(GRID):
            for c in range(GRID):
                if (r, c) == target:
                    V_new[(r, c)] = 1.0
                    copter_policy[(r, c)] = 0
                    continue
                best_val = -1.0
                candidates = []
                for a in range(1, len(ACTIONS)):
                    transition_distribution = transition_probabilities(
                        r, c, a, p_intended=1.0
                    )
                    val = sum(
                        probability * V.get(destination_cell, 0.0)
                        for destination_cell, probability
                        in transition_distribution.items()
                    )
                    if val > best_val + 1e-4:
                        best_val = val
                        candidates = [a]
                    elif abs(val - best_val) < 1e-4:
                        candidates.append(a)
                V_new[(r, c)] = best_val
                copter_policy[(r, c)] = np.random.choice(candidates)
                delta = max(delta, abs(best_val - V.get((r, c), 0.0)))
        V = V_new
        if delta < tol:
            return copter_policy, sweep + 1
    return copter_policy, vi_steps

# --- Copter Shared Helpers ---
def copter_footprint_mask():
    """Cache the copter sensor footprint for every grid cell."""
    global _COPTER_FOOTPRINT_MASK
    if _COPTER_FOOTPRINT_MASK is not None:
        return _COPTER_FOOTPRINT_MASK

    footprint_mask = np.zeros((GRID * GRID, GRID * GRID), dtype=bool)
    for r in range(GRID):
        for c in range(GRID):
            src_index = r * GRID + c
            for rr in range(max(0, r - 4), min(GRID, r + 5)):
                for cc in range(max(0, c - 4), min(GRID, c + 5)):
                    if sensor_beta((r, c), (rr, cc), 4, 0.4) > 0.5:
                        footprint_mask[src_index, rr * GRID + cc] = True

    _COPTER_FOOTPRINT_MASK = footprint_mask
    return _COPTER_FOOTPRINT_MASK


# --- Game Copter Planning ---
def _copter_rollout_cache(horizon: int):
    """Cache action sequences and per-step motion deltas for one horizon."""
    cached_rollout = _UTILITY_ROLLOUT_CACHE.get(horizon)
    if cached_rollout is not None:
        return cached_rollout

    from itertools import product as iproduct

    sequence_actions = np.array(
        list(iproduct(range(len(ACTIONS)), repeat=horizon)),
        dtype=np.int8,
    )
    action_row_delta = np.array([dr for dr, _ in ACTIONS], dtype=np.int8)
    action_col_delta = np.array([dc for _, dc in ACTIONS], dtype=np.int8)

    cached_rollout = {
        'sequence_actions': sequence_actions,
        'row_delta_by_sequence': action_row_delta[sequence_actions],
        'col_delta_by_sequence': action_col_delta[sequence_actions],
    }
    _UTILITY_ROLLOUT_CACHE[horizon] = cached_rollout
    return cached_rollout


def plan_copter_action_game(beliefs, copter_pos, T_c, alpha=1.5,
                            lambda_=0.0, rover_pos=(0, 0),
                            distance_budget=float('inf'), rho=1.0):
    """Return the best feasible game-theoretic copter action sequence."""
    # Inline this entropy field for vectorized batch scoring; the single-path
    # metric helper computes the same quantity after one path is committed.
    entropy_flat = np.array(
        [
            binary_entropy(beliefs[(r, c)]['O'])
            for r in range(GRID)
            for c in range(GRID)
        ],
        dtype=np.float64,
    )
    footprint_mask_by_cell = copter_footprint_mask()

    rollout_cache = _copter_rollout_cache(T_c)
    sequence_actions = rollout_cache['sequence_actions']
    row_delta_by_sequence = rollout_cache['row_delta_by_sequence']
    col_delta_by_sequence = rollout_cache['col_delta_by_sequence']

    row_position = np.full(len(sequence_actions), copter_pos[0], dtype=np.int16)
    col_position = np.full(len(sequence_actions), copter_pos[1], dtype=np.int16)
    flight_cost = np.zeros(len(sequence_actions), dtype=np.float64)
    observed_by_sequence = np.zeros(
        (len(sequence_actions), GRID * GRID),
        dtype=bool,
    )

    for horizon_index in range(T_c):
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

        flight_cost += np.hypot(
            next_row_position - row_position,
            next_col_position - col_position,
        )

        row_position = next_row_position
        col_position = next_col_position
        cell_index = row_position * GRID + col_position
        observed_by_sequence |= footprint_mask_by_cell[cell_index]

    information_gain = np.sum(
        observed_by_sequence * entropy_flat[None, :],
        axis=1,
        dtype=np.float64,
    )

    return_cost = rho * np.hypot(
        row_position - rover_pos[0],
        col_position - rover_pos[1],
    )
    total_cost = flight_cost + return_cost
    feasible_sequence = total_cost <= distance_budget
    if not np.any(feasible_sequence):
        return None

    score_by_sequence = alpha * information_gain - lambda_ * total_cost
    feasible_score = np.where(feasible_sequence, score_by_sequence, -np.inf)
    best_sequence_index = int(np.argmax(feasible_score))
    return tuple(int(action) for action in sequence_actions[best_sequence_index])


# --- D* Lite Planner ---
class DStarLitePlanner:
    """incremental d* lite on the product graph."""

    def __init__(self, spatial, beliefs, gamma=0.0, tau_r=0.9,
                 *, risk_beliefs=None, use_h2: bool = False,
                 use_h6: bool = False):
        self.spatial = spatial
        self.gamma = gamma
        self.tau_r = tau_r
        self._use_h2 = use_h2
        self._use_h6 = use_h6
        if self._use_h2 and self._use_h6:
            raise ValueError("use_h2 and use_h6 are mutually exclusive")
        self._active_q = tuple(self._active_qs())
        self._h = {}
        self._c_min_plus = 1e-9
        self._tq_sizes = {
            q: 0 for q in self._active_q
        }
        self._set_beliefs(beliefs, risk_beliefs)

        reverse_adjacency_sets = {
            (r, c): set() for r in range(GRID) for c in range(GRID)
        }
        for (r, c), neighbor_cells in spatial.items():
            for (nr, nc), _ in neighbor_cells:
                reverse_adjacency_sets[(nr, nc)].add((r, c))
        self._rev = {
            cell: list(predecessor_cells)
            for cell, predecessor_cells in reverse_adjacency_sets.items()
        }
        self._predecessor_states_by_cell = {
            cell: tuple(
                (r, c, q)
                for r, c in predecessor_cells
                for q in self._active_q
            )
            for cell, predecessor_cells in self._rev.items()
        }

        self._nodes = [
            (r, c, q)
            for r in range(GRID)
            for c in range(GRID)
            for q in FSA_ALL
        ]

        infinity = float('inf')
        self.g = {node: infinity for node in self._nodes}
        self.rhs = {node: infinity for node in self._nodes}

        self._heap = []
        self._keys = {}
        self._cnt  = 0

        for node in self._nodes:
            if node[2] in FSA_ACCEPT:
                self.rhs[node] = 0.0
                self._push(node)

        self._snap_mission = self._take_snap(self._mission_beliefs)
        self._snap_risk = self._take_snap(self._risk_beliefs)

        self._refresh_successor_cache()
        if self._use_h2 or self._use_h6:
            self._refresh_heuristic()

        self.n_expanded = 0
        self._run()

    @staticmethod
    def _take_snap(beliefs):
        """copy scalar cell beliefs for change detection."""
        return {
            (r, c): dict(beliefs[(r, c)])
            for r in range(GRID)
            for c in range(GRID)
        }

    def _set_beliefs(self, mission_beliefs, risk_beliefs=None):
        """set mission and risk belief snapshots for rover planning."""
        # formula: split-belief rover utility (Formulation §6 and §13)
        # mission beliefs carry B_t^+ for G, risk beliefs carry B_t for C_r
        self._mission_beliefs = mission_beliefs
        self._risk_beliefs = mission_beliefs if risk_beliefs is None else risk_beliefs
        self.beliefs = self._mission_beliefs

    @staticmethod
    def _active_qs():
        """return the automaton states that are still active."""
        return [
            q
            for q in FSA_ALL
            if q not in FSA_ACCEPT and q != FSA_DEAD
        ]

    @staticmethod
    def _needed_aps(q):
        # formula: progress alphabet (Formulation §1, remaining mission labels)
        # these are the atomic propositions that can still advance the mission
        required_atomic_propositions = {
            0: ('A', 'B', 'C'),
            1: ('C', 'A'),
            2: ('D', 'A'),
        }
        return required_atomic_propositions.get(q, ())

    def _key(self, state):
        # formula: d* queue key (D* Lite implementation, Formulation §13)
        # k(s) = (min(g, rhs) + h, min(g, rhs))
        best_estimate = min(self.g[state], self.rhs[state])
        heuristic_value = (
            self._h.get(state, 0.0) if (self._use_h2 or self._use_h6) else 0.0
        )
        return (best_estimate + heuristic_value, best_estimate)

    def _push(self, state):
        key = self._key(state)
        self._cnt += 1
        heap_entry = (key, self._cnt, state)
        heapq.heappush(self._heap, heap_entry)
        self._keys[state] = (key, self._cnt)

    def _pop(self):
        while self._heap:
            key, counter, state = heapq.heappop(self._heap)
            if self._keys.get(state) == (key, counter):
                del self._keys[state]
                return key, state
        return None, None

    def _top_key(self):
        """peek the smallest valid queue key."""
        while self._heap:
            key, counter, state = self._heap[0]
            if self._keys.get(state) == (key, counter):
                return key
            heapq.heappop(self._heap)
        return (float('inf'), float('inf'))

    def _succ(self, state):
        """yield forward successors and edge costs from one state."""
        if state[2] in FSA_ACCEPT or state[2] == FSA_DEAD:
            return ()
        return self._successors_by_state.get(state, ())

    def _compute_c_min_plus(self):
        """return the smallest positive edge cost in the current graph."""
        return self._c_min_plus

    @staticmethod
    def _zero_terminal_heuristic_values(heuristic_values):
        """Set all accepting and dead heuristic values to zero."""
        for r in range(GRID):
            for c in range(GRID):
                for q in FSA_ACCEPT:
                    heuristic_values[(r, c, q)] = 0.0
                heuristic_values[(r, c, FSA_DEAD)] = 0.0

    def _compute_h2(self):
        """build the admissible chebyshev heuristic h2."""
        heuristic_values = {}
        target_set_sizes = {}
        for q in self._active_q:
            # step 1 build t_q, the cells that could still advance the fsa
            required_atomic_propositions = self._needed_aps(q)
            target_cells = [
                (r, c)
                for r in range(GRID)
                for c in range(GRID)
                if any(
                    self._mission_beliefs[(r, c)][atomic_proposition] > 1e-9
                    for atomic_proposition in required_atomic_propositions
                )
            ]
            target_set_sizes[q] = len(target_cells)

            for r in range(GRID):
                for c in range(GRID):
                    if target_cells:
                        # formula: chebyshev lower bound (Formulation §12, h2)
                        chebyshev_distance = min(
                            max(abs(r - target_r), abs(c - target_c))
                            for target_r, target_c in target_cells
                        )
                    else:
                        chebyshev_distance = GRID
                    heuristic_values[(r, c, q)] = (
                        chebyshev_distance * self._c_min_plus
                    )

        self._zero_terminal_heuristic_values(heuristic_values)
        self._tq_sizes = target_set_sizes
        return heuristic_values

    def _compute_h6(self):
        """build the admissible risk-only heuristic h6."""
        heuristic_values = {}
        target_set_sizes = {}
        for q in self._active_q:
            # step 1 build t_q, the cells that could still advance the fsa
            required_atomic_propositions = self._needed_aps(q)
            target_cells = {
                (r, c)
                for r in range(GRID)
                for c in range(GRID)
                if any(
                    self._mission_beliefs[(r, c)][atomic_proposition] > 1e-9
                    for atomic_proposition in required_atomic_propositions
                )
            }
            target_set_sizes[q] = len(target_cells)

            distance_to_target = {
                (r, c): float('inf')
                for r in range(GRID)
                for c in range(GRID)
            }
            heap = []
            for target_cell in target_cells:
                distance_to_target[target_cell] = 0.0
                heapq.heappush(heap, (0.0, target_cell))

            while heap:
                # step 2 reverse dijkstra accumulates only the risk term
                path_cost, current_cell = heapq.heappop(heap)
                if path_cost > distance_to_target[current_cell]:
                    continue
                for predecessor_cell in self._rev[current_cell]:
                    obstacle_belief = max(
                        1e-9,
                        min(1.0 - 1e-9, self._risk_beliefs[current_cell]['O']),
                    )
                    # formula: risk-only corridor bound (Formulation §12, h6)
                    edge_risk = self.gamma * (-math.log(1.0 - obstacle_belief))
                    candidate_cost = path_cost + edge_risk
                    if candidate_cost < distance_to_target[predecessor_cell]:
                        distance_to_target[predecessor_cell] = candidate_cost
                        heapq.heappush(heap, (candidate_cost, predecessor_cell))

            for r in range(GRID):
                for c in range(GRID):
                    heuristic_values[(r, c, q)] = (
                        distance_to_target[(r, c)]
                    )

        self._zero_terminal_heuristic_values(heuristic_values)
        self._tq_sizes = target_set_sizes
        return heuristic_values

    def _refresh_heuristic(self):
        """refresh the active heuristic for the current belief snapshot."""
        # step 1 activate only the requested lower bound
        if self._use_h2:
            self._h = self._compute_h2()
        elif self._use_h6:
            self._h = self._compute_h6()
        else:
            self._h = {}

    def _refresh_successor_cache(self):
        """cache per-cell and per-state transition costs for one belief snapshot.

        for fixed mission beliefs B_t^+ and risk beliefs B_t, every successor
        edge out of (r, c, q) has additive cost

            c((r,c,q) -> (nr,nc,q')) =
                t((nr,nc), q, q') + r(nr,nc)

        where

            t(x', q, q') = -log B_en^{mission}(x', q -> q')
            r(x')        = gamma * (-log(1 - B_O^{risk}(x')))

        neither term depends on g or rhs, so the full successor list S(r,c,q)
        can be cached once per replan. this turns _succ(...) into a lookup
        instead of recomputing logs and transition distributions inside the d*
        queue updates.
        """
        risk_penalty_by_cell = {}
        for r in range(GRID):
            for c in range(GRID):
                # step 1 precompute the destination-cell survival toll
                obstacle_belief = max(
                    1e-9,
                    min(1.0 - 1e-9, self._risk_beliefs[(r, c)]['O']),
                )
                if obstacle_belief > self.tau_r:
                    risk_penalty_by_cell[(r, c)] = None
                else:
                    risk_penalty_by_cell[(r, c)] = (
                        self.gamma * (-math.log(1.0 - obstacle_belief))
                    )

        transition_cost_by_cell_and_q = {}
        for r in range(GRID):
            for c in range(GRID):
                # step 2 precompute the destination-cell mission logcosts
                cell_beliefs = self._mission_beliefs[(r, c)]
                for q in self._active_q:
                    transition_cost_by_cell_and_q[(r, c, q)] = tuple(
                        (q_next, -math.log(transition_probability))
                        for q_next, transition_probability in compute_B_en(
                            cell_beliefs, q
                        ).items()
                        if transition_probability >= 1e-12
                    )

        successors_by_state = {}
        smallest_positive_cost = float('inf')
        for r in range(GRID):
            for c in range(GRID):
                for q in self._active_q:
                    # step 3 stitch mission logcost and risk toll into edge cost
                    state_successors = []
                    for (nr, nc), action_index in self.spatial[(r, c)]:
                        risk_penalty = risk_penalty_by_cell[(nr, nc)]
                        if risk_penalty is None:
                            continue
                        for q_next, transition_cost in transition_cost_by_cell_and_q[
                            (nr, nc, q)
                        ]:
                            edge_cost = transition_cost + risk_penalty
                            state_successors.append(
                                ((nr, nc, q_next), action_index, edge_cost)
                            )
                            if 0.0 < edge_cost < smallest_positive_cost:
                                smallest_positive_cost = edge_cost
                    successors_by_state[(r, c, q)] = tuple(state_successors)

        self._risk_penalty_by_cell = risk_penalty_by_cell
        self._transition_cost_by_cell_and_q = transition_cost_by_cell_and_q
        self._successors_by_state = successors_by_state
        self._c_min_plus = (
            smallest_positive_cost if smallest_positive_cost < float('inf') else 1e-9
        )

    def _heuristic_stats(self):
        """return heuristic tightness stats for benchmarks."""
        active_nodes = [
            (r, c, q)
            for r in range(GRID)
            for c in range(GRID)
            for q in self._active_q
        ]
        if (self._use_h2 or self._use_h6) and active_nodes:
            h_mean = sum(self._h.get(n, 0.0) for n in active_nodes) / len(active_nodes)
        else:
            h_mean = 0.0
        return {
            'tq0_size': self._tq_sizes.get(0, 0),
            'tq1_size': self._tq_sizes.get(1, 0),
            'tq2_size': self._tq_sizes.get(2, 0),
            'h_mean_active': h_mean,
        }

    def _update(self, state):
        """recompute rhs(state) and queue it if inconsistent."""
        if state[2] not in FSA_ACCEPT:
            # formula: one-step rhs backup (D* Lite implementation, Formulation §13)
            # rhs(s) = min_{s'} c(s,s') + g(s')
            best_rhs_value = float('inf')
            for destination_state, _, edge_cost in self._succ(state):
                candidate_value = edge_cost + self.g[destination_state]
                if candidate_value < best_rhs_value:
                    best_rhs_value = candidate_value
            self.rhs[state] = best_rhs_value
        # drop stale queue keys and reinsert if still inconsistent
        self._keys.pop(state, None)
        if self.g[state] != self.rhs[state]:
            self._push(state)

    def _pred_nodes(self, cell):
        """lift spatial predecessors of one cell into product states."""
        return self._predecessor_states_by_cell[cell]

    def _changed_cells(self, mission_beliefs, risk_beliefs):
        """find cells whose mission or risk beliefs changed."""
        changed = set()
        for r in range(GRID):
            for c in range(GRID):
                # step 1 mission beliefs drive b_en changes
                for atomic_proposition in mission_beliefs[(r, c)]:
                    if abs(
                        mission_beliefs[(r, c)][atomic_proposition]
                        - self._snap_mission[(r, c)].get(atomic_proposition, 0.5)
                    ) > 1e-9:
                        changed.add((r, c))
                        break
                if (r, c) in changed:
                    continue
                # step 2 risk beliefs drive gamma * survival changes
                if abs(
                    risk_beliefs[(r, c)]['O']
                    - self._snap_risk[(r, c)].get('O', 0.5)
                ) > 1e-9:
                    changed.add((r, c))
        return changed

    def _run(self):
        """process the queue until all relevant states are consistent."""
        infinite_key = (float('inf'), float('inf'))
        while self._top_key() < infinite_key:
            popped_key, state = self._pop()
            if state is None:
                break

            current_key = self._key(state)
            if popped_key < current_key:
                # Re-enqueue with the corrected key while reusing ``current_key``.
                self._cnt += 1
                heapq.heappush(self._heap, (current_key, self._cnt, state))
                self._keys[state] = (current_key, self._cnt)
                continue

            self.n_expanded += 1
            r, c = state[0], state[1]

            if self.g[state] > self.rhs[state]:
                # step 1 under-consistent state becomes locally optimal
                self.g[state] = self.rhs[state]
                for predecessor_state in self._pred_nodes((r, c)):
                    self._update(predecessor_state)
            else:
                # step 2 over-consistent state is reopened and pushed backward
                self.g[state] = float('inf')
                self._update(state)
                for predecessor_state in self._pred_nodes((r, c)):
                    self._update(predecessor_state)

    def replan(self, beliefs, risk_beliefs=None):
        """incrementally replan after belief updates."""
        # step 1 refresh the split belief snapshots from the new epoch
        self._set_beliefs(beliefs, risk_beliefs)
        self.n_expanded = 0

        # step 2 skip all queue work if the belief snapshot is unchanged
        changed = self._changed_cells(self._mission_beliefs, self._risk_beliefs)

        if not changed:
            return self.solution()

        # step 3 rebuild only the edge-local caches that depend on beliefs
        self._refresh_successor_cache()
        if self._use_h2 or self._use_h6:
            self._refresh_heuristic()

        # step 4 only predecessors of changed cells can need rhs repairs
        affected_states = {
            predecessor_state
            for changed_cell in changed
            for predecessor_state in self._pred_nodes(changed_cell)
        }

        for affected_state in affected_states:
            self._update(affected_state)

        # step 5 resume the incremental shortest-path repair
        self._run()
        self._snap_mission = self._take_snap(self._mission_beliefs)
        self._snap_risk = self._take_snap(self._risk_beliefs)
        return self.solution()

    def solution(self):
        """extract v, policy, and expansion count from current g values."""
        value_function = {}
        policy = {}
        for node in self._nodes:
            r, c, q = node
            path_cost = self.g[node]
            # Formula: exp-of-cost value map (Formulation §7 and §13, V = exp(-g)).
            # When gamma > 0 this stored value is the remaining rover path score
            # exp(G_future - gamma C_r_future), not a pure mission-success V_phi.
            value_function[node] = (
                0.0 if path_cost == float('inf') else math.exp(-path_cost)
            )
            if q in FSA_ACCEPT or q == FSA_DEAD:
                policy[node] = 0
                continue
            best_action_index, best_total_cost = 1, float('inf')
            for destination_state, action_index, edge_cost in self._succ(node):
                # formula: greedy q backup (Formulation §13, rover best response)
                total_cost = edge_cost + self.g.get(destination_state, float('inf'))
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_action_index = action_index
            policy[node] = best_action_index
        return value_function, policy, self.n_expanded

def rover_value_iteration(beliefs, vi_steps=80, T_r=3, tol=1e-3):
    """Rover product-MDP value iteration.

    Solve the finite-horizon reachability problem on the product belief MDP.

    This keeps the baseline control flow but uses the current deterministic motion:
        - deterministic rover motion with ``p_intended = 1.0``
        - source-cell automaton transitions ``B_en(x, q -> q')``
        - randomized tie-breaking among equally optimal actions
        - convergence tolerance ``1e-3``
    """
    cells = [(r, c) for r in range(GRID) for c in range(GRID)]

    V = {}
    for r, c in cells:
        for q in FSA_ALL:
            V[(r, c, q)] = 1.0 if q in FSA_ACCEPT else 0.0

    # Beliefs are fixed for one VI call, so cache source-cell B_en once.
    B_en_cache = _baseline_b_en_cache(beliefs)

    rover_policy = {}
    # For longer rover phases, exclude stay to avoid frozen high-belief local
    # maxima from dominating the finite-horizon baseline rollout.
    action_range = range(1, len(ACTIONS)) if T_r > 5 else range(len(ACTIONS))

    for sweep in range(vi_steps):
        V_new = {}
        delta = 0.0
        for r, c in cells:
            for q in FSA_ALL:
                if q in FSA_ACCEPT:
                    V_new[(r, c, q)] = 1.0
                    rover_policy[(r, c, q)] = 0
                    continue
                if q == FSA_DEAD:
                    V_new[(r, c, q)] = 0.0
                    rover_policy[(r, c, q)] = 0
                    continue
                ben_source = B_en_cache.get((r, c, q), {})
                best_val = -1.0
                candidates = []
                for a in action_range:
                    transition_distribution = transition_probabilities(
                        r, c, a, p_intended=1.0
                    )
                    val = 0.0
                    for (r2, c2), p_move in transition_distribution.items():
                        # Baseline alignment: evaluate B_en at the source cell.
                        for q_next, p_fsa in ben_source.items():
                            val += p_move * p_fsa * V.get((r2, c2, q_next), 0.0)
                    if val > best_val + 1e-5:
                        best_val = val
                        candidates = [a]
                    elif abs(val - best_val) < 1e-5:
                        candidates.append(a)
                V_new[(r, c, q)] = best_val
                rover_policy[(r, c, q)] = np.random.choice(candidates)
                delta = max(delta, abs(best_val - V.get((r, c, q), 0.0)))
        V = V_new
        if delta < tol:
            return V, rover_policy, sweep + 1
    return V, rover_policy, vi_steps
