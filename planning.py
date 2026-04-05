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
from fsa import FSA_ACCEPT, FSA_DEAD, FSA_ALL, compute_B_en, fsa_step
from sensor import bayes_update, sensor_beta


# --- Public Helpers ---
_COPTER_FOOTPRINT_MASK = None
_UTILITY_ROLLOUT_CACHE = {}


def grid_to_graph() -> dict:
    """build the spatial adjacency graph for the grid."""
    adjacency_by_cell = {}
    for r in range(GRID):
        for c in range(GRID):
            neighbor_cells = []
            for action_index in range(1, len(ACTIONS)):   # skip stay (action 0)
                # encode the 8-connected spatial graph used by both planners
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
    # Step 1: score every cell by local obstacle uncertainty and rover need.
    acquisition_values = acquisition_function(beliefs, b_max, alpha)

    # Step 2: evaluate each immediate action by expected next-cell acquisition.
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

    # Step 3: return the best one-step baseline copter action.
    return best_action


def plan_copter_action_global(beliefs, copter_pos, b_max,
                              alpha=1.5, vi_steps=80):
    """Return the current baseline global copter target and policy."""
    # Step 1: build the global acquisition field over the whole grid.
    acquisition_values = acquisition_function(beliefs, b_max, alpha)
    candidate_values = {
        cell: value
        for cell, value in acquisition_values.items()
        if cell != tuple(copter_pos)
    }
    if not candidate_values:
        return None, {}, 0

    # Step 2: pick the highest-value target distinct from the current cell.
    target_cell = max(candidate_values, key=candidate_values.get)

    # Step 3: solve a point-to-point baseline policy toward that target.
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
    # Step 1: initialise reachability values with the target as absorbing.
    V = {}
    for r in range(GRID):
        for c in range(GRID):
            V[(r, c)] = 1.0 if (r, c) == target else 0.0

    copter_policy = {}
    for sweep in range(vi_steps):
        # Step 2: run one Bellman backup sweep and refresh the greedy policy.
        V_new = {}
        delta = 0.0
        for r in range(GRID):
            for c in range(GRID):

                # Step 2.1: keep the target cell absorbing with value 1.
                if (r, c) == target:
                    V_new[(r, c)] = 1.0
                    copter_policy[(r, c)] = 0
                    continue

                # Step 2.2: evaluate every moving action from the current cell.
                best_val = -1.0
                candidates = []
                for a in range(1, len(ACTIONS)):
                    transition_distribution = transition_probabilities(
                        r, c, a, p_intended=1.0
                    )
                    
                    # Step 2.3: back up the expected next-state reachability value.
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

                # Step 2.4: write the best backed-up value and one tied action.
                V_new[(r, c)] = best_val
                copter_policy[(r, c)] = np.random.choice(candidates)
                delta = max(delta, abs(best_val - V.get((r, c), 0.0)))

        # Step 2.5: commit the full sweep before checking convergence.
        V = V_new

        # Step 3: stop once the value updates fall below the tolerance.
        if delta < tol:
            return copter_policy, sweep + 1
    return copter_policy, vi_steps


# --- Game Copter Planning ---
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


def _simulate_copter_obstacle_beliefs(prior_beliefs, copter_substeps, rng):
    """Sample one hypothetical post-copter obstacle map from the current belief."""
    # Step 1: start from a copy of the current shared belief map.
    predicted_beliefs = {
        cell: dict(cell_beliefs)
        for cell, cell_beliefs in prior_beliefs.items()
    }

    for agent_position in copter_substeps:
        # Step 2: sample one hypothetical obstacle observation at each sensed cell.
        for row in range(max(0, agent_position[0] - 4), min(GRID, agent_position[0] + 5)):
            for col in range(max(0, agent_position[1] - 4), min(GRID, agent_position[1] + 5)):
                beta = sensor_beta(agent_position, (row, col), 4, 0.4)
                if beta <= 0.5:
                    continue

                current_belief = predicted_beliefs[(row, col)]['O']
                detection_probability = (
                    beta * current_belief
                    + (1.0 - beta) * (1.0 - current_belief)
                )
                observation = int(rng.random() < detection_probability)
                predicted_beliefs[(row, col)]['O'] = bayes_update(
                    current_belief,
                    observation,
                    beta,
                )

    return predicted_beliefs


def _estimate_copter_shared_term(prior_beliefs, copter_substeps, *,
                                 spatial, rover_pos, rover_q, tau_r,
                                 mc_samples, rng):
    """Estimate the shared mission term after one candidate copter sequence."""
    # Step 1: sample multiple hypothetical post-copter belief maps.
    mission_term_samples = []
    for _ in range(mc_samples):
        predicted_beliefs = _simulate_copter_obstacle_beliefs(
            prior_beliefs,
            copter_substeps,
            rng,
        )

        # Step 2: run a mission-only rover replan on each hypothetical map.
        mission_planner = DStarLitePlanner(
            spatial,
            predicted_beliefs,
            gamma=0.0,
            tau_r=tau_r,
            risk_beliefs=prior_beliefs,
            use_h2=True,
        )
        mission_values, _, _ = mission_planner.solution()
        mission_path_score = mission_values.get(
            (rover_pos[0], rover_pos[1], rover_q),
            0.0,
        )
        mission_term_samples.append(math.log(max(mission_path_score, 1e-9)))

    # Step 3: average the sampled log mission scores into one estimate.
    return float(np.mean(mission_term_samples))


def plan_copter_action_game(beliefs, copter_pos, T_c, alpha=1.5,
                            lambda_=0.0, rover_pos=(0, 0), rover_q=0,
                            distance_budget=float('inf'), rho=1.0,
                            tau_r=0.9, spatial=None, top_k=5,
                            mc_samples=3, rng=None):
    """Return the best feasible game-theoretic copter action sequence.

    Runtime strategy:
        1. Score every feasible sequence by the vectorized copter-private
           surrogate `alpha * I - lambda * C_c`.
        2. Keep the top-K surrogate sequences.
        3. For each shortlisted sequence, sample hypothetical post-copter
           beliefs and rerun a mission-only rover planner to estimate the
           shared mission term.
        4. Select the best reranked sequence.

    Setting `top_k <= 0` or `mc_samples <= 0` disables the mission-aware
    reranking pass and falls back to the surrogate-only copter objective.
    """
    # Step 1: build the vectorized entropy field and static sensor footprint map.
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
    row_substeps_by_sequence = np.empty(
        (len(sequence_actions), T_c),
        dtype=np.int16,
    )
    col_substeps_by_sequence = np.empty(
        (len(sequence_actions), T_c),
        dtype=np.int16,
    )
    flight_cost = np.zeros(len(sequence_actions), dtype=np.float64)
    observed_by_sequence = np.zeros(
        (len(sequence_actions), GRID * GRID),
        dtype=bool,
    )

    # Step 2: batch-roll out every horizon-T_c sequence and accumulate costs.
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
        row_substeps_by_sequence[:, horizon_index] = row_position
        col_substeps_by_sequence[:, horizon_index] = col_position
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

    # Step 3: enforce the hard distance budget and stop if nothing is feasible.
    if not np.any(feasible_sequence):
        return None

    surrogate_score_by_sequence = alpha * information_gain - lambda_ * total_cost
    feasible_score = np.where(feasible_sequence, surrogate_score_by_sequence, -np.inf)

    # Step 4: optionally fall back to the cheap surrogate-only copter objective.
    if top_k <= 0 or mc_samples <= 0:
        best_sequence_index = int(np.argmax(feasible_score))
        return tuple(int(action) for action in sequence_actions[best_sequence_index])

    if spatial is None:
        spatial = grid_to_graph()
    if rng is None:
        rng = np.random.default_rng()

    # Step 5: shortlist the top-K feasible surrogate sequences.
    feasible_indices = np.flatnonzero(feasible_sequence)
    shortlist_size = max(1, min(int(top_k), len(feasible_indices)))
    mc_samples = max(1, int(mc_samples))

    feasible_surrogate_scores = surrogate_score_by_sequence[feasible_indices]
    if shortlist_size < len(feasible_indices):
        top_relative_indices = np.argpartition(
            feasible_surrogate_scores,
            -shortlist_size,
        )[-shortlist_size:]
    else:
        top_relative_indices = np.arange(len(feasible_indices))

    shortlist_indices = feasible_indices[top_relative_indices]
    shortlist_indices = shortlist_indices[
        np.argsort(surrogate_score_by_sequence[shortlist_indices])[::-1]
    ]

    # Step 6: rerank the shortlist with Monte Carlo mission-only rover replans.
    best_sequence_index = int(shortlist_indices[0])
    best_reranked_score = -np.inf
    for sequence_index in shortlist_indices:
        copter_substeps = list(
            zip(
                row_substeps_by_sequence[sequence_index].tolist(),
                col_substeps_by_sequence[sequence_index].tolist(),
            )
        )
        shared_mission_term = _estimate_copter_shared_term(
            beliefs,
            copter_substeps,
            spatial=spatial,
            rover_pos=rover_pos,
            rover_q=rover_q,
            tau_r=tau_r,
            mc_samples=mc_samples,
            rng=rng,
        )
        reranked_score = (
            shared_mission_term
            + surrogate_score_by_sequence[sequence_index]
        )
        if reranked_score > best_reranked_score:
            best_reranked_score = reranked_score
            best_sequence_index = int(sequence_index)

    # Step 7: return the best reranked copter sequence.
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

    @staticmethod
    def _fsa_steps_to_acceptance():
        """Return the optimistic number of FSA transitions to acceptance."""
        atomic_propositions = ('O', 'A', 'B', 'C', 'D')
        reverse_successors = {q: set() for q in FSA_ALL}

        for q in FSA_ALL:
            for mask in range(1 << len(atomic_propositions)):
                labels = frozenset(
                    atomic_proposition
                    for bit_index, atomic_proposition in enumerate(atomic_propositions)
                    if mask & (1 << bit_index)
                )
                q_next = fsa_step(q, labels)
                reverse_successors[q_next].add(q)

        step_distance = {
            q: (0 if q in FSA_ACCEPT else float('inf'))
            for q in FSA_ALL
        }
        queue = list(FSA_ACCEPT)
        while queue:
            current_q = queue.pop(0)
            for predecessor_q in reverse_successors[current_q]:
                candidate_distance = step_distance[current_q] + 1
                if candidate_distance < step_distance[predecessor_q]:
                    step_distance[predecessor_q] = candidate_distance
                    queue.append(predecessor_q)
        step_distance[FSA_DEAD] = 0
        return step_distance

    def _compute_h2(self):
        """Build the admissible FSA-step heuristic h2 = d_FSA * c_min^+."""
        heuristic_values = {}
        step_distance = self._fsa_steps_to_acceptance()
        for q in self._active_q:
            # Step 1: count the optimistic FSA steps still needed from q.
            remaining_fsa_steps = step_distance.get(q, float('inf'))
            heuristic_value = (
                remaining_fsa_steps * self._c_min_plus
                if remaining_fsa_steps < float('inf')
                else float('inf')
            )
            for r in range(GRID):
                for c in range(GRID):
                    # formula: FSA-step lower bound (Formulation §6)
                    heuristic_values[(r, c, q)] = heuristic_value

        self._zero_terminal_heuristic_values(heuristic_values)
        self._tq_sizes = {
            q: int(step_distance.get(q, float('inf')))
            if step_distance.get(q, float('inf')) < float('inf')
            else 0
            for q in self._active_q
        }
        return heuristic_values

    def _compute_h6(self):
        """build the admissible risk-only heuristic h6."""
        heuristic_values = {}
        target_set_sizes = {}
        for q in self._active_q:
                # Step 1: build t_q, the cells that could still advance the FSA.
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
                # Step 2: reverse Dijkstra accumulates only the risk term.
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
        # Step 1: activate only the requested lower bound.
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
                # Step 1: precompute the destination-cell survival toll.
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
                # Step 2: precompute the destination-cell mission logcosts.
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
                    # Step 3: stitch mission logcost and risk toll into each edge.
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
                # Step 1: mission beliefs drive B_en changes.
                for atomic_proposition in mission_beliefs[(r, c)]:
                    if abs(
                        mission_beliefs[(r, c)][atomic_proposition]
                        - self._snap_mission[(r, c)].get(atomic_proposition, 0.5)
                    ) > 1e-9:
                        changed.add((r, c))
                        break
                if (r, c) in changed:
                    continue
                # Step 2: risk beliefs drive gamma * survival changes.
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
                # Step 1: under-consistent state becomes locally optimal.
                self.g[state] = self.rhs[state]
                for predecessor_state in self._pred_nodes((r, c)):
                    self._update(predecessor_state)
            else:
                # Step 2: over-consistent state is reopened and pushed backward.
                self.g[state] = float('inf')
                self._update(state)
                for predecessor_state in self._pred_nodes((r, c)):
                    self._update(predecessor_state)

    def replan(self, beliefs, risk_beliefs=None):
        """incrementally replan after belief updates."""
        # Step 1: refresh the split belief snapshots from the new epoch.
        self._set_beliefs(beliefs, risk_beliefs)
        self.n_expanded = 0

        # Step 2: skip all queue work if the belief snapshot is unchanged.
        changed = self._changed_cells(self._mission_beliefs, self._risk_beliefs)

        if not changed:
            return self.solution()

        # Step 3: rebuild only the edge-local caches that depend on beliefs.
        self._refresh_successor_cache()
        if self._use_h2 or self._use_h6:
            self._refresh_heuristic()

        # Step 4: only predecessors of changed cells can need rhs repairs.
        affected_states = {
            predecessor_state
            for changed_cell in changed
            for predecessor_state in self._pred_nodes(changed_cell)
        }

        for affected_state in affected_states:
            self._update(affected_state)

        # Step 5: resume the incremental shortest-path repair.
        self._run()
        self._snap_mission = self._take_snap(self._mission_beliefs)
        self._snap_risk = self._take_snap(self._risk_beliefs)
        return self.solution()

    def solution(self):
        """extract v, policy, and expansion count from current g values."""
        # Step 1: convert current path costs into the stored exp(-g) value map.
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
            # Step 2: extract the greedy one-step rover action from successor costs.
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

    # Step 1: initialise the product-state value map and cache source-cell B_en.
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
        # Step 2: run one full Bellman backup sweep over the product graph.
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

        # Step 3: stop once the value updates have converged.
        if delta < tol:
            return V, rover_policy, sweep + 1
    return V, rover_policy, vi_steps
