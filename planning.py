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
import numpy as np
# from numpy import random

from environment import GRID, ACTIONS, clip, transition_probabilities
from fsa import FSA_ACCEPT, FSA_DEAD, FSA_ALL, fsa_step, ALL_SUBSETS, compute_B_en


def grid_to_graph() -> dict:
    """build the spatial adjacency graph for the grid."""
    adjacency_by_cell = {}
    for r in range(GRID):
        for c in range(GRID):
            neighbor_cells = []
            for action_index in range(1, len(ACTIONS)):   # skip stay (action 0)
                # step 1 encode the 8-connected spatial graph used by both planners
                dr, dc = ACTIONS[action_index]
                nr, nc = clip(r + dr, c + dc)
                if (nr, nc) != (r, c):
                    neighbor_cells.append(((nr, nc), action_index))
            adjacency_by_cell[(r, c)] = neighbor_cells
    return adjacency_by_cell


_FSA_DIST_CACHE = None

def _fsa_distances() -> dict:
    """cache the minimum automaton hops from each q to acceptance."""
    global _FSA_DIST_CACHE
    if _FSA_DIST_CACHE is not None:
        return _FSA_DIST_CACHE

    distance_to_accept = {q: float('inf') for q in FSA_ALL}
    for q in FSA_ACCEPT:
        distance_to_accept[q] = 0

    reverse_automaton = {q: [] for q in FSA_ALL}
    for q in FSA_ALL:
        if q in FSA_ACCEPT or q == FSA_DEAD:
            continue
        visited_successors = set()
        for label_subset in ALL_SUBSETS:
            # step 1 enumerate unique automaton successors under realized labels
            q_next = fsa_step(q, label_subset)
            if q_next not in visited_successors:
                visited_successors.add(q_next)
                reverse_automaton[q_next].append(q)

    queue = deque(q for q in FSA_ACCEPT)
    while queue:
        # step 2 reverse bfs gives the minimum remaining automaton hops
        q_current = queue.popleft()
        for q_prev in reverse_automaton[q_current]:
            if distance_to_accept[q_prev] == float('inf'):
                distance_to_accept[q_prev] = (
                    distance_to_accept[q_current] + 1
                )
                queue.append(q_prev)
    _FSA_DIST_CACHE = distance_to_accept
    return _FSA_DIST_CACHE


class VIPlanner:
    """full value iteration for the stochastic baseline rover."""

    def __init__(self, spatial, beliefs, vi_steps=80, p_rover=0.95):
        self.spatial = spatial
        self.beliefs = beliefs
        self.vi_steps = vi_steps
        self.p_rover = p_rover
        self.n_expanded = 0
        self._nodes = [
            (r, c, q)
            for r in range(GRID)
            for c in range(GRID)
            for q in FSA_ALL
        ]
        self._state_index_by_node = {
            node: state_index
            for state_index, node in enumerate(self._nodes)
        }
        self._accept_state_index = tuple(
            self._state_index_by_node[node]
            for node in self._nodes
            if node[2] in FSA_ACCEPT
        )
        self._nonterminal_nodes = list(self._nonterminal_states())
        self._nonterminal_state_index = tuple(
            self._state_index_by_node[(r, c, q)]
            for r, c, q in self._nonterminal_nodes
        )
        self._transitions = {
            (r, c, action_index): transition_probabilities(
                r, c, action_index, self.p_rover
            )
            for r in range(GRID)
            for c in range(GRID)
            for action_index in range(len(ACTIONS))
        }
        self.V, self.policy = self._solve(beliefs)

    def _belief_transition_cache(self, beliefs):
        """cache automaton transitions for one fixed belief snapshot."""
        # formula: branch-probability kernel
        # b_en(x', q -> q') is frozen for one value-iteration solve
        return {
            (r, c, q): compute_B_en(
                beliefs[(r, c)], q
            )
            for r in range(GRID)
            for c in range(GRID)
            for q in FSA_ALL
            if q not in FSA_ACCEPT and q != FSA_DEAD
        }

    def _combined_transition_cache(self, beliefs):
        """cache combined spatial and automaton outcomes for one solve.

        this precomputes the sparse bellman kernel

            P_u(i, j) = sum_x' P(x' | x, u) B_en(x', q -> q_j)

        where i indexes one product state (r, c, q) and j indexes one
        destination product state. with that cache, the backup becomes

            V_{k+1}(i) = max_u sum_j P_u(i, j) V_k(j)

        so the inner loop no longer rebuilds tuple-key dict lookups for every
        sweep. the result is exactly the same stochastic backup, only flattened
        into indexed sparse outcomes.
        """
        belief_transition_cache = self._belief_transition_cache(beliefs)
        combined_transition_cache = [None] * len(self._nodes)
        for r, c, q in self._nonterminal_nodes:
            state_index = self._state_index_by_node[(r, c, q)]
            action_outcomes = []
            for action_index in range(len(ACTIONS)):
                destination_probability = {}
                for (nr, nc), transition_probability in self._transitions[
                    (r, c, action_index)
                ].items():
                    for q_next, automaton_probability in belief_transition_cache[
                        (nr, nc, q)
                    ].items():
                        destination_index = self._state_index_by_node[
                            (nr, nc, q_next)
                        ]
                        destination_probability[destination_index] = (
                            destination_probability.get(destination_index, 0.0)
                            + transition_probability * automaton_probability
                        )
                action_outcomes.append(tuple(destination_probability.items()))
            combined_transition_cache[state_index] = tuple(action_outcomes)
        return combined_transition_cache

    @staticmethod
    def _nonterminal_states():
        """iterate over product states that need backups."""
        for r in range(GRID):
            for c in range(GRID):
                for q in FSA_ALL:
                    if q not in FSA_ACCEPT and q != FSA_DEAD:
                        yield r, c, q

    def _action_value(
        self,
        value_by_index,
        combined_transition_cache,
        state_index,
        action_index,
    ):
        """evaluate one bellman term for one action."""
        action_value = 0.0
        for destination_index, outcome_probability in combined_transition_cache[
            state_index
        ][action_index]:
            action_value += outcome_probability * value_by_index[destination_index]
        return action_value

    def _best_action_value(
        self, value_by_index, combined_transition_cache, state_index
    ):
        """return the best bellman value and its action."""
        best_action_index, best_action_value = 0, 0.0
        for action_index in range(len(ACTIONS)):
            candidate_value = self._action_value(
                value_by_index,
                combined_transition_cache,
                state_index,
                action_index,
            )
            if candidate_value > best_action_value:
                best_action_value = candidate_value
                best_action_index = action_index
        return best_action_value, best_action_index

    def _solve(self, beliefs):
        # step 1 freeze the sparse bellman kernel for this belief snapshot
        combined_transition_cache = self._combined_transition_cache(beliefs)

        # step 2 initialize the value function with acceptance as value 1
        value_by_index = [0.0] * len(self._nodes)
        for state_index in self._accept_state_index:
            value_by_index[state_index] = 1.0

        converged_at = self.vi_steps
        for sweep in range(self.vi_steps):
            # step 3 apply one full stochastic bellman sweep
            updated_value_by_index = value_by_index.copy()
            maximum_change = 0.0
            for state_index in self._nonterminal_state_index:
                best_value, _ = self._best_action_value(
                    value_by_index,
                    combined_transition_cache,
                    state_index,
                )
                updated_value_by_index[state_index] = best_value
                maximum_change = max(
                    maximum_change,
                    abs(best_value - value_by_index[state_index]),
                )
            value_by_index = updated_value_by_index
            if maximum_change < 1e-6:
                converged_at = sweep + 1
                break

        # step 4 report the number of bellman sweeps as the planner workload
        self.n_expanded = converged_at

        # step 5 extract the greedy action under the converged value function
        policy = {}
        for state_index, node in enumerate(self._nodes):
            r, c, q = node
            if q in FSA_ACCEPT or q == FSA_DEAD:
                policy[node] = 0
                continue
            _, best_action_index = self._best_action_value(
                value_by_index,
                combined_transition_cache,
                state_index,
            )
            policy[node] = best_action_index
        value_function = {
            node: value_by_index[state_index]
            for state_index, node in enumerate(self._nodes)
        }
        return value_function, policy

    def solution(self):
        """return the current planner solution."""
        return self.V, self.policy, self.n_expanded

    def replan(self, beliefs, risk_beliefs=None):
        """re-solve from scratch after belief updates."""
        # step 1 vi is non-incremental, so replanning is a full resolve
        self.beliefs = beliefs
        self.V, self.policy = self._solve(beliefs)
        return self.V, self.policy, self.n_expanded


class DStarLitePlanner:
    """incremental d* lite on the product graph."""

    def __init__(self, spatial, beliefs, gamma=0.0, tau_r=1.0,
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
        # formula: split-belief rover utility
        # mission beliefs carry b_t^+ for g, risk beliefs carry b_t for c_r
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
        # formula: progress alphabet
        # these are the atomic propositions that can still advance the mission
        required_atomic_propositions = {
            0: ('A', 'B', 'C'),
            1: ('C', 'A'),
            2: ('D', 'A'),
        }
        return required_atomic_propositions.get(q, ())

    def _key(self, state):
        # formula: d* queue key
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
                        # formula: chebyshev lower bound
                        chebyshev_distance = min(
                            max(abs(r - target_r), abs(c - target_c))
                            for target_r, target_c in target_cells
                        )
                    else:
                        chebyshev_distance = GRID
                    heuristic_values[(r, c, q)] = (
                        chebyshev_distance * self._c_min_plus
                    )

        for r in range(GRID):
            for c in range(GRID):
                for q in FSA_ACCEPT:
                    heuristic_values[(r, c, q)] = 0.0
                heuristic_values[(r, c, FSA_DEAD)] = 0.0

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
                    # formula: risk-only corridor bound
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

        for r in range(GRID):
            for c in range(GRID):
                for q in FSA_ACCEPT:
                    heuristic_values[(r, c, q)] = 0.0
                heuristic_values[(r, c, FSA_DEAD)] = 0.0

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
            # formula: one-step rhs backup
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
            # formula: exp-of-cost value map
            # v = exp(-g) is the multiplicative path score recovered from d*
            value_function[node] = (
                0.0 if path_cost == float('inf') else math.exp(-path_cost)
            )
            if q in FSA_ACCEPT or q == FSA_DEAD:
                policy[node] = 0
                continue
            best_action_index, best_total_cost = 1, float('inf')
            for destination_state, action_index, edge_cost in self._succ(node):
                # formula: greedy q backup
                total_cost = edge_cost + self.g.get(destination_state, float('inf'))
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_action_index = action_index
            policy[node] = best_action_index
        return value_function, policy, self.n_expanded

def d_star_lite_plan(spatial, beliefs, gamma=0.0, tau_r=1.0, risk_beliefs=None):
    """run one one-shot d* lite solve."""
    planner = DStarLitePlanner(
        spatial, beliefs, gamma, tau_r, risk_beliefs=risk_beliefs
    )
    return planner.solution()

def compute_b_max(spatial: dict, beliefs: dict, policy: dict,
                  rover_pos: tuple, rover_q: int, T_r: int) -> dict:
    """compute the rover reachability belief used by the Hashimoto copter."""
    # formula: forward reachability plume
    reach_probability = {(rover_pos[0], rover_pos[1], rover_q): 1.0}
    b_max = {(r, c): 0.0 for r in range(GRID) for c in range(GRID)}
    b_max[(rover_pos[0], rover_pos[1])] = 1.0

    for _ in range(T_r):
        # step 1 push one rover policy step through the belief-weighted fsa
        next_reach_probability = {}
        for (r, c, q), state_probability in reach_probability.items():
            if state_probability < 1e-9:
                continue
            if q in FSA_ACCEPT or q == FSA_DEAD:
                next_reach_probability[(r, c, q)] = (
                    next_reach_probability.get((r, c, q), 0.0)
                    + state_probability
                )
                continue

            action_index = policy.get((r, c, q), 1)
            destination_by_action = {
                available_action_index: destination_cell
                for destination_cell, available_action_index in spatial[(r, c)]
            }
            nr, nc = destination_by_action.get(
                action_index, (r, c)
            )
            for q_next, automaton_probability in compute_B_en(
                beliefs[(nr, nc)], q
            ).items():
                # step 2 spread mass over automaton successors at the destination
                next_state = (nr, nc, q_next)
                next_reach_probability[next_state] = (
                    next_reach_probability.get(next_state, 0.0)
                    + state_probability * automaton_probability
                )

        reach_probability = next_reach_probability

        for r in range(GRID):
            for c in range(GRID):
                # step 3 b_max keeps the best cell reachability seen so far
                cell_reach_probability = sum(
                    reach_probability.get((r, c, q), 0.0)
                    for q in FSA_ALL
                )
                if cell_reach_probability > b_max[(r, c)]:
                    b_max[(r, c)] = cell_reach_probability

    return b_max
