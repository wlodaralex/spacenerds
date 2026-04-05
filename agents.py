"""
agents.py

- execute rover and copter motion after planning chooses a policy or action sequence
- update the shared belief map from realized sensing

class
- `CopterPlayer`: shared copter path helper base
- `BaselineRoverPlayer`: baseline rover runtime wrapper
- `BaselineCopterPlayer`: baseline copter runtime wrapper
- `GameRoverPlayer`: game rover runtime wrapper
- `GameCopterPlayer`: game copter runtime wrapper

function
- `rover_sense`: update local rover beliefs
- `copter_sense`: update local copter obstacle beliefs
"""

import copy
import math

from scenario import SimulationScenario
from environment import (
    GRID,
    AP_LIST,
    ACTIONS,
    TRUE_L,
    clip,
    transition_probabilities,
)
from sensor import sensor_beta, bayes_update, observe
from fsa import FSA_ACCEPT, FSA_DEAD, FSA_ALL, compute_B_en, fsa_step
from planning import (DStarLitePlanner,
                      grid_to_graph,
                      rover_value_iteration,
                      plan_copter_action_game,
                      plan_copter_action_global,
                      plan_copter_action_local)

# --- Sensing (Common) ---
def _sense_grid(beliefs: dict, pos: tuple, R: int, M: float, ap_list) -> None:
    """Fuse noisy observations into beliefs for the given APs within range R."""
    for r in range(max(0, pos[0] - R), min(GRID, pos[0] + R + 1)):
        for c in range(max(0, pos[1] - R), min(GRID, pos[1] + R + 1)):
            beta = sensor_beta(pos, (r, c), R, M)
            if beta <= 0.5:
                continue
            for ap in ap_list:
                z = observe(TRUE_L[(r, c)], ap, beta)
                beliefs[(r, c)][ap] = bayes_update(beliefs[(r, c)][ap], z, beta)


def rover_sense(beliefs: dict, pos: tuple) -> None:
    """update all local beliefs visible to the rover."""
    _sense_grid(beliefs, pos, R=2, M=0.5, ap_list=AP_LIST)


def copter_sense(beliefs: dict, pos: tuple) -> None:
    """update local obstacle beliefs visible to the copter."""
    _sense_grid(beliefs, pos, R=4, M=0.4, ap_list=['O'])


# --- Player Wrappers ---
class CopterPlayer:
    """Shared copter path helpers used by both runtime wrappers."""

    @staticmethod
    def _move_copter(pos: list, action: int) -> list:
        """Apply one deterministic copter action."""
        dr, dc = ACTIONS[action]
        nr, nc = clip(pos[0] + dr, pos[1] + dc)
        return [nr, nc]

    @staticmethod
    def scout_distance(substeps, start_pos=None):
        """Return total copter scout distance."""
        if not substeps:
            return 0.0

        if start_pos is None:
            if len(substeps) < 2:
                return 0.0
            full_path = substeps
        else:
            full_path = [start_pos] + list(substeps)

        return sum(
            math.hypot(full_path[i][0] - full_path[i + 1][0],
                       full_path[i][1] - full_path[i + 1][1])
            for i in range(len(full_path) - 1)
        )

    @staticmethod
    def return_distance(substeps, rover_pos):
        """Return frozen return distance to the rover reference pose."""
        if not substeps:
            return 0.0
        return math.hypot(substeps[-1][0] - rover_pos[0],
                          substeps[-1][1] - rover_pos[1])


class BaselineRoverPlayer:
    """Baseline rover runtime wrapper around the standalone VI planner."""

    def __init__(self, scenario: SimulationScenario):
        self.T_r = scenario.T_r
        self.vi_steps = scenario.vi_steps or 80
        self.V = {}
        self.policy = {}
        self.n_expanded = 0
        self.b_max = {}

    @staticmethod
    def _compute_b_max(beliefs, rover_policy, rover_pos, rover_q, T_r):
        """Compute the baseline-only rover reachability field ``b_max``."""
        # baseline rover execution holds one fixed belief snapshot for this rollout,
        # so cache each source-cell B_en(q' | x, q) once up front.
        B_en_cache = {
            (r, c, q): compute_B_en(beliefs[(r, c)], q)
            for r in range(GRID)
            for c in range(GRID)
            for q in FSA_ALL
        }

        dist = {(rover_pos[0], rover_pos[1], rover_q): 1.0}
        b_max = {
            (r, c): sum(dist.get((r, c, q), 0.0) for q in FSA_ALL)
            for r in range(GRID)
            for c in range(GRID)
        }

        for _ in range(T_r):
            dist_new = {}
            for (r, c, q), prob in dist.items():
                if prob < 1e-9:
                    continue
                if q in FSA_ACCEPT or q == FSA_DEAD:
                    dist_new[(r, c, q)] = dist_new.get((r, c, q), 0.0) + prob
                    continue

                action = rover_policy.get((r, c, q), 1)
                ben_source = B_en_cache[(r, c, q)]
                transition_distribution = transition_probabilities(r, c, action, 1.0)
                for (r2, c2), p_move in transition_distribution.items():
                    for q_next, p_fsa in ben_source.items():
                        next_state = (r2, c2, q_next)
                        dist_new[next_state] = (
                            dist_new.get(next_state, 0.0) + prob * p_move * p_fsa
                        )
            dist = dist_new
            for r in range(GRID):
                for c in range(GRID):
                    cell_reach_probability = sum(
                        dist.get((r, c, q), 0.0)
                        for q in FSA_ALL
                    )
                    if cell_reach_probability > b_max[(r, c)]:
                        b_max[(r, c)] = cell_reach_probability
        return b_max

    def initialise(self, beliefs: dict, rover_pos: tuple, rover_q: int):
        """Build the initial baseline rover value map, policy, and b_max field."""
        self.V, self.policy, self.n_expanded = rover_value_iteration(
            beliefs,
            vi_steps=self.vi_steps,
            T_r=self.T_r,
        )
        self.b_max = self._compute_b_max(
            beliefs, self.policy, rover_pos, rover_q, self.T_r
        )
        print(f"  Initial VI (baseline): {self.n_expanded} expanded.")

    def replan(self, beliefs_t_plus, beliefs_t, rover_pos, rover_q):
        """Replan the baseline rover and refresh b_max.

        `beliefs_t` is kept for interface parity with the game path but is
        unused on the baseline path.
        """
        self.V, self.policy, self.n_expanded = rover_value_iteration(
            beliefs_t_plus,
            vi_steps=self.vi_steps,
            T_r=self.T_r,
        )
        self.b_max = self._compute_b_max(
            beliefs_t_plus, self.policy, rover_pos, rover_q, self.T_r
        )

    def execute(self, beliefs, rover_pos, rover_q):
        """Execute the baseline rover policy for one epoch."""
        pos = list(rover_pos)
        q = rover_q
        substeps = []
        belief_snapshots = []

        q_prev = q
        q = fsa_step(q, TRUE_L[tuple(pos)])
        if q != q_prev:
            print(
                f"    FSA transition: q={q_prev} -> q={q} "
                f"at pos={tuple(pos)}, labels={TRUE_L[tuple(pos)]}"
            )

        rover_sense(beliefs, tuple(pos))
        substeps.append((tuple(pos), q))
        # The epoch-start state is logged as a substep, but belief snapshots only
        # track realized post-move rover ticks.
        if q in FSA_ACCEPT or q == FSA_DEAD:
            return tuple(pos), q, substeps, belief_snapshots

        for _ in range(self.T_r):
            if q in FSA_ACCEPT or q == FSA_DEAD:
                break

            a = self.policy.get((pos[0], pos[1], q), 1)
            dr, dc = ACTIONS[a]
            nr, nc = clip(pos[0] + dr, pos[1] + dc)
            pos = [nr, nc]

            q_prev = q
            q = fsa_step(q, TRUE_L[tuple(pos)])
            if q != q_prev:
                print(
                    f"    FSA transition: q={q_prev} -> q={q} "
                    f"at pos={tuple(pos)}, labels={TRUE_L[tuple(pos)]}"
                )

            rover_sense(beliefs, tuple(pos))
            substeps.append((tuple(pos), q))
            belief_snapshots.append(copy.deepcopy(beliefs))

        return tuple(pos), q, substeps, belief_snapshots


class BaselineCopterPlayer(CopterPlayer):
    """
    Object wrapper for the baseline copter exploration policy.

    The constructor stores only static baseline configuration. The evolving
    belief map is passed to `explore(...)` each epoch because it changes
    after every sensing phase.
    """

    def __init__(self, scenario: SimulationScenario):
        self.copter_mode = scenario.copter_mode
        self.T_c = scenario.T_c
        self.alpha = scenario.alpha
        self.vi_steps = scenario.vi_steps or 80

    def _explore_local(self, beliefs, copter_pos, b_max):
        """Run the baseline local copter exploration phase."""
        pos = list(copter_pos)
        substeps = []
        belief_snapshots = []

        for _ in range(self.T_c):
            best_action = plan_copter_action_local(
                beliefs, tuple(pos), b_max, alpha=self.alpha
            )
            pos = self._move_copter(pos, best_action)

            copter_sense(beliefs, tuple(pos))
            substeps.append(tuple(pos))
            belief_snapshots.append(copy.deepcopy(beliefs))

        # Baseline always commits a fresh local action, so the unified
        # ``action_changed`` field is always true on this path.
        return tuple(pos), substeps, belief_snapshots, True

    def _explore_global(self, beliefs, copter_pos, b_max):
        """Run the baseline global copter exploration phase."""
        pos = list(copter_pos)
        substeps = []
        belief_snapshots = []
        steps_used = 0
        max_copter_vi_sweeps = 0

        while steps_used < self.T_c:
            target_cell, copter_policy, copter_vi_sweeps = (
                plan_copter_action_global(
                    beliefs,
                    tuple(pos),
                    b_max,
                    alpha=self.alpha,
                    vi_steps=self.vi_steps,
                )
            )
            if target_cell is None:
                break
            max_copter_vi_sweeps = max(max_copter_vi_sweeps, copter_vi_sweeps)

            while tuple(pos) != target_cell and steps_used < self.T_c:
                action = copter_policy.get(tuple(pos), 0)
                pos = self._move_copter(pos, action)

                copter_sense(beliefs, tuple(pos))
                substeps.append(tuple(pos))
                belief_snapshots.append(copy.deepcopy(beliefs))
                steps_used += 1

        print(
            f"  Copter value iteration maximum sweeps used "
            f"{max_copter_vi_sweeps}/{self.vi_steps}."
        )

        # baseline global mode also commits a fresh epoch plan every time.
        return tuple(pos), substeps, belief_snapshots, True

    def explore(self, beliefs: dict, copter_pos: tuple, b_max: dict) -> tuple:
        """Run one baseline copter exploration phase."""
        if self.copter_mode == 'local':
            return self._explore_local(beliefs, copter_pos, b_max)
        if self.copter_mode == 'global':
            return self._explore_global(beliefs, copter_pos, b_max)
        raise ValueError(
            f"Unsupported baseline copter_mode={self.copter_mode!r}; "
            "baseline supports only 'local' or 'global'."
        )


class GameRoverPlayer:
    """Game-theoretic rover runtime wrapper around D* Lite."""

    def __init__(self, scenario: SimulationScenario):
        self.T_r = scenario.T_r
        self.gamma = scenario.gamma
        self.tau_r = scenario.tau_r
        self.spatial = grid_to_graph()
        self.planner = None
        self.V = {}
        self.policy = {}
        self.n_expanded = 0

    def initialise(self, beliefs: dict):
        """Build the initial game-theoretic rover planner."""
        self.planner = DStarLitePlanner(
            self.spatial,
            beliefs,
            gamma = self.gamma,
            tau_r = self.tau_r,
        )
        self.V, self.policy, self.n_expanded = self.planner.solution()
        print(f"  Initial D* Lite: {self.n_expanded} expanded.")

    def replan(self, beliefs_t_plus, beliefs_t):
        """Replan the game-theoretic rover on the split-belief map."""
        self.V, self.policy, self.n_expanded = self.planner.replan(
            beliefs_t_plus,
            risk_beliefs = beliefs_t,
        )

    def execute(self, beliefs, rover_pos, rover_q):
        """Execute the game-theoretic rover policy for one epoch."""
        pos = list(rover_pos)
        q = rover_q
        substeps = [(tuple(pos), q)]
        belief_snapshots = []

        for _ in range(self.T_r):
            if q in FSA_ACCEPT or q == FSA_DEAD:
                break

            a = self.policy.get((pos[0], pos[1], q), 1)
            dr, dc = ACTIONS[a]
            nr, nc = clip(pos[0] + dr, pos[1] + dc)
            pos = [nr, nc]

            q = fsa_step(q, TRUE_L[tuple(pos)])
            rover_sense(beliefs, tuple(pos))
            substeps.append((tuple(pos), q))
            belief_snapshots.append(copy.deepcopy(beliefs))

        return tuple(pos), q, substeps, belief_snapshots


class GameCopterPlayer(CopterPlayer):
    """
    Object wrapper for the game-theoretic copter exploration policy.

    The constructor stores only static game parameters. The current belief
    map, rover position, and remaining distance budget are epoch-dependent state, so
    they are passed to `explore(...)` rather than fixed at initialization.
    """

    def __init__(self, scenario: SimulationScenario):
        self.T_c = scenario.T_c
        self.alpha = scenario.alpha
        self.lambda_ = scenario.lambda_
        self.rho = scenario.rho

    def explore(self, beliefs: dict, copter_pos: tuple,
                rover_pos: tuple = (0, 0),
                distance_budget: float = float('inf')) -> tuple:
        """Run one game-theoretic copter exploration phase."""
        pos = list(copter_pos)
        substeps = []
        belief_snapshots = []

        action_sequence = plan_copter_action_game(
            beliefs,
            tuple(pos),
            self.T_c,
            alpha=self.alpha,
            lambda_=self.lambda_,
            rover_pos=rover_pos,
            distance_budget=distance_budget,
            rho=self.rho,
        )
        if action_sequence is None:
            return tuple(copter_pos), [], [], False

        for action in action_sequence:
            pos = self._move_copter(pos, action)
            copter_sense(beliefs, tuple(pos))
            substeps.append(tuple(pos))
            belief_snapshots.append(copy.deepcopy(beliefs))

        return tuple(pos), substeps, belief_snapshots, True
