"""
simulation.py

- dispatch one scenario through a simulation runner
- run the copter phase, rover replan, rover execution, and history logging

class
- `State`: mutable rollout state and history handle
- `SimulationRunner`: one scenario rollout driver

function
- `run_simulation`: public simulation entry point
"""

import copy
from dataclasses import dataclass

import numpy as np

from environment import GRID
from history import EpochRecord, SimHistory
from planning import binary_entropy, copter_footprint_mask
from scenario import SimulationScenario
from sensor import init_beliefs
from fsa import FSA_ACCEPT, FSA_DEAD
from agents import (
    copter_sense,
    CopterPlayer,
    BaselineRoverPlayer,
    GameRoverPlayer,
    BaselineCopterPlayer,
    GameCopterPlayer,
)


# --- Constants ---
_PHI_LABELS = {3: 'φ1 (found A)', 4: 'φ2 (B→C)', 5: 'φ3 (C→D)'}


# --- Public API ---
def run_simulation(scenario: SimulationScenario) -> SimHistory:
    """Run one full rover-copter simulation for the given scenario."""
    return SimulationRunner(scenario).run()


# --- State ---
@dataclass
class State:
    """Mutable public state and history handle for one rollout."""

    rover_pos: tuple
    copter_pos: tuple
    beliefs: dict
    rover_q: int
    k: int
    history: SimHistory

    @classmethod
    def from_scenario(cls, scenario: SimulationScenario):
        """Bootstrap the public state and initial history at k = 0."""
        np.random.seed(scenario.seed)

        rover_pos = scenario.rover_start
        copter_pos = scenario.copter_start
        beliefs = init_beliefs()

        history = SimHistory(
            rover_start=scenario.rover_start,
            copter_start=scenario.copter_start,
            T_c=scenario.T_c,
            T_r=scenario.T_r,
            alpha=scenario.alpha,
            gamma=scenario.gamma,
            lambda_=scenario.lambda_,
            rho=scenario.rho,
            tau_r=scenario.tau_r,
            vi_steps=scenario.vi_steps,
            max_k=scenario.max_k,
            seed=scenario.seed,
            copter_mode=scenario.copter_mode,
            D_c_budget=scenario.D_c,
            scenario_name=scenario.name,
        )
        history.enable_belief_history()

        copter_sense(beliefs, copter_pos)
        history.snapshot(beliefs, 0, rover_pos, copter_pos)
        history.record_substep(0, 'I', rover_pos, copter_pos, beliefs)

        return cls(
            rover_pos=rover_pos,
            copter_pos=copter_pos,
            beliefs=beliefs,
            rover_q=0,
            k=0,
            history=history,
        )

    def finalise(self, complete: bool, max_k: int):
        """Seal the final rollout state into the history object."""
        fail = self.rover_q == FSA_DEAD
        if not complete and not fail:
            print(f"  Time limit reached at k={max_k}")

        if self.history.snap_meta[-1]['k'] < self.k:
            self.history.snapshot(
                self.beliefs, self.k, self.rover_pos, self.copter_pos
            )

        if not self.history.substeps or self.history.substeps[-1]['k'] != self.k:
            self.history.record_substep(
                self.k, 'E', self.rover_pos, self.copter_pos, self.beliefs
            )

        which_phi = _PHI_LABELS.get(self.rover_q)
        self.history.finalise(
            complete,
            fail,
            which_phi,
            self.k,
            self.rover_pos,
            self.copter_pos,
            self.rover_q,
        )
        return self.history


# --- Simulation Runner ---
class SimulationRunner:
    """Run one scenario through the baseline or game-theoretic epoch loop."""

    def __init__(self, scenario: SimulationScenario):
        self.scenario = scenario
        self.is_baseline = self._uses_baseline_path()
        self.state = State.from_scenario(scenario)
        self.complete = False
        self.D_c = scenario.D_c

        if self.is_baseline:
            self.rover_player = BaselineRoverPlayer(scenario)
            self.copter_player = BaselineCopterPlayer(scenario)
            self.planner_label = 'VI (baseline)'
            self.rover_player.initialise(
                beliefs=self.state.beliefs,
                rover_pos=self.state.rover_pos,
                rover_q=self.state.rover_q,
            )
        else:
            self.rover_player = GameRoverPlayer(scenario)
            self.copter_player = GameCopterPlayer(scenario)
            self.planner_label = 'D* Lite'
            self.rover_player.initialise(beliefs=self.state.beliefs)

    def run(self) -> SimHistory:
        """Run the configured scenario and return its structured history."""
        if self.is_baseline:
            return self._run_baseline()
        return self._run_game()

    def _run_baseline(self) -> SimHistory:
        """Run the standalone baseline rollout."""
        while self.state.k < self.scenario.max_k and not self.complete:
            # Step 1: freeze epoch-start references for replanning and metrics.
            (
                k_before_copter,
                beliefs_t,
                copter_pos_at_epoch_start,
                rover_pos_at_epoch_start,
            ) = self._epoch_start_refs()

            # Step 2: execute the copter exploration phase under the frozen
            # baseline rover threat map.
            (
                self.state.copter_pos,
                copter_substep_path,
                copter_belief_snapshots,
                action_changed,
            ) = self.copter_player.explore(
                beliefs=self.state.beliefs,
                copter_pos=self.state.copter_pos,
                b_max=self.rover_player.b_max,
            )

            # Step 3: advance the epoch clock through the copter phase.
            self.state.k += self.scenario.T_c

            # Step 4: materialize B_t^+ and log the copter substeps.
            beliefs_t_plus = self.state.beliefs

            self._record_copter_substeps(
                copter_substep_path,
                copter_belief_snapshots,
                k_before_copter,
            )

            # Step 5: replan the rover immediately on B_t^+ versus B_t.
            self.rover_player.replan(
                beliefs_t_plus,
                beliefs_t,
                self.state.rover_pos,
                self.state.rover_q,
            )

            # Step 6: execute the rover segment and log rover substeps.
            k_before_rover = self.state.k
            (
                self.state.rover_pos,
                self.state.rover_q,
                substeps,
                r_bsnaps,
            ) = self.rover_player.execute(
                self.state.beliefs,
                self.state.rover_pos,
                self.state.rover_q,
            )
            self.state.k += self.scenario.T_r

            self._record_rover_substeps(substeps, r_bsnaps, k_before_rover)

            # Step 7: score the epoch, update completion/failure, and stop if
            # the rollout has terminated.
            metrics = self._compute_epoch_metrics(
                substeps,
                copter_substep_path,
                beliefs_t,
                copter_pos_at_epoch_start=copter_pos_at_epoch_start,
                rover_pos_at_epoch_start=rover_pos_at_epoch_start,
            )
            self.complete, failed = self._finish_epoch(
                metrics,
                k_start=k_before_copter,
                copter_substep_path=copter_substep_path,
                substeps=substeps,
                action_changed=action_changed,
            )
            if failed:
                break

        return self.state.finalise(self.complete, self.scenario.max_k)

    def _run_game(self) -> SimHistory:
        """Run the game-theoretic rollout."""
        while self.state.k < self.scenario.max_k and not self.complete:
            # Step 1: freeze epoch-start references for replanning and metrics.
            (
                k_before_copter,
                beliefs_t,
                copter_pos_at_epoch_start,
                rover_pos_at_epoch_start,
            ) = self._epoch_start_refs()

            # Step 2: execute the copter exploration phase under the current
            # game budget and rover state.
            (
                self.state.copter_pos,
                copter_substep_path,
                copter_belief_snapshots,
                action_changed,
            ) = self.copter_player.explore(
                beliefs=self.state.beliefs,
                copter_pos=self.state.copter_pos,
                rover_pos=self.state.rover_pos,
                distance_budget=self.D_c,
            )

            # Step 3: advance the epoch clock and debit the realized copter
            # scouting cost from the remaining distance budget.
            self.state.k += self.scenario.T_c

            if action_changed and self.D_c < float('inf'):
                self.D_c -= CopterPlayer.scout_distance(
                    copter_substep_path,
                    start_pos=copter_pos_at_epoch_start,
                )

            # Step 4: materialize B_t^+ and log the copter substeps.
            beliefs_t_plus = self.state.beliefs

            self._record_copter_substeps(
                copter_substep_path,
                copter_belief_snapshots,
                k_before_copter,
            )

            # Step 5: replan the rover immediately on B_t^+ versus B_t.
            self.rover_player.replan(beliefs_t_plus, beliefs_t)

            # Step 6: execute the rover segment and log rover substeps.
            k_before_rover = self.state.k
            (
                self.state.rover_pos,
                self.state.rover_q,
                substeps,
                r_bsnaps,
            ) = self.rover_player.execute(
                self.state.beliefs,
                self.state.rover_pos,
                self.state.rover_q,
            )
            self.state.k += self.scenario.T_r

            self._record_rover_substeps(substeps, r_bsnaps, k_before_rover)

            # Step 7: score the epoch, update completion/failure, and stop if
            # the rollout has terminated.
            metrics = self._compute_epoch_metrics(
                substeps,
                copter_substep_path,
                beliefs_t,
                copter_pos_at_epoch_start=copter_pos_at_epoch_start,
                rover_pos_at_epoch_start=rover_pos_at_epoch_start,
            )
            self.complete, failed = self._finish_epoch(
                metrics,
                k_start=k_before_copter,
                copter_substep_path=copter_substep_path,
                substeps=substeps,
                action_changed=action_changed,
            )
            if failed:
                break

        return self.state.finalise(self.complete, self.scenario.max_k)

    def _epoch_start_refs(self):
        """Return the frozen references used across one epoch."""
        return (
            self.state.k,
            copy.deepcopy(self.state.beliefs),
            self.state.copter_pos,
            self.state.rover_pos,
        )

    def _uses_baseline_path(self) -> bool:
        """Validate mode settings and return whether this run uses baseline code."""
        if self.scenario.copter_mode in ('local', 'global'):
            if self.scenario.gamma != 0.0 or self.scenario.lambda_ != 0.0:
                raise ValueError(
                    "copter_mode='local'/'global' requires γ=0 and λ=0."
                )
            return True
        if self.scenario.copter_mode == 'utility':
            return False
        raise ValueError(
            f"Unsupported copter_mode={self.scenario.copter_mode!r}; "
            "use 'local', 'global', or 'utility'."
        )

    def _finish_epoch(self, metrics, *, k_start, copter_substep_path,
                      substeps, action_changed):
        """Record one epoch summary, check termination, and print status."""
        self.state.history.record_epoch(EpochRecord(
            epoch=len(self.state.history.epochs),
            k_start=k_start,
            k_end=self.state.k,
            rover_pos=self.state.rover_pos,
            copter_pos=self.state.copter_pos,
            rover_q=self.state.rover_q,
            c_substeps=list(copter_substep_path),
            action_changed=action_changed,
            r_substeps=list(substeps),
            Phi=metrics['Phi'],
            G=metrics['G'],
            C_r=metrics['C_r'],
            C_c=metrics['C_c'],
            I=metrics['I'],
            u_r=metrics['u_r'],
            u_c=metrics['u_c'],
            D_c=self.D_c,
            n_expanded=self.rover_player.n_expanded,
        ))
        if self.state.rover_q in FSA_ACCEPT:
            which = _PHI_LABELS.get(self.state.rover_q, '?')
            print(
                f"   MISSION COMPLETE at k={self.state.k}, branch={which}, "
                f"pos={self.state.rover_pos}  "
                f"[{self.planner_label}={self.rover_player.n_expanded} expanded]"
            )
            return True, False
        if self.state.rover_q == FSA_DEAD:
            print(
                f"   MISSION FAILED (obstacle) at k={self.state.k}, "
                f"pos={self.state.rover_pos}"
            )
            return False, True
        return False, False

    def _compute_epoch_metrics(self, rover_substeps, copter_substep_path,
                               beliefs_t, *,
                               copter_pos_at_epoch_start,
                               rover_pos_at_epoch_start):
        """Compute the logged metrics for one completed rover-copter epoch.

        ``G`` is logged as ``log(V)`` at the rover's realized endpoint. When
        ``gamma > 0``, the stored planner value ``V`` already includes the
        remaining future risk term, so this logged quantity should be read as a
        remaining rover path score rather than a pure mission-only ``log V_phi``.
        The separately logged ``C_r`` covers only the realized path segment of
        the current epoch, so there is no past/future double-counting.
        """
        rover_cells = (
            [substep[0] for substep in rover_substeps[1:]]
            if len(rover_substeps) > 1
            else []
        )

        C_r = (
            -sum(
                np.log(np.clip(1.0 - beliefs_t[pos]['O'], 1e-9, 1.0))
                for pos in rover_cells
            )
            if rover_cells else 0.0
        )

        C_c = (
            CopterPlayer.scout_distance(
                copter_substep_path,
                start_pos=copter_pos_at_epoch_start,
            )
            + self.scenario.rho * CopterPlayer.return_distance(
                copter_substep_path,
                rover_pos_at_epoch_start,
            )
        )

        I = self._copter_information_gain(beliefs_t, copter_substep_path)

        remaining_path_score = self.rover_player.V.get(
            (self.state.rover_pos[0], self.state.rover_pos[1], self.state.rover_q),
            0.0,
        )
        G = np.log(np.clip(remaining_path_score, 1e-9, 1.0))
        u_r = G - self.scenario.gamma * C_r
        Phi = (
            G
            + self.scenario.alpha * I
            - self.scenario.gamma * C_r
            - self.scenario.lambda_ * C_c
        )
        u_c = G + self.scenario.alpha * I - self.scenario.lambda_ * C_c
        return dict(Phi=Phi, G=G, C_r=C_r, C_c=C_c, I=I, u_r=u_r, u_c=u_c)

    @staticmethod
    def _copter_information_gain(beliefs_t, copter_substep_path):
        """Return copter information gain over the committed footprint union."""
        if not copter_substep_path:
            return 0.0

        footprint_mask_by_cell = copter_footprint_mask()
        observed_mask = np.zeros(GRID * GRID, dtype=bool)
        for r, c in copter_substep_path:
            observed_mask |= footprint_mask_by_cell[r * GRID + c]

        entropy_flat = np.array(
            [
                binary_entropy(beliefs_t[(r, c)]['O'])
                for r in range(GRID)
                for c in range(GRID)
            ],
            dtype=np.float64,
        )
        return float(entropy_flat[observed_mask].sum())

    def _record_rover_substeps(self, substeps, r_bsnaps, k_base):
        """Record the scheduled rover phase history for one epoch."""
        rover_substeps = substeps[1:]
        final_rover_pos = substeps[-1][0] if substeps else None
        for step_i in range(self.scenario.T_r):
            # Pad short executions with the final realized state so the history
            # keeps a fixed-width rover phase for visualization and table alignment.
            rover_substep_pos = (
                rover_substeps[step_i][0]
                if step_i < len(rover_substeps)
                else final_rover_pos
            )
            snap = (
                r_bsnaps[step_i]
                if step_i < len(r_bsnaps)
                else copy.deepcopy(self.state.beliefs)
            )
            self.state.history.record_substep(
                k_base + step_i + 1,
                'R',
                rover_substep_pos,
                self.state.copter_pos,
                snap,
            )

    def _record_copter_substeps(self, copter_substep_path,
                                copter_belief_snapshots, k_base):
        """Record the scheduled copter phase history for one epoch."""
        for step_i in range(self.scenario.T_c):
            # Pad short or empty copter paths so the scheduled copter phase still
            # occupies exactly T_c ticks in the recorded trace.
            copter_substep_pos = (
                copter_substep_path[step_i]
                if step_i < len(copter_substep_path)
                else self.state.copter_pos
            )
            snap = (
                copter_belief_snapshots[step_i]
                if step_i < len(copter_belief_snapshots)
                else copy.deepcopy(self.state.beliefs)
            )
            self.state.history.record_substep(
                k_base + step_i + 1,
                'C',
                self.state.rover_pos,
                copter_substep_pos,
                snap,
            )
