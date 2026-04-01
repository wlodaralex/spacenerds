"""structured simulation history for plots, csv export, and analysis."""

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EpochRecord:
    """one planning epoch."""
    epoch:          int
    k_start:        int          # timestep at epoch start
    k_end:          int          # timestep at epoch end

    # positions
    rover_pos:      tuple        # rover position after execution
    copter_pos:     tuple        # copter position after exploration
    rover_q:        int          # FSA state after execution

    # copter trajectory
    c_substeps:     list         # [(r,c), ...] copter path this epoch
    action_changed: bool         # did copter commit a new trajectory?

    # rover trajectory
    r_substeps:     list         # [((r,c), q), ...] rover path this epoch

    # metrics
    Phi:            float        # potential function
    G:              float        # log V_phi (mission value)
    C_r:            float        # rover risk cost
    C_c:            float        # copter flight cost
    I:              float        # information gain
    u_c:            float        # copter utility
    E_c:            float        # remaining copter energy

    # planning
    n_expanded:     int = 0      # D* Lite nodes expanded this epoch


class SimHistory:
    """run history with epoch summaries and timestep traces."""

    def __init__(self, rover_start=(0,0), copter_start=(0,0), **params):
        self.params = {
            'rover_start': rover_start,
            'copter_start': copter_start,
            **params,
        }
        self.meta = {}

        self.epochs: list[EpochRecord] = []
        self.substeps: list[dict] = []
        self.belief_snapshots: list[dict] = []
        self.snap_meta: list[dict] = []
        self.belief_history: Optional[list] = None
        self.complete = False
        self.fail = False
        self.which_phi = None
        self.k_final = 0

    def enable_belief_history(self):
        """enable per-step belief recording."""
        self.belief_history = []

    def record_substep(self, k: int, phase: str,
                       rover_pos: tuple, copter_pos: tuple,
                       beliefs=None):
        """record one timestep."""
        self.substeps.append({
            'k': k, 'phase': phase,
            'rover_pos': rover_pos, 'copter_pos': copter_pos,
        })
        if self.belief_history is not None and beliefs is not None:
            self.belief_history.append(copy.deepcopy(beliefs))

    def record_epoch(self, rec: EpochRecord):
        """record one epoch summary."""
        self.epochs.append(rec)

    def snapshot(self, beliefs, k, rover_pos, copter_pos):
        """save one sparse belief snapshot."""
        self.belief_snapshots.append(copy.deepcopy(beliefs))
        self.snap_meta.append({
            'k': k, 'rover_pos': rover_pos, 'copter_pos': copter_pos,
        })

    def finalise(self, complete, fail, which_phi, k_final,
                 rover_pos, copter_pos, rover_q):
        """seal the run after termination."""
        self.complete = complete
        self.fail = fail
        self.which_phi = which_phi
        self.k_final = k_final
        self.params['final_rover'] = rover_pos
        self.params['final_copter'] = copter_pos
        self.params['final_q'] = rover_q

    @staticmethod
    def _polyline_length(path: list[tuple]) -> float:
        """return euclidean polyline length."""
        if len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(1, len(path)):
            prev_r, prev_c = path[i - 1]
            r, c = path[i]
            length += math.hypot(r - prev_r, c - prev_c)
        return length

    def rover_path_length(self) -> float:
        """return total rover path length."""
        return self._polyline_length([s['rover_pos'] for s in self.substeps])

    def copter_path_length(self) -> float:
        """return total copter path length."""
        return self._polyline_length([s['copter_pos'] for s in self.substeps])

    def to_dataframe(self):
        """convert epoch records to a dataframe."""
        import pandas as pd
        gamma = self.params.get('gamma')
        lambda_ = self.params.get('lambda_')
        rover_path_length = self.rover_path_length()
        copter_path_length = self.copter_path_length()
        solve_time_s = self.meta.get('solve_time_s')

        rows = []
        for e in self.epochs:
            rows.append({
                'epoch': e.epoch, 'k_start': e.k_start, 'k_end': e.k_end,
                'rover_r': e.rover_pos[0], 'rover_c': e.rover_pos[1],
                'copter_r': e.copter_pos[0], 'copter_c': e.copter_pos[1],
                'q': e.rover_q,
                'gamma': gamma,
                'lambda': lambda_,
                'Phi': e.Phi, 'G': e.G,
                'C_r': e.C_r, 'C_c': e.C_c, 'I': e.I, 'u_c': e.u_c,
                'E_c': e.E_c,
                'action_changed': e.action_changed,
                'n_expanded': e.n_expanded,
                'copter_flight_steps': len(e.c_substeps),
                'rover_steps': len(e.r_substeps) - 1,
                'rover_path_length': rover_path_length,
                'copter_path_length': copter_path_length,
                'solve_time_s': solve_time_s,
            })
        return pd.DataFrame(rows)

    def to_legacy_dict(self) -> dict:
        """convert to the older dict format used by visualization code."""
        h = {
            'rover_path': [self.params['rover_start']],
            'copter_path': [self.params['copter_start']],
            'rover_substeps': [s['rover_pos'] for s in self.substeps],
            'copter_substeps': [s['copter_pos'] for s in self.substeps],
            'fsa_states': [0] + [e.rover_q for e in self.epochs],
            'k_list': [s['k'] for s in self.substeps],
            'phase_list': [s['phase'] for s in self.substeps],
            'beliefs_snapshot': self.belief_snapshots,
            'snap_k': [m['k'] for m in self.snap_meta],
            'snap_rover': [m['rover_pos'] for m in self.snap_meta],
            'snap_copter': [m['copter_pos'] for m in self.snap_meta],
            'complete': self.complete,
            'fail': self.fail,
            'k_final': self.k_final,
            'which_phi': self.which_phi,
            'belief_history': self.belief_history,
            'Phi': [e.Phi for e in self.epochs],
            'C_r': [e.C_r for e in self.epochs],
            'C_c': [e.C_c for e in self.epochs],
            'I_vals': [e.I for e in self.epochs],
            'u_c_vals': [e.u_c for e in self.epochs],
            'action_changed': [e.action_changed for e in self.epochs],
            'E_c': [e.E_c for e in self.epochs],
            'final_rover': self.params.get('final_rover'),
            'final_copter': self.params.get('final_copter'),
            'final_q': self.params.get('final_q'),
            'scenario_name': self.meta.get('scenario_name',
                                           self.params.get('scenario_name', '')),
            'gamma': self.params.get('gamma'),
            'lambda': self.params.get('lambda_'),
            'lambda_': self.params.get('lambda_'),
            'rover_path_length': self.rover_path_length(),
            'copter_path_length': self.copter_path_length(),
            'solve_time_s': self.meta.get('solve_time_s'),
        }
        for e in self.epochs:
            h['rover_path'].append(e.rover_pos)
            h['copter_path'].append(e.copter_pos)
        h.update(self.meta)
        return h


    def save(self, name: str, outdir: str = 'outputs'):
        """save the epoch table to disk."""
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)

        # save the epoch table only
        try:
            df = self.to_dataframe()
            df.to_csv(outdir / f'{name}.csv', index=False)
        except ImportError:
            pass  # pandas not available; skip CSV

    def summary(self) -> str:
        """return a one-line run summary."""
        status = ('DONE' if self.complete else
                  'FAIL' if self.fail else 'TIMEOUT')
        n = len(self.epochs)
        return (f"{status:7s} k={self.k_final:4d}  "
                f"epochs={n:3d}  "
                f"phi={self.which_phi or '-':15s}  "
                f"Φ_final={self.epochs[-1].Phi:.3f}" if self.epochs else
                f"{status:7s} k={self.k_final:4d}  epochs=0")

    def __getitem__(self, key):
        if key == 'epochs':
            return self.epochs
        if key == 'substeps':
            return self.substeps
        if key == 'belief_snapshots':
            return self.belief_snapshots
        if key == 'snap_meta':
            return self.snap_meta
        if key == 'complete':
            return self.complete
        if key == 'fail':
            return self.fail
        if key == 'which_phi':
            return self.which_phi
        if key == 'k_final':
            return self.k_final
        if key == 'belief_history':
            return self.belief_history
        if key in self.meta:
            return self.meta[key]
        if key in self.params:
            return self.params[key]
        legacy = self.to_legacy_dict()
        if key in legacy:
            return legacy[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key in {'complete', 'fail', 'which_phi', 'k_final', 'belief_history'}:
            setattr(self, key, value)
            return
        if key in self.params:
            self.params[key] = value
            return
        self.meta[key] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
