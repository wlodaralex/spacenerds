"""main rover-copter simulation loop."""

import copy
import math
import random
import numpy as np

from environment import GRID
from history import EpochRecord, SimHistory
from sensor import init_beliefs
from fsa import FSA_ACCEPT, FSA_DEAD
from planning import grid_to_graph, VIPlanner, DStarLitePlanner, compute_b_max
from agents import (copter_sense, rover_execute, compute_u_c, H,
                    copter_explore_local,
                    copter_explore_global,
                    copter_explore_utility)

_COPTER_MODES = {
    'local':    copter_explore_local,
    'global':   copter_explore_global,
    'utility':  copter_explore_utility,
}

def _flight_cost(substeps):
    """return total copter flight distance."""
    if len(substeps) < 2:
        return 0.0
    # formula: path-length ledger
    # c_c^dist = sum_k ||x_c^{k+1} - x_c^k||_2
    return sum(math.hypot(substeps[i][0] - substeps[i+1][0],
                          substeps[i][1] - substeps[i+1][1])
               for i in range(len(substeps) - 1))


def _return_cost(substeps, rover_pos):
    """return frozen return distance to the rover reference pose."""
    if not substeps:
        return 0.0
    # formula: frozen-return leash
    # rho * d(x_c^{t+t_c}, x_ref) with x_ref frozen at epoch start
    return math.hypot(substeps[-1][0] - rover_pos[0],
                      substeps[-1][1] - rover_pos[1])

def _run_copter(copter_explore, copter_mode, beliefs, copter_pos, b_max,
                T_c, alpha,
                lambda_=0.0, rover_pos=(0, 0), energy=float('inf')):
    """step 5 dispatch to the selected copter solver."""
    # step 5a utility mode needs the private copter terms lambda and c_c
    extra = {}
    if copter_mode == 'utility':
        extra = dict(lambda_=lambda_, rover_pos=rover_pos, energy=energy)
    # step 5b local/global modes use only the shared acquisition score
    return copter_explore(beliefs, copter_pos, b_max, T_c, alpha,
                          record_beliefs=True, **extra)

def _gate_copter_action(c_substeps, copter_pos, beliefs,
                        beliefs_backup, copter_pos_backup,
                        prev_copter_substeps, c_bsnaps,
                        V, rover_pos, rover_q,
                        alpha, gamma, lambda_, tau_r,
                        E_c, T_c):
    """step 6 apply the energy and utility gates."""
    action_changed = True

    # step 6a apply the hard energy gate
    if E_c < float('inf'):
        # formula: hard budget gate
        # c_c^dist + e_return <= e_t^c must hold before any utility logic
        flight = _flight_cost(c_substeps)
        ret = _return_cost(c_substeps, rover_pos)
        if flight + ret > E_c:
            # step 6a reject and keep the copter in place
            beliefs = copy.deepcopy(beliefs_backup)
            copter_pos = copter_pos_backup
            c_substeps = [copter_pos_backup] * T_c
            for s in c_substeps:
                copter_sense(beliefs, s)
            c_bsnaps = []
            action_changed = False
        else:
            E_c -= flight

    # step 6b apply the utility-improvement gate
    if lambda_ > 0 and action_changed and prev_copter_substeps is not None:
        # formula: monotone copter gate
        # accept only if the candidate improves the logged copter utility
        # approximation note: compute_u_c uses the current planner value v as a
        # proxy for the shared mission term g, so this gate is exact for the
        # implemented surrogate and approximate to theorem-level u_c.
        u_c_new = compute_u_c(V, beliefs_backup, rover_pos, rover_q,
                              c_substeps, alpha, lambda_)
        u_c_prev = compute_u_c(V, beliefs_backup, rover_pos, rover_q,
                               prev_copter_substeps, alpha, lambda_)
        if u_c_new <= u_c_prev:
            # step 6b reject and replay the previous trajectory
            beliefs = copy.deepcopy(beliefs_backup)
            copter_pos = copter_pos_backup
            c_substeps = list(prev_copter_substeps)
            c_bsnaps = []
            action_changed = False

    return c_substeps, copter_pos, beliefs, c_bsnaps, E_c, action_changed

def _compute_epoch_metrics(rover_substeps, c_substeps, beliefs_backup,
                           V, rover_pos, rover_q,
                           rover_pos_at_epoch_start,
                           alpha, gamma, lambda_):
    """step 9 compute the logged epoch metrics."""
    # step 9a collect the realized rover cells for this epoch
    rover_cells = [s[0] for s in rover_substeps[1:]] if len(rover_substeps) > 1 else []

    # formula: survival toll
    # c_r = -sum_k log(1 - b_t(x_r^k |= o))
    C_r = (-sum(np.log(np.clip(1.0 - beliefs_backup[pos]['O'], 1e-9, 1.0))
                for pos in rover_cells) if rover_cells else 0.0)

    # formula: copter trip ledger
    # c_c = flight distance + frozen return distance to the epoch-start rover
    C_c = _flight_cost(c_substeps)
    if c_substeps:
        C_c += _return_cost(c_substeps, rover_pos_at_epoch_start)

    # formula: entropy scout bonus
    # i(a_c; b_t) = sum_x h(b_t(x |= o)) over unique observed cells
    I = sum(H(beliefs_backup[pos]['O']) for pos in set(c_substeps))

    # formula: mission logscore
    # g = log v_phi, using the planner value at the rover's realized end state
    V_phi = V.get((rover_pos[0], rover_pos[1], rover_q), 0.0)
    G = np.log(np.clip(V_phi, 1e-9, 1.0))

    # formula: exact-potential ledger
    Phi = G + alpha * I - gamma * C_r - lambda_ * C_c

    # formula: copter utility ledger
    u_c = G + alpha * I - lambda_ * C_c

    return dict(Phi=Phi, G=G, C_r=C_r, C_c=C_c, I=I, u_c=u_c)

def _record_copter_substeps(history, c_substeps, rover_pos, k_base,
                            c_bsnaps, beliefs):
    """record step-level copter history."""
    for step_i, cpos in enumerate(c_substeps):
        # step 6c store either the recorded belief snapshot or a late fallback
        snap = (c_bsnaps[step_i] if step_i < len(c_bsnaps)
                else copy.deepcopy(beliefs))
        history.record_substep(k_base + step_i + 1, 'C',
                               rover_pos, cpos, snap)


def _record_rover_substeps(history, substeps, copter_pos, k_base,
                           r_bsnaps, beliefs):
    """record step-level rover history."""
    for step_i, (spos, sq) in enumerate(substeps[1:]):
        # step 8b store the rover-side belief trail for animation and audit
        snap = (r_bsnaps[step_i] if step_i < len(r_bsnaps)
                else copy.deepcopy(beliefs))
        history.record_substep(k_base + step_i + 1, 'R',
                               spos, copter_pos, snap)


def _maybe_snapshot(history, beliefs, k, rover_pos, copter_pos,
                    rover_q, max_k, snap_targets):
    """save sparse snapshots during the run."""
    n_snaps = len(history.snap_meta)
    next_snap_k = history.snap_meta[0]['k'] + (max_k // snap_targets) * n_snaps
    # step 10a save snapshots on mission finish, failure, or sparse time anchors
    if (rover_q in FSA_ACCEPT or rover_q == FSA_DEAD or k >= next_snap_k):
        if n_snaps < snap_targets + 1:
            history.snapshot(beliefs, k, rover_pos, copter_pos)

def run_simulation(
    rover_start  = (0, 0), copter_start = (0, 0),
    T_c      = 5,   T_r      = 3, alpha    = 1.5,
    gamma    = 0.0, lambda_  = 0.0,
    tau_r    = 1.0,
    *,
    vi_steps = None,
    max_k    = 600,
    seed     = 42,
    copter_mode = 'global',
    copter_energy = float('inf'),
) -> dict:
    """run one full rover-copter simulation."""

    # step 0 set seed and choose the copter solver
    copter_explore = _COPTER_MODES[copter_mode]
    np.random.seed(seed)
    random.seed(seed)
    is_baseline = (gamma == 0.0 and copter_mode in ('local', 'global'))

    # step 1 initialize state, beliefs, and energy
    k = 0
    copter_pos, rover_pos, rover_q = copter_start, rover_start, 0
    complete = False
    beliefs = init_beliefs()
    E_c = copter_energy
    prev_copter_substeps = None
    SNAP_TARGETS = 3
    
    history = SimHistory(
        rover_start, copter_start,
        T_c=T_c, T_r=T_r, alpha=alpha,
        gamma=gamma, lambda_=lambda_,
        tau_r=tau_r, vi_steps=vi_steps,
        max_k=max_k, seed=seed,
        copter_mode=copter_mode,
        copter_energy=copter_energy,
        planner_name=None,
    )
    history.enable_belief_history()

    # step 2 do the initial copter sense
    # formula: b_0 -> b_0^+ bootstrap
    # the shared map starts from the prior and immediately absorbs local o data
    copter_sense(beliefs, copter_pos)
    history.snapshot(beliefs, 0, rover_pos, copter_pos)
    history.record_substep(0, 'I', rover_pos, copter_pos, beliefs)

    # step 3 build the graph and initialize the rover planner
    # formula: planner split
    # vi is the altruistic baseline path; d* lite is the risk-aware rover path
    spatial = grid_to_graph()
    if is_baseline:
        planner = VIPlanner(spatial, beliefs, vi_steps=vi_steps or 80)
        planner_name = 'VI'
    else:
        planner = DStarLitePlanner(spatial, beliefs, gamma=gamma, tau_r=tau_r)
        planner_name = 'D* Lite'
    history['planner_name'] = planner_name
    V, policy, n_expanded = planner.solution()
    print(f"  Initial {planner_name}: {n_expanded} expanded.")

    b_max = (compute_b_max(spatial, beliefs, policy, rover_pos, rover_q, T_r)
             if copter_mode != 'utility' else {})

    epoch_idx = 0
    while k < max_k and not complete:

        # step 4 snapshot b_t before the copter moves
        k_before_c = k
        beliefs_backup = copy.deepcopy(beliefs)
        copter_pos_backup = copter_pos
        rover_pos_at_epoch_start = rover_pos

        # step 5 let the copter explore and update beliefs to b_t+
        # formula: scout-first belief lift
        # the copter acts on b_t and writes the mission belief snapshot b_t^+
        copter_pos, c_substeps, c_bsnaps = _run_copter(
            copter_explore, copter_mode, beliefs, copter_pos, b_max,
            T_c, alpha,
            lambda_=lambda_, rover_pos=rover_pos, energy=E_c)
        k += T_c

        # step 6 apply the copter gates
        (c_substeps, copter_pos, beliefs, c_bsnaps,
         E_c, action_changed) = _gate_copter_action(
            c_substeps, copter_pos, beliefs,
            beliefs_backup, copter_pos_backup,
            prev_copter_substeps, c_bsnaps,
            V, rover_pos, rover_q,
            alpha, gamma, lambda_, tau_r,
            E_c, T_c,
        )
        prev_copter_substeps = list(c_substeps)

        _record_copter_substeps(history, c_substeps, rover_pos, k_before_c,
                                c_bsnaps, beliefs)

        # step 7 replan the rover on b_t+ with frozen risk beliefs
        # formula: split-belief rover utility
        # mission term g uses b_t^+ while c_r uses the frozen pre-copter b_t
        V, policy, n_expanded = planner.replan(
            beliefs, risk_beliefs=beliefs_backup
        )
        b_max = (compute_b_max(spatial, beliefs, policy, rover_pos, rover_q, T_r)
                 if copter_mode != 'utility' else {})

        # step 8 execute the rover policy for t_r steps
        # formula: committed best response
        # the rover follows the current policy without replanning mid-epoch
        k_before_r = k
        rover_pos, rover_q, substeps, r_bsnaps = rover_execute(
            beliefs, rover_pos, rover_q, policy, T_r,
            record_beliefs=True)
        k += T_r

        _record_rover_substeps(history, substeps, copter_pos, k_before_r,
                               r_bsnaps, beliefs)
        # step 9 compute logged epoch metrics
        metrics = _compute_epoch_metrics(
            substeps, c_substeps, beliefs_backup,
            V, rover_pos, rover_q, rover_pos_at_epoch_start,
            alpha, gamma, lambda_)
        history.record_epoch(EpochRecord(
            epoch=epoch_idx,
            k_start=k_before_c,
            k_end=k,
            rover_pos=rover_pos,
            copter_pos=copter_pos,
            rover_q=rover_q,
            c_substeps=list(c_substeps),
            action_changed=action_changed,
            r_substeps=list(substeps),
            Phi=metrics['Phi'],
            G=metrics['G'],
            C_r=metrics['C_r'],
            C_c=metrics['C_c'],
            I=metrics['I'],
            u_c=metrics['u_c'],
            E_c=E_c,
            n_expanded=n_expanded,
        ))
        epoch_idx += 1

        _maybe_snapshot(history, beliefs, k, rover_pos, copter_pos,
                        rover_q, max_k, SNAP_TARGETS)

        # step 10 check termination
        # step 10b this is runtime reporting only; it does not affect utilities
        complete = rover_q in FSA_ACCEPT
        if complete:
            which = {3: 'φ1 (found A)', 4: 'φ2 (B→C)',
                     5: 'φ3 (C→D)'}.get(rover_q, '?')
            print(f"   MISSION COMPLETE at k={k}, branch={which}, "
                  f"pos={rover_pos}  [{planner_name}={n_expanded} expanded]")
        elif rover_q == FSA_DEAD:
            print(f"   MISSION FAILED (obstacle) at k={k}, pos={rover_pos}")
            break
        elif k % 1 == 0 and verbose:
            V_here = V.get((rover_pos[0], rover_pos[1], rover_q), 0.0)
            print(f"  k={k:4d}  rover={rover_pos}  q={rover_q}  "
                  f"V={V_here:.3f}  copter={copter_pos}  "
                  f"{planner_name}={n_expanded} expanded")

    # step 11 finalize the run history
    # step 11a finalize the run with the last belief map and terminal status
    fail = rover_q == FSA_DEAD
    if not complete and not fail:
        print(f"  Time limit reached at k={max_k}")

    if history.snap_meta[-1]['k'] < k:
        history.snapshot(beliefs, k, rover_pos, copter_pos)

    if not history.substeps or history.substeps[-1]['k'] != k:
        history.record_substep(k, 'E', rover_pos, copter_pos, beliefs)

    which_phi = {3: 'φ1 (found A)', 4: 'φ2 (B→C)', 5: 'φ3 (C→D)'}.get(rover_q)
    history.finalise(complete, fail, which_phi, k, rover_pos, copter_pos, rover_q)
    return history
