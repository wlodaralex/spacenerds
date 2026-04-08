"""
sensor.py

- define the sensor accuracy model and bayes update rule
- initialize the shared prior belief map

function
- `sensor_beta`: compute distance-based sensor accuracy
- `bayes_update`: update one bernoulli belief
- `observe`: sample one noisy observation
- `init_beliefs`: build the initial belief map
"""

import numpy as np
from environment import GRID


# --- Sensor Model ---
TARGET_PRIOR = 0.5
OBSTACLE_PRIOR = 0.5
BAYES_DENOMINATOR_EPS = 1e-12


def sensor_beta(agent_pos: tuple, cell_pos: tuple, R: float, M: float) -> float:
    """
    Compute β(agent, cell) for the probability of a correct binary observation.

    β = {
        M/R^4 * (d^2 - R^2)^2 + 0.5, if d ≤ R   (Eq. 3)
        0.5,                         if d > R   (Eq. 4)
    }

    Function Parameters
        * agent_pos : (row, col) of the sensing agent.
        * cell_pos  : (row, col) of the cell being observed.
        * R         : sensor range with β = 0.5 (random guess) beyond this.
        * M         : accuracy margin in range (0, 0.5], with distance of 0 the maximum becomes '0.5 + M'.

    Sensor Parameters (Section 6.1.1)
        - Rover:
            * R = 2
            * M = 0.5
        - Copter:
            * R = 4
            * M = 0.4
    """
    d = np.hypot(agent_pos[0] - cell_pos[0], agent_pos[1] - cell_pos[1])
    if d > R:
        return 0.5
    return min(M / (R ** 4) * (d ** 2 - R ** 2) ** 2 + 0.5, 0.5 + M)


def bayes_update(belief: float, z: int, beta: float) -> float:
    """
    Update B(x |= ap) given observation z ∈ {0, 1} and sensor accuracy β.

    Derivation from Bayes' theorem with Bernoulli likelihood (Eq. 8):
        - If z=1: posterior ∝ β * prior
        - If z=0: posterior ∝ (1-β) * prior

    The denominator is floored to keep the update stable if a belief becomes
    numerically saturated near 0 or 1.
    """
    if z == 1:
        # z=1: Pr[Z=1 | ap true] = β^1 * (1-β)^0 = β
        num = beta * belief
        # Pr[Z=1] = β*B + (1-β)*(1-B)
        den = beta * belief + (1 - beta) * (1 - belief)
    else:
        # z=0: Pr[Z=0 | ap true] = β^0 * (1-β)^1 = (1-β)
        num = (1 - beta) * belief
        # Pr[Z=0] = (1-β)*B + β*(1-B)
        den = (1 - beta) * belief + beta * (1 - belief)
    return num / den if den > BAYES_DENOMINATOR_EPS else belief


def observe(true_labels: frozenset, ap: str, beta: float) -> int:
    """
    Simulate one Bernoulli sensor reading for atomic proposition, ap.

    Return
        * 1 (ap detected)
        * 0 (ap not detected)
        - With probability 'β' the reading is correct or '1-β' it is flipped.
    """
    truth   = (ap in true_labels)
    correct = (np.random.rand() < beta)
    return int(truth if correct else not truth)


# --- Belief State ---
def init_beliefs() -> dict:
    """
    Initialize the shared environmental belief map.

    The paper's Algorithm 1 suggests a uniform ``0.5`` prior. The current
    implementation instead uses map-density priors:
        - targets (A, B, C, D): 0.1
        - obstacles (O): 0.3
    """
    base_cell_belief = {
        'A': TARGET_PRIOR,
        'B': TARGET_PRIOR,
        'C': TARGET_PRIOR,
        'D': TARGET_PRIOR,
        'O': OBSTACLE_PRIOR,
    }
    beliefs = {}
    for r in range(GRID):
        for c in range(GRID):
            beliefs[(r, c)] = dict(base_cell_belief)
    return beliefs
