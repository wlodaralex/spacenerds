"""
sensor.py - Bernoulli sensor model and Bayesian belief update.

Setup:
    - Bernoulli sensor (Section 3.2.2):
        * Observations are binary z ∈ {0,1} with accuracy β that decays with distance (Eqs. 3-4).
            * β:
                * Beyond range R it becomes '0.5'.
                * At a distance of 0 it becomes '0.5 + M'.
                - Motivates the agent to move directly over targets rather than sensing from range.
    - Bayesian belief (Section 4.2.1) performs recursive updates (Eq. 8) using the sensor model.
    - Binary observation simulation to generate sensor readings.
    - Belief initialization (Algorithm 1, line 2) sets priors for all cells/APs.

    - The paper suggests uniform informed 0.5 priors for maximum entropy.
        * Option 1: The 0.5 prior overstates initial uncertainty and slows early pathfinding.
            - Every unseen cell is equally likely to be an obstacle or a target, making early VI more conservative.
        * Option 2: Prior scaled to the approximate percentage of cells they occupy in the grid.
            * 0.1 for targets.
            * 0.3 for obstacles.
"""

import numpy as np
from environment import *



# Sensor Model #
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
    Updates B(x |= ap) given observation z ∈ {0, 1} and sensor accuracy β.

    Derivation from Bayes' theorem with Bernoulli likelihood (Eq. 8):
        - If z=1: posterior ∝ β * prior        (correct positive)
        - If z=0: posterior ∝ (1-β) * prior    (correct negative)
    
    As per 'algorithm 1', we initialize B ∈ (0,1). To ensure numerical stability the denominator is floored at 1e-12
    to prevent division by zero in case the belief ever reaches 0 or 1 due to prior initialization.
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
    return num / den if den > 1e-12 else belief


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



# Belief State #
def init_beliefs() -> dict:
    """
    Initialize the environmental belief dict. B[(r,c)][ap] ∈ (0,1).
        - Option 1: uniform prior of 0.5 (Algorithm 1, line 2).
        - Option 2: 
            * 0.1 for targets (A, B, C, D)
            * 0.3 for obstacles (O)
    Initialise beliefs. Paper Algorithm 1 line 2 says B ∈ (0,1); we use
    empirically-motivated priors that match actual cell densities:
      targets  ~3% of cells → prior 0.1
      obstacles ~20% of cells → prior 0.3
    Pure paper replication would use 0.5 everywhere; change here if needed.
    """
    B = {}
    for r in range(GRID):
        for c in range(GRID):
            B[(r, c)] = {'A': 0.1, 'B': 0.1, 'C': 0.1, 'D': 0.1, 'O': 0.3}
    return B