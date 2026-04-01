"""sensor model, bayes update, and belief initialization."""

import numpy as np
from environment import *



# Sensor Model #
def sensor_beta(agent_pos: tuple, cell_pos: tuple, R: float, M: float) -> float:
    """return the probability of a correct binary observation."""
    # formula: quartic sensor dome
    # beta(d) rises smoothly from 0.5 at the sensing boundary to 0.5 + m nearby
    d = np.hypot(agent_pos[0] - cell_pos[0], agent_pos[1] - cell_pos[1])
    if d > R:
        return 0.5
    return min(M / (R ** 4) * (d ** 2 - R ** 2) ** 2 + 0.5, 0.5 + M)


def bayes_update(belief: float, z: int, beta: float) -> float:
    """
    Update B(x |= ap) given observation z ∈ {0, 1} and sensor accuracy β.

    Derivation from Bayes' theorem with Bernoulli likelihood (Eq. 8):
        - If z=1: posterior ∝ β * prior        (correct positive)
        - If z=0: posterior ∝ (1-β) * prior    (correct negative)
    
    As per 'algorithm 1', we initialize B ∈ (0,1). To ensure numerical stability the denominator is floored at 1e-12
    to prevent division by zero in case the belief ever reaches 0 or 1 due to prior initialization.
    """
    if z == 1:
        num = beta * belief
        den = beta * belief + (1 - beta) * (1 - belief)
    else:
        num = (1 - beta) * belief
        den = (1 - beta) * belief + beta * (1 - belief)
    return num / den if den > 1e-12 else belief


def observe(true_labels: frozenset, ap: str, beta: float) -> int:
    """sample one noisy binary observation for one atomic proposition."""
    # formula: noisy truth coin
    truth   = (ap in true_labels)
    correct = (np.random.rand() < beta)
    return int(truth if correct else not truth)


def init_beliefs() -> dict:
    """build the initial shared belief map."""
    B = {}
    for r in range(GRID):
        for c in range(GRID):
            # step 1 seed the shared prior used by both agents at epoch 0
            B[(r, c)] = {'A': 0.1, 'B': 0.1, 'C': 0.1, 'D': 0.1, 'O': 0.3}
            # B[(r, c)] = {'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.5, 'O': 0.5}
    return B
