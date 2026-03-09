"""
environment.py - Grid utilities and MDP motion model.

Setup:
    - 10x10 grid from Fig. 2.
    - Set of atmoic propositions (Section 3.1):
        * Obstacles are not hard-coded walls for collision checking, instead avoidance is handled by the FSA and belief-weighted value iteration.
    - Stochastic transition model for both agents (Section 3.2.1 & 6.1.1):
        * Motion with 'p_intended' probability of reaching the desired cell and '1 - p_intended' slip probability split among cardinal neighbours of the target cell.
"""

# Grid (10x10)
GRID = 10
# Atomic propositions
AP_LIST = ['A', 'B', 'C', 'D', 'O'] # Target objects A-D and obstacle O
NUM_AP = len(AP_LIST)

# Ground-truth labelling from Fig. 2 (row=0: top, col=0: left)
TRUE_LABELS_RAW = { # Each cell maps to at most one AP (empty string means no label)
    (0, 3): 'O', (0, 4): 'D',
    (1, 0): 'O',
    (2, 0): 'D', (2, 5): 'O', (2, 6): 'O', (2, 7): 'O', (2, 8): 'O', (2, 9): 'O',
    (3, 1): 'O', (3, 7): 'O', (3, 8): 'C',
    (4, 2): 'O',
    (5, 0): 'C',
    (6, 0): 'O', (6, 1): 'O', (6, 4): 'O', (6, 6): 'O', (6, 7): 'O',
    (7, 0): 'A', (7, 3): 'O', (7, 4): 'C', (7, 6): 'D', (7, 7): 'O',
    (8, 7): 'O',
    (9, 0): 'B', (9, 1): 'O', (9, 9): 'B',
}

# Uncertain environment
TRUE_L = {} # Map every cell (row, col) -> frozenset of APs true at that cell
for r in range(GRID):
    for c in range(GRID):
        ap = TRUE_LABELS_RAW.get((r, c), '')
        TRUE_L[(r, c)] = frozenset([ap]) if ap else frozenset() # frozenset() for empty cells so membership tests are O(1), i.e. fast



# MDP Motion Model #
# Action indices
ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] # stay, up, down, left, right
# Cardinal offsets used to compute slip destinations
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)] 


def clip(r: int, c: int) -> tuple[int, int]:
    """ Restrict (row, col) coordinates to only valid grid indices [0, GRID-1]. """
    return max(0, min(GRID - 1, r)), max(0, min(GRID - 1, c))


def transition_probabilities(r: int, c: int, a_idx: int, p_intended: float) -> dict[tuple[int, int], float]:
    """
    Stochastic transition distribution for action a_idx from (row, col) with 'p_intended' to desired cell,
    and with '1-p_intended' slip split uniformly among distinct cardinal neighbours of the desired cell.
        
    - Rover:  p_intended = 0.95.
    - Copter: p_intended = 0.90.
    """
    dr, dc = ACTIONS[a_idx]
    r2, c2 = clip(r + dr, c + dc) # intended destination (clamped)
    
    slips = list(set(
        clip(r2 + ddr, c2 + ddc) 
        for ddr, ddc in CARDINALS
        if clip(r2 + ddr, c2 + ddc) != (r2, c2) # exclude the intended cell itself
    ))
    if not slips: # handle edge cases (e.g., corners)
        slips = [(r2, c2)]

    slip_each = (1.0 - p_intended) / len(slips)
    probabilities = {(r2, c2): p_intended}
    for slip in slips:
        probabilities[slip] = probabilities.get(slip, 0.0) + slip_each # probability distribution of where the agent actually ends up
    
    return probabilities