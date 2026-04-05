"""
environment.py

- define the grid, labels, and action set
- provide shared motion helpers used by planning and simulation

function
- `clip`: clamp one cell to the grid
- `transition_probabilities`: build one action transition kernel
"""

# --- Grid And Labels ---

# grid size
GRID = 10

# atomic propositions
AP_LIST = ['A', 'B', 'C', 'D', 'O']

# ground-truth labels from the map
TRUE_LABELS_RAW = {
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

# map every cell to a frozenset of true atomic propositions
TRUE_L = {}
for r in range(GRID):
    for c in range(GRID):
        ap = TRUE_LABELS_RAW.get((r, c))
        TRUE_L[(r, c)] = frozenset([ap]) if ap else frozenset()

# --- Motion Model ---

# action set: stay plus 8-connected motion
ACTIONS = [
    (0, 0),
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]

# cardinal offsets remain available for the generic transition helper
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# --- Motion Helpers ---
def clip(r: int, c: int) -> tuple[int, int]:
    """restrict one coordinate pair to valid grid indices."""
    return max(0, min(GRID - 1, r)), max(0, min(GRID - 1, c))


def transition_probabilities(r: int, c: int, action_index: int,
                             p_intended: float) -> dict[tuple[int, int], float]:
    """
    Return the transition distribution for one action.

    Structure:
        - Pick the intended destination from the selected action.
        - Clamp it to the grid.
        - If ``p_intended < 1``, split the residual mass uniformly across
          distinct cardinal neighbours of the intended destination.

    Notes:
        - The active codepaths use ``p_intended = 1.0``.
        - Utility-mode copter motion is deterministic and does not call this.
    """
    dr, dc = ACTIONS[action_index]
    r2, c2 = clip(r + dr, c + dc)
    
    slips = list(set(
        clip(r2 + ddr, c2 + ddc) 
        for ddr, ddc in CARDINALS
        if clip(r2 + ddr, c2 + ddc) != (r2, c2)
    ))
    if not slips:
        slips = [(r2, c2)]

    slip_each = (1.0 - p_intended) / len(slips)
    probabilities = {(r2, c2): p_intended}
    for slip in slips:
        probabilities[slip] = probabilities.get(slip, 0.0) + slip_each
    return probabilities
