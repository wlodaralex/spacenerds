"""finite-state mission logic and belief-weighted automaton transitions."""

from environment import AP_LIST

FSA_ACCEPT = {3, 4, 5}
FSA_DEAD = -1
FSA_ALL = [FSA_DEAD, 0, 1, 2, 3, 4, 5]


def fsa_step(q: int, labels: frozenset) -> int:
    """advance one automaton state using one realized label set."""
    # formula: obstacle veto
    # o has highest priority and sends every active state to dead
    if q == FSA_DEAD or q in FSA_ACCEPT:
        return q
    if 'O' in labels:
        return FSA_DEAD
    if q == 0:
        # formula: branch ladder
        # a wins immediately, then b starts phi_2, then c starts phi_3
        if 'A' in labels: return 3  # φ1 complete
        if 'B' in labels: return 1  # start φ2 branch
        if 'C' in labels: return 2  # start φ3 branch
        return 0
    if q == 1:                      # found B and looking for C
        # formula: shortcut ladder
        # c finishes phi_2; a is still an immediate shortcut to acceptance
        if 'C' in labels: return 4  # φ2 complete
        if 'A' in labels: return 3  # φ1 complete (shortcut)
        return 1
    if q == 2:                      # found C and looking for D
        # formula: delayed-finish ladder
        if 'D' in labels: return 5  # φ3 complete
        if 'A' in labels: return 3  # φ1 complete (shortcut)
        return 2
    return q

ALL_SUBSETS = [
    frozenset(AP_LIST[i] for i in range(len(AP_LIST)) if mask & (1 << i))
    for mask in range(2 ** len(AP_LIST))
]


def compute_B_en(cell_beliefs, q):
    """return the distribution over automaton successors at one cell.

    mathematically this is

        B_en(x', q -> q') =
            sum_sigma 1[delta(q, sigma) = q']
            prod_ap B_ap(x')^{1[ap in sigma]} (1-B_ap(x'))^{1[ap not in sigma]}

    the old implementation evaluated that sum by enumerating all 2^|AP| label
    subsets. the current implementation is algebraically equivalent but uses the
    fsa priority rules to collapse the subset sum into closed-form bernoulli
    products. for example, at q=0 the only relevant events are:

        q'=-1: O
        q'=3 : (not O) and A
        q'=1 : (not O) and (not A) and B
        q'=2 : (not O) and (not A) and (not B) and C
        q'=0 : (not O) and (not A) and (not B) and (not C)

    so the returned probabilities are exact, but the exponential subset loop is
    removed.
    """
    if q == FSA_DEAD or q in FSA_ACCEPT:
        return {q: 1.0}

    # step 1 unpack the independent bernoulli cell beliefs
    obstacle_probability = cell_beliefs['O']
    non_obstacle_probability = 1.0 - obstacle_probability
    a_probability = cell_beliefs['A']
    b_probability = cell_beliefs['B']
    c_probability = cell_beliefs['C']
    d_probability = cell_beliefs['D']

    if q == 0:
        # step 2 collapse the subset sum with the q=0 priority chain
        successor_probability = {
            FSA_DEAD: obstacle_probability,
            3: non_obstacle_probability * a_probability,
            1: non_obstacle_probability * (1.0 - a_probability) * b_probability,
            2: (
                non_obstacle_probability
                * (1.0 - a_probability)
                * (1.0 - b_probability)
                * c_probability
            ),
            0: (
                non_obstacle_probability
                * (1.0 - a_probability)
                * (1.0 - b_probability)
                * (1.0 - c_probability)
            ),
        }
    elif q == 1:
        # step 2 collapse the subset sum with the q=1 shortcut chain
        successor_probability = {
            FSA_DEAD: obstacle_probability,
            4: non_obstacle_probability * c_probability,
            3: non_obstacle_probability * (1.0 - c_probability) * a_probability,
            1: non_obstacle_probability * (1.0 - c_probability) * (1.0 - a_probability),
        }
    else:
        # step 2 collapse the subset sum with the q=2 delayed-finish chain
        successor_probability = {
            FSA_DEAD: obstacle_probability,
            5: non_obstacle_probability * d_probability,
            3: non_obstacle_probability * (1.0 - d_probability) * a_probability,
            2: non_obstacle_probability * (1.0 - d_probability) * (1.0 - a_probability),
        }

    # step 3 drop zero-mass successors to keep later planner loops sparse
    return {
        q_next: probability
        for q_next, probability in successor_probability.items()
        if probability > 0.0
    }
