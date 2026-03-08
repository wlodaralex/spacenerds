"""
fsa.py - Finite state automaton (FSA) for φ = φ1 V φ2 V φ3.

Setup:
    - FSA (Section 2.2 & 4.3.1):
        * Implements A_φ that accepts all good prefixes of the scLTL mision formula.
    - Product belief MDP (Section 4.3.1):
        * Belief-weighted transition function B_en (Eqs. 14-15).

Mission Formula (Section 6.1.1):
    * φ = φ1 V φ2 V φ3
        * φ1 = F_o A             reach A while always avoiding O
        * φ2 = F_o B ^ ⃝ F_o C   reach B then C while avoiding O throughout
        * φ3 = F_o C ^ ⃝ F_o D   reach C then D while avoiding O throughout
    * where F_o ap = ¬O U (¬O ^ ap)    (reach ap while never touching O)

FSA States:
    * 0  : initial
    * 1  : found B        (intermediate for φ2)
    * 2  : found C        (intermediate for φ3)
    * 3  : accept (φ1)    (reached A)
    * 4  : accept (φ2)    (reached C after B)
    * 5  : accept (φ3)    (reached D after C)
    * 6  : found C and found B, looking for D or C
    * -1 : dead           (obstacle encountered)
"""

from environment import AP_LIST



# FSA Transition #
FSA_ACCEPT = {3, 4, 5}
FSA_DEAD = -1
FSA_ALL = [FSA_DEAD, 0, 1, 2, 3, 4, 5]


def fsa_step(q: int, labels: frozenset) -> int:
    """
    Advance FSA state q given the true labels at the rover's current cell.

    Transition priority (applied in order):
        1. Dead / accept states are absorbing (i.e., once entered, it remains forever).
        2. Any obstacle label results in transition to dead state.
        3. Target labels drive branch transitions (where A precedes B/C because φ1 is 
           always open in parallel with φ2/φ3).
    """
    if q == FSA_DEAD or q in FSA_ACCEPT:
        return q
    if 'O' in labels:
        return FSA_DEAD
    if q == 0:
        if 'A' in labels: return 3  # φ1 complete
        if 'B' in labels: return 1  # start φ2 branch
        if 'C' in labels: return 2  # start φ3 branch
        return 0
    if q == 1:                      # found B and looking for C
        if 'C' in labels: return 4  # φ2 complete
        if 'A' in labels: return 3  # φ1 complete (shortcut)
        return 1
    if q == 2:                      # found C and looking for D
        if 'D' in labels: return 5  # φ3 complete
        if 'A' in labels: return 3  # φ1 complete (shortcut)
        return 2
    return q


ALL_SUBSETS = [ # construct all possible subsets of atomic propositions
    frozenset(AP_LIST[i] for i in range(len(AP_LIST)) if mask & (1 << i))
    for mask in range(2 ** len(AP_LIST))
]


def compute_B_en(cell_beliefs, q):
    """
    Compute B_en(x |= en(q, q')) for all successor states q'.

    For each possible label set σ ⊆ AP, compute the joint belief that exactly σ is satisfied at cell x.
    Then accumulate by the FSA successor q' = δ(q, σ).

    Note: 
        - This is called once per (cell, q) pair in value_iteration and cached, since beliefs are fixed within one VI call.
    
    Return
        * {q_next: probability} = 1.0
        - Represents the belief distribution over FSA transitions from state q at cell x.  
    """
    result = {}
    for sigma in ALL_SUBSETS:
        # Joint belief that, in this cell, all APs in σ are true and all other APs are false.
        b = 1.0
        for ap in AP_LIST:
            b *= cell_beliefs[ap] if ap in sigma else (1.0 - cell_beliefs[ap])
        q_next = fsa_step(q, sigma)
        result[q_next] = result.get(q_next, 0.0) + b
    return result