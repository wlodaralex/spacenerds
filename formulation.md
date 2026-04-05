# Problem Formulation

## 1. Player and State

### 1.1 Players

Let $r$ denote the rover and $c$ the copter.

$$\mathcal{N} = \{r, c\}$$

We model a two-player game with aligned but distinct objectives:
- both agents share the mission-success term $G$; the copter additionally values
exploration through $I$
- each agent bears its own private operational cost,
$C_r$ for rover risk and $C_c$ for copter travel

### 1.2 Sensitivity Parameters $\gamma, \lambda \in \mathbb{R}_{\geq 0}$

Marginal rates of substitution between mission value and private operational
cost:

$$\gamma = \left.\frac{dG}{dC_r}\right|_{du_r=0} \qquad \lambda = \left.\frac{dG}{dC_c}\right|_{du_c=0}$$

| | $\gamma$ (rover) | $\lambda$ (copter) |
|---|---|---|
| **Meaning** | Sensitivity to traversal risk | Sensitivity to copter travel cost |
| **Large** | Conservative, safe paths | Budget-saving, short flights |
| **Small** | Aggressive, prioritises mission | Explores freely |

### 1.3 Public State

$$s_t = (x_t^r, \; x_t^c, \; \mathcal{B}_t, \; q_t)$$

| Symbol | Meaning | Update |
|--------|---------|--------|
| $x_t^r \in X$ | Rover position | Deterministic motion |
| $x_t^c \in X$ | Copter position | Deterministic motion |
| $\mathcal{B}_t$ | Shared belief map: $\mathcal{B}_t(x \models ap) \in [0,1]$ | Bayesian sensing |
| $q_t \in Q$ | FSA state tracking scLTL progress | $q' = \delta(q, \sigma(x))$ |

## 2. Action Space

- **Rover:** edges into cells whose obstacle belief exceeds $\tau_r$
  are pruned. The rover policy is defined only over successor edges
  whose destination cell satisfies this threshold.
  $$a_r \in \mathcal{A}_r(s_t) = \{a_r : \mathcal{B}_t(x' \models O) \leq \tau_r \;\;\forall (x, x') \in a_r\}$$

- **Copter:** total flight cost must fit within the remaining budget
  $D_t^c$, which depletes across epochs. $x_{\text{ref}} = x_t^r$ is
  the rover's position frozen at epoch start.

  $$a_c \in \mathcal{A}_c(s_t) = \left\{a_c : \underbrace{\sum_k d(x_c^k, x_c^{k+1})}_{\text{scout}} + \;\rho \cdot \underbrace{d(x_c^{T_c}, x_{\text{ref}})}_{\text{return to rover}} \leq D_t^c\right\}$$

## 3. Objective and Cost

### 3.1 Private Costs

Each agent pays a private operational cost for its actions.

- **Rover risk.** The negative log survival probability along the rover's path:

  $$C_r(a_r; \mathcal{B}_t) = -\sum_k \log(1 - \mathcal{B}_t(x_r^k \models O))$$

  The sum is over cells newly entered during the current epoch, so the
  epoch-start cell is not counted again.

  Each cell the rover enters contributes a penalty proportional to its
  obstacle belief. 
  - Safe cells (low $B_O$) are cheap
  - dangerous cells (high $B_O$) are expensive.

- **Copter travel.** The scout distance plus a return margin:

  $$C_c(a_c; x_{\text{ref}}) = \sum_k d(x_c^k, x_c^{k+1}) + \rho \cdot d(x_c^{T_c}, x_{\text{ref}})$$

  $x_{\text{ref}} = x_t^r$ is the rover's position frozen at epoch start.

### 3.2 Copter Information Gain

The copter is rewarded for visiting cells whose obstacle belief is uncertain.
$$I(a_c; \mathcal{B}_t) = \sum_{x \in \text{observed by } a_c} H(\mathcal{B}_t(x \models O))$$

$H$ is binary Shannon entropy: 
- cells near $B_O = 0.5$ are most valuable to observe, 
- cells near 0 or 1 are almost known. 

$I$ depends only on $a_c$, so it appears only in the copter's utility.

### 3.3 Shared Mission Term
The shared mission term is the probability that the rover's path reaches an accepting FSA state under the post-copter belief map.

$$V_\phi(a_r; \mathcal{B}_t^+, q_t) = \prod_{(x',q,q') \in a_r} B_{en}(x', q \to q')$$

Each edge contributes one transition-success probability $B_{en}$, so the full
product is the probability that the whole path succeeds.

$V_\phi$ depends on both agents: the rover chooses $a_r$, and the copter's
scouting shapes $\mathcal{B}_t^+$. We use the log form

$$G = \log V_\phi \in (-\infty, 0]$$

so the product becomes additive along the path.

### 3.4 Utilities

The rover balances mission progress against traversal risk.

$$u_r = G - \gamma \cdot C_r(a_r)$$

The copter balances mission progress and exploration against flight cost.

$$u_c = G + \alpha \cdot I(a_c) - \lambda \cdot C_c(a_c)$$

$\alpha$ is a public exploration reward weight; $\gamma$ and $\lambda$ are private cost sensitivities.

## 4. Epoch Structure

### 4.1 Timeline

This timeline describes the utility-mode rollout; the baseline
comparator follows a separate control flow.

```text
Epoch t
  |
  |- 1. Copter explores for T_c steps, sensing along the way.
  |      Beliefs update from B_t to B_t^+.
  |      Distance budget depletes by realised scout distance:
  |      D_{t+1}^c = D_t^c - \sum_k d(x_c^k, x_c^{k+1}).
  |
  |- 2. Rover replans on B_t^+, then executes for T_r steps.
  |      FSA state q updates from realised labels.
  |      Rover sensing contributes to B_{t+1}.
  |
Epoch t+1
```

### 4.2 What Each Agent Sees When It Plans

Not all terms use the same belief map. This is a deliberate design
choice.

| Belief | Used in | What it captures |
|--------|---------|------------------|
| $\mathcal{B}_t$ | $C_r$, $I$ | What was known before this epoch's scouting |
| $\mathcal{B}_t^+$ | $V_\phi$ | What is known after the copter explored |

The rover plans its mission path using the copter's fresh belief
$\mathcal{B}_t^+$, but evaluates traversal risk against the
pre-scouting map $\mathcal{B}_t$. This means the copter's action
changes the rover's mission outlook but cannot change the rover's risk
cost for the current epoch.

**Why this split.** The rover trusts new scouting data for route
planning but does not drop survival caution based on a single epoch of
aerial sensing

### 4.3 Which Terms Appear in Which Utility

If a term depends on both agents' actions, it appears in both utilities. If it
depends on only one agent's action, it appears only in that agent's utility.

| Term | Depends on | In $u_r$? | In $u_c$? |
|------|-----------|-----------|-----------|
| $G = \log V_\phi$ | both agents | ✓ | ✓ |
| $\alpha I$ | copter only |  | ✓ |
| $\gamma C_r$ | rover only | ✓ |  |
| $\lambda C_c$ | copter only |  | ✓ |


This is what makes the three-line proof in §5 possible: when one agent
deviates, the other agent's private terms do not change.

## 5. Game Analysis

### 5.1 Exact Potential Game

**Proposition.** The per-epoch rover-copter game with finite action
spaces is an exact potential game with

$$\Phi = G + \alpha I - \gamma C_r - \lambda C_c$$

where $G = \log V_\phi$. 

**What this gives.** A pure-strategy Nash equilibrium exists for every
epoch game (the global maximiser of $\Phi$ over finite action sets).
Unilateral improvements by either agent are aligned with $\Phi$:
selfish improvements cannot cancel each other out.

**Proof.**

> **Definition.** A game is an *exact potential game* if there exists a 
single function $\Phi$ over all joint action profiles such that, 
whenever one agent unilaterally changes its action,
>   $$\Delta u_i = \Delta \Phi.$$
> That is, the change in that agent's utility exactly equals the change
in $\Phi$. Since $\Phi$ is bounded above on a finite action space, it
has a global maximiser, and that maximiser is a pure-strategy Nash
equilibrium.

*Rover deviates* (fix $a_c$):

$$\Delta u_r = \Delta G - \gamma \Delta C_r$$

$$\Delta \Phi = \Delta G + \alpha \underbrace{\Delta I}_{=0} - \gamma \Delta C_r - \lambda \underbrace{\Delta C_c}_{=0} = \Delta G - \gamma \Delta C_r$$

$$\Delta u_r = \Delta \Phi$$

*Copter deviates* (fix $a_r$):

$$\Delta u_c = \Delta G + \alpha \Delta I - \lambda \Delta C_c$$

$$\Delta \Phi = \Delta G + \alpha \Delta I - \gamma \underbrace{\Delta C_r}_{=0} - \lambda \Delta C_c = \Delta G + \alpha \Delta I - \lambda \Delta C_c$$

$$\Delta u_c = \Delta \Phi \quad$$

In both cases $\Delta u_i = \Delta \Phi$, satisfying the definition.
Therefore the per-epoch rover-copter game is an exact potential game,
and a pure-strategy Nash equilibrium exists for every epoch.

### 5.2 Design Insight

The proof is three lines of algebra. The contribution is the
cost-function design that makes those three lines work:

| Design choice | What it ensures | Without it |
|--------------|----------------|------------|
| Same $G$ in both utilities | $\Delta G$ cancels in both directions | One side has an unmatched $\Delta G$ |
| $C_r$ uses $\mathcal{B}_t$ not $\mathcal{B}_t^+$ | Copter's action cannot affect $C_r$ | Copter deviating changes $C_r$; proof breaks |
| $x_{\text{ref}}$ frozen at epoch start | Rover's action cannot affect $C_c$ | Rover deviating changes $C_c$; proof breaks |
| $\alpha I$ in copter only | Legal because $I$ depends only on $a_c$ | Would also work if shared, but less honest |

### 5.3 Corollaries

- **Altruistic limit.** In the epoch-game definition, setting
$\gamma = \lambda = 0$ gives
$u_r = G$ and $u_c = G + \alpha I$. The rover optimises pure mission
value; the copter optimises mission value plus exploration. 

- **Decentralisation.** Neither agent needs to know the other's
sensitivity, predict the other's action, or know $\Phi$ exists. Each
selfishly optimises its own $u_i$. The potential structure ensures
these selfish improvements are globally aligned.

- **Cross-epoch progress.** The exact-potential result is per-epoch:
each epoch defines a different game $\mathcal{G}_t$ because beliefs,
budget, and FSA state change across epochs. Cross-epoch mission
progress is driven by repeated sensing, which improves the belief
map over visited regions.


## 6. Implementation

Solver choices are implementation decisions, 
separate from the game definition.

### 6.1 Rover: D* Lite on $(X \times Q)$

Each product-graph edge has cost

$$c(e) = \underbrace{-\log B_{en}(x', q, q')}_{\text{mission uncertainty}} + \;\gamma \cdot \underbrace{\bigl(-\log(1-\mathcal{B}_t(x' \models O))\bigr)}_{\text{obstacle risk}}$$

Summing over a path gives $-u_r$, so the cheapest path maximises
rover utility. D* Lite replans incrementally when beliefs change
across epochs, repairing only affected edges.

**Heuristic.** The default heuristic is

$$h(x,q) = d_{\text{FSA}}(q, Q_f) \cdot c_{\min}^+(t)$$

where $d_{\text{FSA}}$ is the minimum FSA transitions to acceptance and
$c_{\min}^+$ is the cheapest positive edge cost. This is admissible:
the rover needs at least $d_{\text{FSA}}$ edges, each costing at least
$c_{\min}^+$. Admissibility requires (1) no zero-cost edges and (2) no
FSA $\varepsilon$-transitions. Condition (2) holds by construction.
Condition (1) is not globally guaranteed because $B_{en} = 1$ can
produce zero-cost mission edges.

### 6.2 Copter: Exhaustive Finite-Horizon Search
```text
1. Enumerate all 9^{T_c} action sequences.
2. Score each by the surrogate alpha * I - lambda * C_c.
3. Keep the top-K feasible surrogate sequences.
4. For each shortlisted sequence, sample possible copter observations,
   form Monte Carlo posteriors, and rerun a mission-only rover planner
   to estimate G.
5. Select the best reranked sequence and execute it, yielding B_t^+.
```

The runtime copter does not evaluate the exact shared term for every
candidate sequence. Instead, it uses $\alpha I - \lambda C_c$ as a
vectorised shortlist score and then estimates $G$ only on the top-K
candidates via Monte Carlo mission-only rover replans. The copter
therefore computes an approximate best response to the epoch game
rather than an exact one.

If no sequence is feasible, the copter hovers and
$\mathcal{B}_t^+ = \mathcal{B}_t$.

## 7. Reference

### 7.1 Variable Table

| Variable | Role | Public/Private | Static/Dynamic |
|----------|------|---------------|----------------|
| $\tau_r$ | Obstacle threshold | Public | Static |
| $D_t^c$ | Remaining distance budget | Copter internal | Dynamic |
| $\gamma$ | Risk sensitivity (MRS) | Private | Static |
| $\lambda$ | Distance-budget sensitivity (MRS) | Private | Static |
| $\alpha$ | Exploration reward weight | Public (system design) | Static |
| $\rho$ | Return-distance ratio | Public | Static |

### 7.2 Acknowledged Modelling Choices

| Choice | Why | Consequence |
|--------|-----|-------------|
| $\mathcal{B}_t$ vs $\mathcal{B}_t^+$ split in $u_r$ | Separability | Optimistic mission + pessimistic survival |
| $x_{\text{ref}}$ frozen | Prevents $C_c \to a_r$ dependency | Return cost from current, not future rover position |
| $I$ as entropy surrogate | Tractable | Not identical to true VoI |
| Runtime copter estimates $G$ via top-K Monte Carlo mission-only replans | Tractability | Copter computes an approximate, not exact, best response; `copter_top_k` and `copter_mc_samples` tune the speed/accuracy tradeoff |
| Logged $G$ when $\gamma > 0$ | Metric reporting | Logged $\log(V)$ includes future risk, so it is not a pure mission-only $G$ |
| Default rover heuristic uses $d_{\text{FSA}}(q, Q_f)\,c_{\min}^+(t)$ | Matches the implementation to the stated admissible lower bound | Admissible, but weak for the current shallow FSA; positive edge costs and no $\varepsilon$-transitions are required |

### 7.3 Summary

We model the rover-copter system as a two-player game with shared mission term
$G = \log V_\phi$, copter exploration reward $\alpha I$, and private costs
$C_r$, $C_c$.

The game is an exact potential game with

$$\Phi = G + \alpha I - \gamma C_r - \lambda C_c.$$

The key design choices are the $\mathcal{B}_t / \mathcal{B}_t^+$ split, the
frozen $x_{\text{ref}}$, and the role-asymmetric information term, which make
the proof go through. In the checked-in runtime, the copter estimates $G$
approximately through top-K Monte Carlo mission-only replans, and the rover
uses the $d_{\text{FSA}}(q, Q_f)\,c_{\min}^+(t)$ heuristic under the same
positive-edge-cost conditions stated above.
