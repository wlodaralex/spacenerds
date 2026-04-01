# Problem Formulation 

## 1. Players

$$\mathcal{N} = \{r, c\}$$

Hashimoto treats the rover-copter pair as a centrally coordinated altruistic team. We model them as independent agents with aligned but distinct objectives: the rover navigates, the copter scouts. Both value mission success; only the copter is explicitly rewarded for information acquisition.

## 2. Public State

$$s_t = (x_t^r, \; x_t^c, \; \mathcal{B}_t, \; q_t)$$

| Symbol | Meaning | Update |
|--------|---------|--------|
| $x_t^r \in X$ | Rover position | Deterministic motion |
| $x_t^c \in X$ | Copter position | Deterministic motion |
| $\mathcal{B}_t$ | Shared belief map: $\mathcal{B}_t(x \models ap) \in [0,1]$ | Bayesian sensing |
| $q_t \in Q$ | FSA state tracking scLTL progress | $q' = \delta(q, \sigma(x))$ |

**Finiteness.** $X$ is a finite grid. $Q$ is a finite automaton state set. Action spaces are finite (bounded horizon on finite graph). All game-theoretic results (NE existence, FIP) depend on this finiteness.

Copter's remaining energy $E_t^c$ is tracked internally. Not part of public state.

## 3. Private Types (Sensitivity Parameters)

$$\gamma, \lambda \in \mathbb{R}_{\geq 0}$$

Marginal rates of substitution between mission value and private operational cost:

$$\gamma = \left.\frac{dG}{dC_r}\right|_{du_r=0} \qquad \lambda = \left.\frac{dG}{dC_c}\right|_{du_c=0}$$

| | γ (rover) | λ (copter) |
|---|---|---|
| **Meaning** | Sensitivity to traversal risk | Sensitivity to energy expenditure |
| **Large** | Conservative, safe paths | Energy-saving, short flights |
| **Small** | Aggressive, prioritizes mission | Explores freely |

**Static.** Physical dynamics handled by hard constraints ($E_t^c$, $\tau_r$), not by sensitivity weights.

**Private.** Each agent's sensitivity depends on local operational information not perfectly observable to its partner. Neither agent needs to know the other's — sequential structure ensures each responds to observed outcomes, not predicted actions.

**γ = λ = 0 → altruistic limit** (see §11 Corollary 3).

## 4. Hard Constraints (Physics — Inviolable)

$$\mathcal{A}_r(s_t) = \{a_r : \mathcal{B}_t(x \models O) \leq \tau_r \;\;\forall x \in a_r\}$$

$$\mathcal{A}_c(s_t) = \{a_c : C_c^{\text{dist}}(a_c) + E_{\text{return}}(a_c) \leq E_t^c\}$$

Not preferences. Physics. Cannot be traded off against mission value.

## 5. Actions

$$a_r \in \mathcal{A}_r: \text{rover's feasible trajectory in } X \times Q, \; T_r \text{ steps}$$

$$a_c \in \mathcal{A}_c: \text{copter's feasible scouting trajectory in } X, \; T_c \text{ steps}$$

Both finite sets (finite grid, bounded horizon). Deterministic motion.

Specific solvers (D* Lite for rover, exhaustive utility search for copter) are **implementation choices** described in §13, not part of the game definition.

## 6. Epoch Timeline and Strategic Model

### Execution (sequential)

```
Epoch t: s_t known to both
  │
  ├─ Copter acts (T_c steps, sensing) → B_t^+
  │    E_{t+1}^c = E_t^c − flight cost (internal)
  │
  ├─ Rover observes B_t^+, acts (T_r steps, sensing) → B_{t+1}
  │    q updated by FSA transitions
  │    Rover sensing contributes to B_{t+1} (side effect, not planned objective)
  │
Epoch t+1
```

### Strategic model (simultaneous commitment)

We model the epoch-wise interaction as a **simultaneous-move game** in which both agents commit to action plans at the start of each epoch. The sequential execution can be interpreted as a particular realization of the simultaneous commitment — the copter's action produces $\mathcal{B}_t^+$, which the rover then uses. The "deviation" in the potential game proof is counterfactual: "had the copter chosen differently, $\mathcal{B}_t^+$ would differ, and with it $V_\phi$."

This is standard: alternating better-response dynamics implement simultaneous Nash equilibrium in potential games (Monderer & Shapley, 1996).

### Beliefs

| Belief | Used in | Why this one |
|--------|---------|-------------|
| $\mathcal{B}_t$ | $C_r$, $I$ | C_r must not depend on $a_c$ (separability) |
| $\mathcal{B}_t^+$ | $V_\phi$ | Rover uses copter's scouting result |

**Modeling note on $\mathcal{B}_t$ vs $\mathcal{B}_t^+$ in rover's utility.** The rover evaluates mission success optimistically (using $\mathcal{B}_t^+$ in $V_\phi$ — post-copter intelligence) but hedges conservatively for survival (using $\mathcal{B}_t$ in $C_r$ — pre-copter prior risk assessment). This split is a deliberate pessimistic-optimistic planning design that ensures action-separability, not merely an artifact.

## 7. Utility Structure

### Shared mission term

$$V_\phi(a_r; \mathcal{B}_t^+, q_t) = \prod_{\text{transitions along path}} B_{en}(x', q, q')$$

Path belief to FSA acceptance. Depends on BOTH $a_r$ (rover's path) and $a_c$ (copter's scouting affects $\mathcal{B}_t^+$). Included in both utilities to obtain the desired exact potential structure — this is a modeling design, not an unavoidable consequence of the domain.

**Log form.** Utilities use $G = \log V_\phi \in (-\infty, 0]$ — the additive log-probability form. The rover's edge cost $-\log B_{en} + \gamma(-\log(1-B_O))$ accumulates additively to $-G + \gamma C_r$, so minimizing path cost $=$ maximizing $G - \gamma C_r = u_r$. The implementation computes $V \cdot (1-B_O)^\gamma = \exp(G - \gamma C_r)$; since $\exp$ is strictly monotone over paths, $\operatorname{argmax}\exp(G - \gamma C_r) = \operatorname{argmax}(G - \gamma C_r)$, so the code is exact.

**Why the original $V_\phi - \gamma C_r$ form was wrong.** $V_\phi - \gamma C_r$ and $V_\phi \cdot e^{-\gamma C_r}$ are not ordinally equivalent: two paths with different $(V_\phi, C_r)$ pairs can be ranked differently by the two forms. Using $G = \log V_\phi$ resolves this — the formulation and implementation now agree exactly.

### Copter information term

$$I(a_c; \mathcal{B}_t, q_t) = \sum_{x \in \text{observed by } a_c} H(\mathcal{B}_t(x \models O))$$

Depends only on $a_c$. Therefore appears in copter's utility alone — rover deviating does not change $I$, so its presence/absence in $u_r$ has no effect on the proof or on rover behavior.

Tractable surrogate for value of information. $H$ is a reasonable proxy under uniform priors and binary observations; the gap between entropy reduction and true VoI depends on the curvature of the value function with respect to belief updates.

### Private costs

$$C_r(a_r; \mathcal{B}_t) = -\sum_k \log(1 - \mathcal{B}_t(x_r^k \models O))$$

Negative log survival probability. $P(\text{safe}) = \prod(1-B_k) \xrightarrow{-\log} \text{additive}$. Dual to $V_\phi$: $V_\phi^{\text{safety}} = e^{-C_r}$.

$$C_c(a_c; x_{\text{ref}}) = \sum_k d(x_c^k, x_c^{k+1}) + \rho \cdot d(x_c^{T_c}, x_{\text{ref}})$$

Cost-to-come + cost-to-go. $x_{\text{ref}} = x_t^r$ frozen at epoch start.

### Utilities

$$u_r = G - \gamma \cdot C_r(a_r)$$

$$u_c = G + \alpha \cdot I(a_c) - \lambda \cdot C_c(a_c)$$

where $G = \log V_\phi$.

### Term placement rule

| Term | Depends on | In $u_r$? | In $u_c$? | Why |
|------|-----------|-----------|-----------|-----|
| $G = \log V_\phi$ | $a_r$ AND $a_c$ | ✓ | ✓ | Coupled → must be shared for $\Delta G$ to cancel |
| $\alpha I$ | only $a_c$ | ✗ | ✓ | Own-action → can be private to copter |
| $\gamma C_r$ | only $a_r$ | ✓ | ✗ | Own-action → private to rover |
| $\lambda C_c$ | only $a_c$ | ✗ | ✓ | Own-action → private to copter |

**Rule: coupled terms are included in both utilities to enable cancellation. Own-action-only terms appear only in the relevant agent's utility.**

### Parameter asymmetry note

$\alpha$ is a **system design parameter** set by the mission operator — it determines how much the copter is rewarded for exploration relative to mission support. It is public because it is part of the system specification, not a private preference. $\gamma$ and $\lambda$ are **agent-specific sensitivities** reflecting internal operational state — private because they depend on local information unavailable to the partner.

## 8. Potential Function

$$\Phi = G + \alpha I - \gamma C_r - \lambda C_c$$

where $G = \log V_\phi$. Contains every term from both utilities. $\Phi$ is an analyst's tool — agents do not compute, optimize, or know it exists. It does not correspond to any natural system welfare objective. Its role is purely structural: it certifies that selfish improvements cannot cycle.

## 9. Main Result: Exact Potential Game

**Proposition.** The per-epoch private-constraint rover-copter game with finite action spaces is an exact potential game with $\Phi = G + \alpha I - \gamma C_r - \lambda C_c$, where $G = \log V_\phi$.

**Proof.**

*Rover deviates* (fix $a_c$):

$$\Delta u_r = \Delta G - \gamma \Delta C_r$$

$$\Delta \Phi = \Delta G + \alpha \underbrace{\Delta I}_{=0} - \gamma \Delta C_r - \lambda \underbrace{\Delta C_c}_{=0} = \Delta G - \gamma \Delta C_r$$

$$\Delta u_r = \Delta \Phi $$

*Copter deviates* (fix $a_r$):

$$\Delta u_c = \Delta G + \alpha \Delta I - \lambda \Delta C_c$$

$$\Delta \Phi = \Delta G + \alpha \Delta I - \gamma \underbrace{\Delta C_r}_{=0} - \lambda \Delta C_c = \Delta G + \alpha \Delta I - \lambda \Delta C_c$$

$$\Delta u_c = \Delta \Phi \quad \square$$

## 10. Why This Is a Potential Game (Structural Insight)

The proof is three lines. The contribution is the engineering design that makes those three lines work:

| Design choice | What it ensures | Without it |
|--------------|----------------|------------|
| Same $G = \log V_\phi$ in both utilities | $\Delta G$ cancels | Gap = $\Delta G$ in one direction |
| $C_r$ uses $\mathcal{B}_t$ (not $\mathcal{B}_t^+$) | $C_r$ does not depend on $a_c$ → $\Delta C_r = 0$ when copter deviates | $C_r$ coupled → separability fails |
| $x_{\text{ref}}$ frozen at epoch start | $C_c$ does not depend on $a_r$ → $\Delta C_c = 0$ when rover deviates | $C_c$ coupled → separability fails |
| $\alpha I$ in copter only | Legal because $I$ depends only on $a_c$ → $\Delta I = 0$ when rover deviates | Would also work if shared, but less honest |

**The proof is the payoff. The cost function design is the contribution.**

## 11. Corollaries

| # | Statement | Scope |
|---|-----------|-------|
| 1 | $\Phi$'s global maximizer is a pure-strategy NE. Pure NE always exists. | Per-epoch, finite action spaces |
| 2 | For each fixed epoch game $\mathcal{G}_t$, any sequence of unilateral utility-improving moves terminates at a NE in finite steps (FIP). | **Per-epoch only.** The cross-epoch process is not a single fixed game — beliefs and feasible sets evolve after each sensing cycle. |
| 3 | Setting $\gamma = \lambda = 0$ recovers the altruistic limit: $u_r = G = \log V_\phi$, $u_c = G + \alpha I$. Since $\log$ is monotone, the rover optimizes pure mission value ($\operatorname{argmax} G = \operatorname{argmax} V_\phi$); the copter optimizes mission value plus exploration. This mirrors Hashimoto's role structure, though not its exact heuristic scoring function $W(x) = \sum H + \alpha \cdot b_{\max}$. |

**Remark (Algorithm 1 as better-response).** Hashimoto's alternating exploration-execution constitutes one round of better-response per epoch on the epoch game $\mathcal{G}_t$. The exact potential structure guarantees that each such round moves toward a NE of that epoch's game. Cross-epoch progress toward mission completion is driven by belief improvement through Bayesian sensing, not by iterated game play on a fixed game.

**Remark (Decentralized).** Neither agent needs to know the other's sensitivity, predict the other's action, or know $\Phi$ exists. Each myopically optimizes its own $u_i$. The potential game structure guarantees these selfish improvements cannot cycle — every improvement increases $\Phi$, and $\Phi$ is bounded on a finite set.

## 12. Rover Heuristic Admissibility

**Proposition.** $h(x,q) = d_{\text{FSA}}(q,Q_f) \cdot c_{\min}^+(t)$ is admissible for D* Lite on the product graph $(X \times Q)$.

**Assumptions:**
1. Every feasible edge has strictly positive cost (beliefs clipped to $[\varepsilon, 1-\varepsilon]$).
2. The FSA has no $\varepsilon$-transitions — each automaton transition along a feasible path requires at least one motion edge in the product graph.

**Definitions:**
- $d_{\text{FSA}}(q, Q_f)$: min FSA transitions from $q$ to acceptance (BFS on FSA graph, computed once)
- $c_{\min}^+(t) = \min\{c_t(e) : e \in E_t, c_t(e) > 0\}$: smallest positive edge cost (recomputed each epoch, $O(|X| \cdot |Q| \cdot \deg_Q)$)

**Proof.** Any accepting path traverses $\geq d_{\text{FSA}}$ FSA transitions (by definition). By Assumption 2, each requires $\geq 1$ motion edge. By Assumption 1, each motion edge costs $\geq c_{\min}^+ > 0$. Therefore $h(x,q) \leq$ true cost-to-go. $\square$

## 13. Implementation (Solvers)

Solver choices are implementation decisions, separate from the game definition.

### Rover: D* Lite on $(X \times Q)$

**Edge cost:** $c(e) = -\log B_{en}(x', q, q') + \gamma \cdot (-\log(1-\mathcal{B}_t(x' \models O)))$

Both terms edge-additive. Minimizing total path cost $= -G + \gamma C_r$ is equivalent to maximizing $u_r = G - \gamma C_r$. The code implements this via multiplicative discounting $V \leftarrow V \cdot (1-B_O)^\gamma$, which maximizes $\exp(G - \gamma C_r)$; since $\exp$ is strictly monotone, the optimal path is the same as maximizing $u_r$ directly. D* Lite with heuristic from §12 computes the **exact best response** (optimal path in the finite product graph). D* Lite's incremental replanning updates only affected nodes when beliefs change across epochs.

At $\gamma = 0$, copter-resolved obstacle beliefs still affect rover planning through $B_{en}$: lowering $\mathcal{B}_t(x \models O)$ shifts probability mass away from the dead state and toward non-dead FSA successors. However, under uniform target priors, this effect is nearly spatially homogeneous, so explored cells receive similar cost reductions and the rover sees only a weak directional gradient. When $\gamma > 0$, the explicit term $\gamma(-\log(1-\mathcal{B}_t(x \models O)))$ turns local obstacle sensing into a cell-specific risk gradient, breaking this symmetry.

**First-order calibration heuristic for $\gamma$.** This is not an exact theorem, only a one-step approximation for choosing a reasonable risk weight. Suppose one neighbor has already been cleared by the copter ($B_O \approx 0$) while the remaining $N-1$ neighbors are still unexplored ($B_O = p_O$). For the common self-loop case $q \to q$, the extra per-edge cost of entering an unexplored cell is approximately

$$\Delta c \approx (1+\gamma)\rho, \qquad \rho = -\log(1-p_O).$$

The competing incentive is the possibility that an unexplored neighbor immediately advances the automaton. Let $p_{\mathrm{progress}}(q)$ denote the aggregate per-cell probability of a progress-making label at the current FSA stage:

$$p_{\mathrm{progress}}(0) \approx \Pr[A \lor B \lor C], \quad p_{\mathrm{progress}}(1) \approx \Pr[A \lor C], \quad p_{\mathrm{progress}}(2) \approx \Pr[A \lor D].$$

If the expected shortcut value of such a discovery is $V_{\mathrm{shortcut}}$, a first-order threshold estimate is

$$\gamma_q^* \approx \max\!\left(0,\; \frac{(N-1)\,p_{\mathrm{progress}}(q)\,V_{\mathrm{shortcut}}}{-\log(1-p_O)} - 1\right).$$

Under the additional approximations $V_{\mathrm{shortcut}} \approx 1$ and $-\log(1-p_O) \approx p_O$, this reduces to

$$\gamma_q^* \sim \frac{(N-1)\,p_{\mathrm{progress}}(q)}{p_O} - 1.$$

The sweep experiments should therefore be interpreted as testing a threshold effect, not as validating a closed-form optimum.

### Copter: Utility-optimal finite-horizon search

```
1. ENUMERATE all copter action sequences of length T_c
2. SCORE each by α · I(a_c; B_t) − λ · C_c(a_c; x_ref)
3. FILTER by energy budget
4. SELECT highest-scoring feasible sequence
5. EXECUTE first action, sense, and replan next epoch
```

For fixed $T_c$, this is an exact maximization over the finite discrete copter action space. The post-hoc utility check in simulation remains as a uniform acceptance gate across copter modes.

## 14. Two Layers

| Layer | What | Mechanism | Potential game? |
|-------|------|-----------|----------------|
| **Hard** | Physics | $\tau_r$ prunes graph; $E_t^c$ shrinks feasible set | Defines feasible sets only |
| **Soft** | Preference | γ, λ weight private costs | Defines utility structure |

## 15. Variable Table

| Variable | Role | Public/Private | Static/Dynamic |
|----------|------|---------------|----------------|
| $\tau_r$ | Obstacle threshold | Public | Static |
| $E_t^c$ | Remaining energy | Copter internal | Dynamic |
| $\gamma$ | Risk sensitivity (MRS) | Private | Static |
| $\lambda$ | Energy sensitivity (MRS) | Private | Static |
| $\alpha$ | Exploration reward weight | **Public** (system design) | Static |
| $\rho$ | Return energy ratio | Public | Static |

## 16. Acknowledged Modeling Choices

| Choice | Why | Consequence |
|--------|-----|-------------|
| $V_\phi$ shared in both utilities | Enables $\Delta V_\phi$ cancellation → exact potential | Design choice, not forced by domain |
| $C_r$ uses $\mathcal{B}_t$ not $\mathcal{B}_t^+$ | Separability | Conservative (pessimistic) risk estimate |
| $\alpha I$ copter-only | $I$ own-action-only → legal; reflects role division | More honest than shared version |
| $I$ = entropy surrogate | Tractable | $\neq$ true VoI; gap depends on value function curvature |
| $V_\phi$ = path belief, not $V^*$ | Compatible with graph search | $V_\phi \leq V^*$; underestimates |
| Simultaneous strategic model | Standard for potential games | Execution is actually sequential |
| $B_t$ vs $B_t^+$ split in $u_r$ | Separability | Optimistic mission + pessimistic survival |
| $x_{\text{ref}}$ frozen | Prevents $C_c \to a_r$ dependency | Return cost from current, not future position |
| Deterministic motion | Isolates informational uncertainty | No slip model |
| Strict-positive edge costs | Heuristic admissibility | Beliefs clipped to $[\varepsilon, 1-\varepsilon]$ |
| FSA no $\varepsilon$-transitions | Heuristic admissibility | Each FSA step requires $\geq 1$ motion edge |
| All parameters user-set | No calibration data | Interpretable ranges |

## 17. What's NOT in the Formulation

| Removed | Why |
|---------|-----|
| $\beta$, $\hat{\gamma}$, $\hat{\lambda}$ | Exact potential. Optional robustness analysis. |
| $\alpha I$ in $u_r$ | Own-action-only → no effect on rover. Removal is more honest. |
| Rover exploration reward | Rover sensing is side effect in $\mathcal{B}_{t+1}$, not planned objective. |
| $\nu \Sigma H$ in $C_r$ | $C_r = -\log(\text{survival})$ is principled. No extra hyperparameter. |
| $\eta T_c$ in $C_c$ | Constant when $T_c$ fixed. |
| Stochastic motion | Orthogonal to game theory. |
| Dynamic $\gamma_t, \lambda_t$ | Physical dynamics in hard layer. |
| $E_t^c$ in public state | Rover doesn't use it. |
| Cross-epoch convergence claim | FIP is per-epoch. Cross-epoch driven by Bayesian sensing. |
| "Exactly Hashimoto" claim | $\gamma=\lambda=0$ gives altruistic limit with analogous role structure, not identical objective functions. |
| Solver names in action definition | D* Lite, VI, exhaustive copter search are implementation, not formulation. |

## 18. Summary

We model the rover-copter system as a two-player game with asymmetric roles. Both agents share the mission value $V_\phi$ — the belief-weighted probability of reaching an accepting FSA state along the planned path. The copter additionally receives an exploration reward $\alpha I$ reflecting its designated scouting role. Each agent bears a private operational cost: the rover's is the negative log survival probability (dual to $V_\phi$); the copter's is flight energy including return margin. Coupled terms ($V_\phi$) appear in both utilities by design; own-action-only terms ($\alpha I$, $C_r$, $C_c$) appear only in the relevant agent's utility.

The game is an exact potential game with $\Phi = G + \alpha I - \gamma C_r - \lambda C_c$, where $G = \log V_\phi$. The three-line proof follows from action-separability of private costs and identical placement of the coupled mission term. Pure-strategy Nash equilibria exist for each epoch game, and the finite improvement property guarantees that unilateral utility-improving moves terminate. Hashimoto's alternating algorithm constitutes one round of better-response per epoch; cross-epoch mission progress is driven by Bayesian belief improvement.

The proof itself is trivial. The contribution is the cost function design — the $\mathcal{B}_t / \mathcal{B}_t^+$ split, the frozen $x_{\text{ref}}$, the role-asymmetric information term, the $-\log$ survival cost — that makes the potential game structure possible. Setting $\gamma = \lambda = 0$ recovers an altruistic limit mirroring Hashimoto's role division.
