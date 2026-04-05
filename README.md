# spacenerds

Rover-copter simulation code for AER1516 Robot Motion Planning, Winter 2026.

This repo simulates a rover that must satisfy a mission automaton on a fixed 10x10 grid while a copter scouts the environment and updates shared beliefs. It supports two planning families:

- baseline Hashimoto-style control flow selected by `copter_mode="local"` or `"global"`
- game-theoretic utility mode selected by `copter_mode="utility"` with rover risk weight `gamma`, copter flight-cost weight `lambda_`, and a hard copter distance budget `D_c`

The theory and notation live in [formulation.md](formulation.md). This README is the repo-level guide: how to run it, how the code is organized, what the scenarios mean, and where outputs come from.

## Quick Start

From the repo root:

```bash
python3 -m pip install numpy matplotlib pandas pillow
python3 main.py
```

Outputs are written to `outputs/`.

Notes:

- `pandas` is only needed for CSV export from `history.py`.
- `pillow` is only needed if you enable animations and `ffmpeg` is unavailable.
- `main.py` currently generates trajectory and convergence plots by default. Other plot/animation calls are present but commented out.

## What The Repo Simulates

At a high level, each epoch is:

1. Freeze the epoch-start belief map `B_t`.
2. Let the copter explore for `T_c` steps and update beliefs.
3. Treat the post-copter belief map as `B_t^+`.
4. Replan the rover on `B_t^+` while evaluating rover risk against the frozen `B_t`.
5. Execute the rover for `T_r` steps.
6. Log metrics, update history, and stop on success, obstacle failure, or timeout.

Mission completion is tracked by a small finite-state automaton:

- `φ1`: find `A`
- `φ2`: find `B` then `C`
- `φ3`: find `C` then `D`

Obstacle `O` sends the rover to the dead state.

## How To Run

### Standard run

```bash
python3 main.py
```

`main.py`:

- loads every scenario from [`scenarios.json`](scenarios.json)
- runs each scenario through [`run_simulation(...)`](simulation.py)
- saves one CSV per scenario
- converts history to the legacy plotting dict
- renders the enabled figures



### Enabling more plots

The following plot/animation hooks already exist in [`main.py`](main.py), but some are commented out:

- belief snapshot grid
- final belief maps
- rover value-function heatmap
- belief animation
- unified animation with obstacle belief, agents, and planner value overlay

If you enable animations:

- MP4 save uses `ffmpeg` when available
- otherwise the code falls back to GIF via `pillow`

## Scenario Configuration

Scenarios are JSON objects parsed by [`SimulationScenario.from_dict(...)`](scenario.py).

The checked-in scenarios keep the epoch schedule fixed, so `T_c = 5` and
`T_r = 3` are omitted from [`scenarios.json`](scenarios.json).
`scenario.py` still accepts them if you want a local override.

Scenario fields are grouped as follows:

| Type | Field | Meaning |
|---|---|---|
| Setup | `name` | Scenario label used in logs and output filenames |
| Setup | `rover_start` | Initial rover cell `[row, col]` |
| Setup | `copter_start` | Initial copter cell `[row, col]` |
| Mode | `copter_mode` | `local`, `global`, or `utility` (`game*` scenarios default to `utility` if omitted) |
| Weights | `alpha` | Exploration reward weight |
| Weights | `gamma` | Rover risk sensitivity |
| Weights | `lambda_` | Copter flight-cost sensitivity |
| Weights | `rho` | Return-distance multiplier inside copter cost |
| Hard constraints | `tau_r` | Rover obstacle-pruning threshold |
| Hard constraints | `D_c` | Copter distance budget; `null` means infinite |
| Utility runtime | `copter_top_k` | Optional mission-aware shortlist size; default `5` (`0` disables reranking) |
| Utility runtime | `copter_mc_samples` | Optional Monte Carlo samples per shortlisted sequence; default `3` (`0` disables reranking) |
| Runtime | `vi_steps` | Baseline value-iteration iteration cap |
| Runtime | `max_k` | Global simulation timestep limit |
| Runtime | `seed` | RNG seed |

Mode rules:

- In baseline mode:
  - `copter_mode = "local"` or `"global"`
  - `gamma = 0`, `lambda_ = 0`

- In utility mode:
  - `copter_mode = "utility"`; in checked-in `game*` scenarios it may be omitted
  - hard constraints: `D_c` for the copter, `tau_r` for the rover

The default `scenarios.json` currently includes:

- `game_copter_altruistic`
- `game_rover_altruistic`
- `game_both_private`
- `game_both_altruistic`
- `baseline_global`

## Outputs

For each scenario named `X`, `main.py` writes files with prefix `outputs/scenX`.

By default you should expect:

- `outputs/scenX.csv`
- `outputs/scenX_trajectories.png`
- `outputs/scenX_convergence.png`

If you uncomment additional plot calls in [`main.py`](main.py), you can also produce:

- `*_belief_snapshots.png`
- `*_final_beliefs.png`
- `*_value_function_q0.png`
- `*_beliefs_animation.mp4` or `.gif`
- `*_unified_animation.mp4` or `.gif`

The CSV is generated from [`SimHistory.to_dataframe()`](history.py) and contains epoch-level metrics such as:

Shared mission terms:

- `G`: common mission term. In the CSV it is logged at the realised epoch endpoint

  $$G = \log V_\phi$$

- `Phi`: exact potential / joint epoch score

  $$\Phi = G + \alpha I - \gamma C_r - \lambda C_c$$

Private costs and information:

- `C_r`: rover risk cost over the executed rover segment.

  $$C_r = -\sum_k \log(1 - \mathcal{B}_t(x_r^k \models O))$$

- `C_c`: copter travel cost for the committed scouting action.

  $$C_c = \sum_k d(x_c^k, x_c^{k+1}) + \rho \, d(x_c^{T_c}, x_{\mathrm{ref}})$$

- `I`: copter information-gain term over the union of observed cells.

  $$I = \sum_{x \in \mathrm{observed}(a_c)} H(\mathcal{B}_t(x \models O))$$

Agent utilities:

- `u_r`: rover utility from formulation §3.4.

  $$u_r = G - \gamma C_r$$

- `u_c`: copter utility from formulation §3.4.

  $$u_c = G + \alpha I - \lambda C_c$$

Runtime accounting:

- `D_c`: remaining copter distance budget after the epoch.
- `n_expanded`: rover planner work count. In utility mode this is D* Lite node expansions; in baseline mode it is the VI sweep count returned by the baseline rover planner.

## Repo Map

### Core files

| File | Role |
|---|---|
| [`main.py`](main.py) | Top-level entry point, scenario loop, output generation |
| [`scenario.py`](scenario.py) | Typed scenario schema and JSON parsing |
| [`simulation.py`](simulation.py) | Epoch loop, baseline/game dispatch, metric logging |
| [`agents.py`](agents.py) | Runtime wrappers that execute planner outputs and apply sensing |
| [`planning.py`](planning.py) | Rover planners, copter planners, shared planning helpers |
| [`history.py`](history.py) | Run history container, CSV export, plotting dict export |
| [`visualization.py`](visualization.py) | Static plots and animations |

### Shared model files

| File | Role |
|---|---|
| [`environment.py`](environment.py) | Fixed 10x10 map, AP labels, action set, motion helpers |
| [`sensor.py`](sensor.py) | Sensor accuracy model, Bayesian belief update, initial priors |
| [`fsa.py`](fsa.py) | Mission automaton and belief-weighted FSA transition probabilities |
| [`formulation.md`](formulation.md) | Theory, notation, assumptions, solver notes |

## Planning Modes

### Baseline path

Triggered when `copter_mode` is `local` or `global`.

- Rover planner: finite-horizon value iteration in [`rover_value_iteration(...)`](planning.py)
- Copter planner:
  - `local`: greedy one-step acquisition in `plan_copter_action_local(...)`
  - `global`: target selection plus copter value iteration in `plan_copter_action_global(...)`
- Baseline rover additionally computes `b_max`, the rover reachability field used in Hashimoto-style acquisition

### Game-theoretic path

Triggered when `copter_mode` is `utility` or when a checked-in `game*`
scenario omits `copter_mode`.

- Rover planner: [`DStarLitePlanner`](planning.py) on the product graph `(grid cell, FSA state)`
  - default heuristic: `d_FSA(q, Q_f) * c_min^+` admissible lower bound
- Copter planner: exhaustive surrogate scoring over all `9^T_c` action sequences in `plan_copter_action_game(...)`, followed by top-`K` mission-aware reranking
- Copter surrogate score:
  - reward: `alpha * information_gain`
  - penalty: `lambda_ * total_cost`
  - hard constraint: `total_cost <= D_c`
- Copter mission-aware rerank:
  - keep the best `copter_top_k` feasible surrogate sequences
  - for each, sample `copter_mc_samples` hypothetical post-copter obstacle maps
  - run a mission-only rover replan on each sample to estimate the shared mission term `G`
  - rerank by `estimated_G + alpha * information_gain - lambda_ * total_cost`

Implementation detail that matters:

- In game mode, rover mission progress uses post-copter beliefs `B_t^+`.
- Rover risk still uses the frozen epoch-start obstacle beliefs `B_t`.
- This split is intentional and documented in [`formulation.md`](formulation.md).

## Runtime Flow In Code

If you need to trace one scenario end to end:

1. [`main.py`](main.py) loads scenarios and calls `run_simulation(...)`.
2. [`simulation.py`](simulation.py) builds the initial state and history.
3. `SimulationRunner` decides baseline vs utility mode from `copter_mode`
   after `SimulationScenario.from_dict()` applies the `game* -> utility`
   shorthand.
4. At each epoch:
   - freeze `beliefs_t`
   - run copter exploration
   - record copter substeps and updated beliefs
   - replan the rover
   - execute the rover
   - compute `Phi`, `G`, `C_r`, `C_c`, `I`, `u_r`, `u_c`
   - append an `EpochRecord`
5. Final history is exported to CSV and plotting helpers.

## Environment And Beliefs

The environment is fixed in [`environment.py`](environment.py):

- grid size: `10 x 10`
- APs: `A`, `B`, `C`, `D`, `O`
- motion: stay plus 8-connected actions

Belief initialization in [`sensor.py`](sensor.py):

- target prior: `0.1`
- obstacle prior: `0.3`

Sensor model:

- rover sensing radius/shape parameters: `R = 2`, `M = 0.5`
- copter obstacle sensing radius/shape parameters: `R = 4`, `M = 0.4`
