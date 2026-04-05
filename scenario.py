"""
scenario.py

- define one simulation configuration
- parse raw scenario dicts into typed objects

class
- `SimulationScenario`: typed input for one run

function
- none
"""

from dataclasses import dataclass


# --- Scenario Definition ---
@dataclass(slots=True)
class SimulationScenario:
    """One simulation configuration."""

    name: str = ''
    rover_start: tuple[int, int] = (0, 0)
    copter_start: tuple[int, int] = (0, 0)
    copter_mode: str = 'global'
    alpha: float = 1.5
    gamma: float = 0.0
    lambda_: float = 0.0
    rho: float = 1.0
    tau_r: float = 0.9
    D_c: float = float('inf')
    # Checked-in scenario files keep the epoch schedule fixed and usually omit
    # these keys, but the parser still accepts overrides for local experiments.
    T_c: int = 5
    T_r: int = 3
    vi_steps: int | None = None
    max_k: int = 600
    seed: int = 42

    @classmethod
    def from_dict(cls, raw: dict) -> "SimulationScenario":
        """Build one scenario from a JSON-style dict."""
        if 'planner_mode' in raw:
            raise ValueError(
                "planner_mode is no longer used; set copter_mode to "
                "'local', 'global', or 'utility'."
            )

        D_c = raw.get('D_c')
        if D_c is None:
            D_c = float('inf')

        copter_mode = raw.get('copter_mode', 'global')
        if copter_mode not in ('local', 'global', 'utility'):
            raise ValueError(
                f"Unsupported copter_mode={copter_mode!r}; "
                "use 'local', 'global', or 'utility'."
            )

        return cls(
            name         = raw.get('name', ''),
            rover_start  = tuple(raw.get('rover_start', (0, 0))),
            copter_start = tuple(raw.get('copter_start', (0, 0))),
            copter_mode  = copter_mode,
            alpha        = raw.get('alpha', 1.5),
            gamma        = raw.get('gamma', 0.0),
            # Accept both the current JSON key and the older alias.
            lambda_      = raw.get('lambda_', raw.get('lambda', 0.0)),
            rho          = raw.get('rho', 1.0),
            tau_r        = raw.get('tau_r', 0.9),
            D_c          = D_c,
            T_c          = raw.get('T_c', 5),
            T_r          = raw.get('T_r', 3),
            vi_steps     = raw.get('vi_steps'),
            max_k        = raw.get('max_k', 600),
            seed         = raw.get('seed', 42),
        )
