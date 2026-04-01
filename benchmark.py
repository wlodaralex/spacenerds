"""compare h0, h2, and h6 on the _both_private scenario."""

import csv
import json
import time
from functools import partial
from pathlib import Path
from statistics import fmean

import simulation
from planning import DStarLitePlanner


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'
SUMMARY_CSV = OUTPUT_DIR / 'heuristic_benchmark_summary.csv'
EPOCH_CSV = OUTPUT_DIR / 'heuristic_benchmark_epochs.csv'


class TimedDStarLitePlanner(DStarLitePlanner):
    """planner wrapper that records runtime and heuristic stats."""

    latest_instance = None

    def __init__(self, spatial, beliefs, gamma=0.0, tau_r=1.0,
                 use_h2=False, use_h6=False):
        self.initial_planner_ms = 0.0
        self.initial_expansions = 0
        self.initial_heuristic_stats = {
            'tq0_size': 0,
            'tq1_size': 0,
            'tq2_size': 0,
            'h_mean_active': 0.0,
        }
        self.replan_metrics = []

        start = time.perf_counter()
        super().__init__(
            spatial,
            beliefs,
            gamma=gamma,
            tau_r=tau_r,
            use_h2=use_h2,
            use_h6=use_h6,
        )
        self.initial_planner_ms = (time.perf_counter() - start) * 1000.0
        self.initial_expansions = self.n_expanded
        self.initial_heuristic_stats = self._heuristic_stats()
        type(self).latest_instance = self

    def replan(self, beliefs, risk_beliefs=None):
        start = time.perf_counter()
        result = super().replan(beliefs, risk_beliefs=risk_beliefs)
        self.replan_metrics.append({
            'epoch': len(self.replan_metrics),
            'planner_ms': (time.perf_counter() - start) * 1000.0,
            'n_expanded': self.n_expanded,
            **self._heuristic_stats(),
        })
        return result


def _load_scenario(name):
    """load one named scenario from scenarios.json."""
    with open(BASE_DIR / 'scenarios.json') as f:
        scenarios = json.load(f)

    for scenario in scenarios:
        if scenario['name'] == name:
            return {
                **scenario,
                'rover_start': tuple(scenario['rover_start']),
                'copter_start': tuple(scenario['copter_start']),
            }
    raise ValueError(f"Scenario not found: {name}")


def _run_variant(scenario, use_h2=False, use_h6=False):
    """run one heuristic variant with a timed planner wrapper."""
    original_cls = simulation.DStarLitePlanner
    TimedDStarLitePlanner.latest_instance = None
    simulation.DStarLitePlanner = partial(
        TimedDStarLitePlanner,
        use_h2=use_h2,
        use_h6=use_h6,
    )
    try:
        hist = simulation.run_simulation(
            rover_start=scenario['rover_start'],
            copter_start=scenario['copter_start'],
            T_c=scenario['T_c'],
            T_r=scenario['T_r'],
            alpha=scenario['alpha'],
            gamma=scenario.get('gamma', 0.0),
            lambda_=scenario.get('lambda', 0.0),
            tau_r=scenario.get('tau_r', 1.0),
            vi_steps=scenario['vi_steps'],
            max_k=scenario['max_k'],
            copter_mode=scenario.get('copter_mode', 'global'),
            seed=scenario['seed'],
            copter_energy=scenario.get('copter_energy', float('inf')) or float('inf'),
        )
        planner = TimedDStarLitePlanner.latest_instance
        if planner is None:
            raise RuntimeError('Timed planner instance was not created.')
        return hist, planner
    finally:
        simulation.DStarLitePlanner = original_cls


def _summary_row(label, hist, planner):
    """build one summary row."""
    replans = planner.replan_metrics
    total_replan_expansions = sum(m['n_expanded'] for m in replans)
    total_replan_ms = sum(m['planner_ms'] for m in replans)
    return {
        'heuristic': label,
        'initial_expansions': planner.initial_expansions,
        'avg_replan_expansions': fmean(m['n_expanded'] for m in replans) if replans else 0.0,
        'total_replan_expansions': total_replan_expansions,
        'initial_planner_ms': planner.initial_planner_ms,
        'avg_replan_ms': fmean(m['planner_ms'] for m in replans) if replans else 0.0,
        'total_planner_ms': planner.initial_planner_ms + total_replan_ms,
        'complete': hist.complete,
        'fail': hist.fail,
        'k_final': hist.k_final,
        'which_phi': hist.which_phi or '',
        'epochs': len(hist.epochs),
    }


def _epoch_rows(label, planner):
    """build per-epoch planner rows."""
    rows = [{
        'heuristic': label,
        'epoch': -1,
        'planner_ms': planner.initial_planner_ms,
        'n_expanded': planner.initial_expansions,
        **planner.initial_heuristic_stats,
    }]
    for metric in planner.replan_metrics:
        rows.append({
            'heuristic': label,
            'epoch': metric['epoch'],
            'planner_ms': metric['planner_ms'],
            'n_expanded': metric['n_expanded'],
            'tq0_size': metric['tq0_size'],
            'tq1_size': metric['tq1_size'],
            'tq2_size': metric['tq2_size'],
            'h_mean_active': metric['h_mean_active'],
        })
    return rows


def _write_csv(path, fieldnames, rows):
    """write one csv file."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(summary_rows):
    """print the benchmark table."""
    print("\nHeuristic benchmark on _both_private")
    print(f"{'heuristic':<10} {'init_exp':>10} {'avg_replan_exp':>15} "
          f"{'init_ms':>10} {'avg_replan_ms':>14} {'status':>10} "
          f"{'k_final':>8} {'mission':>16}")
    for row in summary_rows:
        if row['complete']:
            status = 'complete'
        elif row['fail']:
            status = 'fail'
        else:
            status = 'timeout'
        mission = row['which_phi'] or '-'
        print(f"{row['heuristic']:<10} "
              f"{row['initial_expansions']:>10} "
              f"{row['avg_replan_expansions']:>15.2f} "
              f"{row['initial_planner_ms']:>10.2f} "
              f"{row['avg_replan_ms']:>14.2f} "
              f"{status:>10} "
              f"{row['k_final']:>8} "
              f"{mission:>16}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    scenario = _load_scenario('_both_private')

    summary_rows = []
    epoch_rows = []
    variants = (
        ('h0', {'use_h2': False, 'use_h6': False}),
        ('h2', {'use_h2': True, 'use_h6': False}),
        ('h6', {'use_h2': False, 'use_h6': True}),
    )
    for label, kwargs in variants:
        hist, planner = _run_variant(scenario, **kwargs)
        summary_rows.append(_summary_row(label, hist, planner))
        epoch_rows.extend(_epoch_rows(label, planner))

    _write_csv(
        SUMMARY_CSV,
        ['heuristic', 'initial_expansions', 'avg_replan_expansions',
         'total_replan_expansions', 'initial_planner_ms', 'avg_replan_ms',
         'total_planner_ms', 'complete', 'fail', 'k_final', 'which_phi',
         'epochs'],
        summary_rows,
    )
    _write_csv(
        EPOCH_CSV,
        ['heuristic', 'epoch', 'planner_ms', 'n_expanded',
         'tq0_size', 'tq1_size', 'tq2_size', 'h_mean_active'],
        epoch_rows,
    )

    _print_summary(summary_rows)
    print(f"\nSaved: {SUMMARY_CSV}")
    print(f"Saved: {EPOCH_CSV}")


if __name__ == '__main__':
    main()
