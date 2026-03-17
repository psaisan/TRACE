#!/usr/bin/env python3

import os
import json
import matplotlib.pyplot as plt

from trace.api import make_dense_n_grid, run_all_reference_scenarios
from trace.plotting import make_scenario_figure, make_summary_grid


def main():
    os.makedirs("figs", exist_ok=True)

    n_grid = make_dense_n_grid(
        n_min=30,
        n_mid=1200,
        n_max=5000,
        n_low=16,
        n_high=8,
    )

    scenarios = [
        "sample_efficiency",
        "persistent_advantage",
        "no_advantage",
        "lossy_translation",
    ]

    all_results, all_meta = run_all_reference_scenarios(
        scenarios=scenarios,
        n_samples=4000,
        n_grid=n_grid,
        repeats=64,
        train_fraction=0.7,
    )

    for scenario in scenarios:
        print(f"\n=== {scenario} ===")
        print(json.dumps(all_results[scenario]["summary"], indent=2))

        fig = make_scenario_figure(
            scenario_name=scenario,
            result=all_results[scenario],
            meta=all_meta[scenario],
            save_path=f"figs/{scenario}_full_panel.png",
        )
        plt.close(fig)

    summary_fig = make_summary_grid(
        all_results=all_results,
        save_path="figs/trace_reference_scenarios_summary.png",
    )
    plt.close(summary_fig)

    print("\nSaved figures:")
    for scenario in scenarios:
        print(f"  figs/{scenario}_full_panel.png")
    print("  figs/trace_reference_scenarios_summary.png")


if __name__ == "__main__":
    main()
