#!/usr/bin/env python3

import os
import json
import matplotlib.pyplot as plt

from trace.api import make_dense_n_grid, run_reference_scenario
from trace.plotting import make_scenario_figure


def main():
    os.makedirs("figs", exist_ok=True)

    scenario = "persistent_advantage"

    n_grid = make_dense_n_grid(
        n_min=30,
        n_mid=1200,
        n_max=5000,
        n_low=16,
        n_high=8,
    )

    ds, result = run_reference_scenario(
        regime=scenario,
        n_samples=4000,
        dataset_seed=7,
        diagnostic_seed=123,
        n_grid=n_grid,
        repeats=64,
        train_fraction=0.7,
    )

    print("\n=== meta ===")
    print(json.dumps(ds["meta"], indent=2))

    print(f"\n=== {scenario} summary ===")
    print(json.dumps(result["summary"], indent=2))

    print("\n=== regime scores ===")
    print(json.dumps(result["regime_scores"], indent=2))

    fig = make_scenario_figure(
        scenario_name=scenario,
        result=result,
        meta=ds["meta"],
        save_path=f"figs/{scenario}_single_case.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
