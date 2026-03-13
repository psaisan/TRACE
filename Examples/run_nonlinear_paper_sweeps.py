from pathlib import Path

import matplotlib.pyplot as plt

from translator_toolkit import (
    compute_learning_curve_auc_vs_n_nonlinear,
    get_paper_nonlinear_sweep_specs,
    plot_auc_learning_curve_smoothed_with_band,
    summarize_many_curves,
)
from translator_toolkit.utils import make_dense_n_grid


if __name__ == "__main__":
    outdir = Path("figs_nonlinear")
    outdir.mkdir(exist_ok=True)

    n_grid = make_dense_n_grid(n_min=30, n_mid=3000, n_max=200000, n_low=26, n_high=14)
    specs = get_paper_nonlinear_sweep_specs(n_grid)

    curves = {}
    for spec in specs:
        curves[spec["key"]] = compute_learning_curve_auc_vs_n_nonlinear(**spec["params"])
        fig = plot_auc_learning_curve_smoothed_with_band(curves[spec["key"]], title=spec["title"], band_mode="ci95")
        fig.savefig(outdir / f"{spec['key']}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    summary = summarize_many_curves(curves)
    summary.to_csv(outdir / "learning_curve_summary_metrics.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
