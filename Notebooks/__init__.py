"""Translator-assisted biomarker simulation toolkit.

This package consolidates notebook code for:
- linear Gaussian worlds and theory-driven panels
- nonlinear synthetic worlds and learning-curve experiments
- site-shift shortcut threat models
- plotting and curve summaries
"""

from .config import (
    LinearWorldConfig,
    CorePanelConfig,
    ThreatConfig,
    LearningCurveConfig,
    NonlinearWorldConfig,
    NonlinearSweepConfig,
)
from .linear import (
    sample_world,
    sample_XZ_from_world,
    fit_translator_ridge,
    compute_core_panel_data,
    compute_site_shift_threat,
    make_all_linear_panels_and_save,
)
from .nonlinear import (
    sample_nonlinear_world,
    sample_XZ_from_world_nonlinear,
    fit_translator_rff_ridge,
    compute_learning_curve_auc_vs_n_nonlinear,
    compute_nuisance_sweep_nonlinear,
)
from .plotting import (
    plot_fig_ab,
    plot_fig_cd,
    plot_fig_ef,
    plot_fig_gh_threat,
    plot_auc_learning_curve_smoothed,
    plot_auc_learning_curve_smoothed_with_band,
)
from .summary import summarize_curve_metrics, summarize_many_curves
from .presets import get_paper_nonlinear_sweep_specs

__all__ = [
    "LinearWorldConfig",
    "CorePanelConfig",
    "ThreatConfig",
    "LearningCurveConfig",
    "NonlinearWorldConfig",
    "NonlinearSweepConfig",
    "sample_world",
    "sample_XZ_from_world",
    "fit_translator_ridge",
    "compute_core_panel_data",
    "compute_site_shift_threat",
    "make_all_linear_panels_and_save",
    "sample_nonlinear_world",
    "sample_XZ_from_world_nonlinear",
    "fit_translator_rff_ridge",
    "compute_learning_curve_auc_vs_n_nonlinear",
    "compute_nuisance_sweep_nonlinear",
    "plot_fig_ab",
    "plot_fig_cd",
    "plot_fig_ef",
    "plot_fig_gh_threat",
    "plot_auc_learning_curve_smoothed",
    "plot_auc_learning_curve_smoothed_with_band",
    "summarize_curve_metrics",
    "summarize_many_curves",
    "get_paper_nonlinear_sweep_specs",
]
