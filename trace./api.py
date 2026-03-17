from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from .world import generate_linear_trace_dataset
from .diagnostics import run_trace_diagnostic


def make_dense_n_grid(
    n_min: int = 30,
    n_mid: int = 1200,
    n_max: int = 5000,
    n_low: int = 16,
    n_high: int = 8,
) -> np.ndarray:
    low = np.unique(np.round(np.logspace(np.log10(n_min), np.log10(n_mid), n_low)).astype(int))
    high = np.unique(
        np.round(np.logspace(np.log10(max(n_mid + 1, 1200)), np.log10(n_max), n_high)).astype(int)
    )
    return np.unique(np.concatenate([low, high]))


def run_reference_scenario(
    regime: str,
    n_samples: int = 4000,
    dataset_seed: int = 7,
    diagnostic_seed: int = 123,
    n_grid: Optional[np.ndarray] = None,
    repeats: int = 64,
    train_fraction: float = 0.7,
    **overrides,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if n_grid is None:
        n_grid = make_dense_n_grid()

    ds = generate_linear_trace_dataset(
        regime=regime,
        n_samples=n_samples,
        seed=dataset_seed,
        **overrides,
    )

    meta = ds["meta"]
    result = run_trace_diagnostic(
        X=ds["X"],
        H=ds["H"],
        y=ds["y"],
        n_grid=n_grid,
        repeats=repeats,
        train_fraction=train_fraction,
        seed=diagnostic_seed,
        C_X=meta["C_X"],
        C_H=meta["C_H"],
    )
    return ds, result


def run_all_reference_scenarios(
    scenarios: Optional[List[str]] = None,
    n_samples: int = 4000,
    n_grid: Optional[np.ndarray] = None,
    repeats: int = 64,
    train_fraction: float = 0.7,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if scenarios is None:
        scenarios = [
            "sample_efficiency",
            "persistent_advantage",
            "no_advantage",
            "lossy_translation",
        ]

    if n_grid is None:
        n_grid = make_dense_n_grid()

    all_results = {}
    all_meta = {}

    for i, scenario in enumerate(scenarios):
        ds, result = run_reference_scenario(
            regime=scenario,
            n_samples=n_samples,
            dataset_seed=7 + 17 * i,
            diagnostic_seed=123 + i,
            n_grid=n_grid,
            repeats=repeats,
            train_fraction=train_fraction,
        )
        all_results[scenario] = result
        all_meta[scenario] = ds["meta"]

    return all_results, all_meta
