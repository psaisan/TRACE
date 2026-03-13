# translator-toolkit

A cleaned first-pass package that consolidates the notebook code into a reusable toolkit.

## What this consolidates

The notebooks you pasted contained several overlapping states of the same codebase:

- repeated helper functions
- repeated linear and nonlinear world definitions
- multiple incompatible nonlinear parameterizations
- plotting code mixed with simulation code
- notebook-state errors such as `KeyError: 'latent_axis_strength'` and function-signature drift

This toolkit normalizes that into a single package with consistent entry points.

## Package layout

- `translator_toolkit/linear.py`
  - linear Gaussian world
  - DPI / Bayes ceiling panels
  - nuisance / corruption / phase diagram sweeps
  - site-shift shortcut threat model
- `translator_toolkit/nonlinear.py`
  - nonlinear synthetic world
  - ridge and random-Fourier-feature translators
  - nonlinear learning curves and nuisance sweeps
- `translator_toolkit/plotting.py`
  - all plotting functions
- `translator_toolkit/summary.py`
  - low-n gain and tail-gap summaries across many curves
- `translator_toolkit/presets.py`
  - paper-style nonlinear sweep presets
- `translator_toolkit/utils.py`, `translator_toolkit/theory.py`
  - shared math and smoothing helpers

## Installation

From the package directory:

```bash
pip install -e .
```

## Minimal usage

### 1. Linear figure panels

```python
from translator_toolkit import make_all_linear_panels_and_save

make_all_linear_panels_and_save(outdir="figs", prefix="fig3_")
```

### 2. Nonlinear learning curve

```python
from translator_toolkit import compute_learning_curve_auc_vs_n_nonlinear
from translator_toolkit.plotting import plot_auc_learning_curve_smoothed_with_band
from translator_toolkit.utils import make_dense_n_grid

n_grid = make_dense_n_grid(n_min=30, n_mid=1000, n_max=10000)
curve = compute_learning_curve_auc_vs_n_nonlinear(
    n_label_grid=n_grid,
    repeats=10,
    translator="rff",
)
fig = plot_auc_learning_curve_smoothed_with_band(curve, band_mode="ci95")
```

### 3. Paper-style nonlinear sweep batch

```python
from translator_toolkit import get_paper_nonlinear_sweep_specs
from translator_toolkit.nonlinear import compute_learning_curve_auc_vs_n_nonlinear
from translator_toolkit.summary import summarize_many_curves
from translator_toolkit.utils import make_dense_n_grid

n_grid = make_dense_n_grid(n_min=30, n_mid=3000, n_max=200000, n_low=26, n_high=14)
specs = get_paper_nonlinear_sweep_specs(n_grid)
curves = {spec["key"]: compute_learning_curve_auc_vs_n_nonlinear(**spec["params"]) for spec in specs}
summary_df = summarize_many_curves(curves)
```

## Consolidation choices

The biggest normalization choice is the nonlinear regime.

Your notebook history showed at least three incompatible nonlinear branches:

1. random-feature translator branch
2. ridge-only nonlinear branch
3. patched branches with `label_flip_prob`, `label_temperature`, and changing `sample_world_nonlinear` signatures

This toolkit keeps a **single coherent nonlinear world** with:

- `latent_axis_strength`
- `label_temperature`
- optional translator choice: `translator="ridge"` or `translator="rff"`

That avoids the notebook-state mismatch that caused the `KeyError` and `unexpected keyword argument` failures.

## What I did **not** assume

I did **not** assume every late notebook patch was correct. Where versions conflicted, I chose the more internally consistent implementation rather than trying to preserve every transient branch.

So this is best treated as a **clean base toolkit**, not a guaranteed perfect reproduction of every notebook output.

## Suggested next step

If you want, the next useful refinement is to turn this into a **paper-facing toolkit** with:

- one `paper_figures.py` driver
- one `supplementary_figures.py` driver
- YAML or JSON config files for all published runs
- deterministic seed registry
- cached `.npz` outputs so expensive sweeps do not rerun every time
