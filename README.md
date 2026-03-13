
<img src="Images/trace1.PNG"  style="border: 0;"/>

# TRACE
**Translator Representation Analysis of Ceilings and Efficiency**

TRACE is a simulation and analysis toolkit for studying when translator-derived proxy representations improve finite-sample biomarker prediction under conserved information ceilings.



For more information, see our paper:

**H&E-to-Molecular Translators as a Computational Primitive for Biomarker Discovery: Learnability Gains Under Conserved Information Ceilings**  
Payam Saisan, Sandip Pravin Patel  
**bioRxiv** *(link coming soon)*

This repository contains the code used to generate the simulation results, figures, and supporting analyses described in the paper.

---


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



