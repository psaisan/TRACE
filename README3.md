<img src="Images/Trace1.png" style="border: 0;" />

##

<div align="center">
  <img src="Images/Trace2.png" alt="TRACE overview" />
</div>

##

# TRACE  
## Translator Representation Analysis of Ceilings and Efficiency

TRACE is an open-source diagnostic toolkit for comparing downstream prediction from an original deployable representation \(X\) against prediction from a translated representation \(h(X)\), across label budgets rather than at a single operating point.

The central practical question is:

> When a translated representation \(h(X)\) outperforms direct learning on \(X\), is that advantage persistent, or is it mainly a low-label learnability effect?

TRACE answers this by estimating paired learning curves and their difference across training-set size.

---

# What this repository contains

This repository has two tightly related uses:

1. **Paper companion**  
   Code, notebooks, and simulation workflows used to generate the synthetic scenarios and figures associated with the accompanying manuscript.

2. **Practical diagnostic workflow**  
   A reusable analysis setup for studies that already have:
   - a deployable representation \(X\),
   - a translated representation \(H = h(X)\),
   - a downstream label \(Y\),

   and want to compare prediction from \(X\) versus \(H\) as a function of label budget.

---

# Core diagnostic: the Advantage Representation Curve (ARC)

TRACE works with paired learning curves

$$
A_X(n) \equiv \mathrm{AUC}_n(X), \qquad
A_H(n) \equiv \mathrm{AUC}_n(h(X)),
$$

where \(n\) is the number of labeled training samples, \(A_X(n)\) is the expected test AUROC from models trained on \(X\), and \(A_H(n)\) is the expected test AUROC from models trained on \(h(X)\).

The primary diagnostic quantity is the **Advantage Representation Curve (ARC)**

$$
\mathrm{ARC}(n) \equiv A_H(n) - A_X(n).
$$

ARC is the expected generalization performance gap, indexed by label budget, between models trained on \(h(X)\) and models trained on \(X\).

Its shape helps answer three practical questions:

- where translated learning helps most,
- whether direct learning from \(X\) catches up,
- whether translation becomes negligible or harmful.

A positive ARC at small \(n\) does **not** mean translation created new deployment-time information. In the paper’s setting, deterministic deployed translation cannot raise the deployment ceiling; a positive ARC instead indicates a finite-sample learnability advantage that outweighs any ceiling loss.

---

# Canonical ARC regimes

TRACE uses four canonical ARC geometries as a compact diagnostic vocabulary:

| Regime | Typical ARC behavior | Interpretation |
|---|---|---|
| **sample efficiency** | positive at small \(n\), then decays toward zero | translation mainly helps when labels are scarce |
| **persistent advantage** | remains positive across the studied range | translated representation retains practical value across scales |
| **no advantage** | remains near zero | translation adds little downstream value |
| **lossy translation** | negative throughout, or reverses sign unfavorably | translation discards useful information or distorts task structure |

These are empirical reference regimes, not rigid bins. Real datasets may lie between them.

---

# Why this matters

Many biomarker pipelines now take the form

```text
deployable representation X  ->  translator  ->  translated representation h(X)


# Installation

Clone the repository:

```bash
git clone https://github.com/psaisan/TRACE
cd TRACE
```

Create and activate a Python environment, then install the repository in editable mode:

```bash
pip install -e .
```

This is the most convenient setup for running the notebooks, examples, and local development code from the repository root.

## Important naming note

Python already includes a standard-library module named `trace`. Because this repository also uses `trace/` as a package name, import behavior can depend on how Python is launched and what is on the path.

The most reliable ways to work with this repository are:

- run notebooks from the repository root;
- run scripts from the repository root after local installation;
- prefer the notebook examples in `Notebooks/` as the authoritative entry point.

If you encounter an import collision with the Python standard-library `trace` module, first check that you are launching from the project root and that the local package has been installed with `pip install -e .`.

# Where to start

The recommended entry point is the notebook sequence in `Notebooks/`.

## Recommended order

1. `Notebooks/00_overview_and_reference_scenarios.ipynb`  
   Overview of TRACE outputs and reference synthetic scenarios.

2. `Notebooks/01_sample_efficiency.ipynb`  
   Positive low-label ARC that decays toward zero.

3. `Notebooks/02_persistent_advantage.ipynb`  
   Sustained positive ARC across label budgets.

4. `Notebooks/03_no_advantage.ipynb`  
   Near-null ARC.

5. `Notebooks/04_lossy_translation.ipynb`  
   Negative ARC / failure regime.

6. `Notebooks/05_custom_scenario_playground.ipynb`  
   User-defined custom scenarios.

For paper-style synthetic outputs and the clearest overview, start with notebook `00`.

# Typical practical workflow

A common applied setting is:

```text
X = deployable features
H = translated features = h(X)
Y = downstream labels
```

TRACE then estimates:

- paired learning curves from \(X\) and \(H\),
- the Advantage Representation Curve (ARC),
- compact summaries of low-label, late-label, and crossover behavior,
- regime-oriented interpretation.

At a high level, the workflow is:

1. provide `X`, `H`, and `y`,
2. specify a label-budget grid `n_grid`,
3. fit and evaluate paired downstream models across repeated subsamples,
4. inspect:
   - learning curves,
   - ARC,
   - scalar summaries,
   - regime-like behavior.

Because the exposed API may evolve, the notebooks in `Notebooks/` should be treated as the primary runnable examples for this repository.

# Example outputs

TRACE is organized around three coordinated outputs:

1. **paired learning curves**  
   \(A_X(n)\) and \(A_H(n)\), typically with uncertainty bands,

2. **ARC curve**  
   \(\mathrm{ARC}(n)=A_H(n)-A_X(n)\),

3. **compact summaries / regime scores**  
   low-label advantage, late-regime behavior, and approximate crossover scale.

These outputs support interpretations such as:

- translation helps mainly when labels are scarce,
- direct learning on \(X\) catches up with increasing label budget,
- translated learning remains advantageous across scale,
- translation is effectively neutral,
- translation is lossy or misleading.

# Synthetic scenarios in this repository

The repository includes controlled reference scenarios illustrating the main ARC geometries:

- **sample efficiency**
- **persistent advantage**
- **no advantage**
- **lossy translation**

These serve both as paper companions and as diagnostic templates for interpreting real studies.

# Practical cautions

A translated representation can appear useful for the wrong reasons. TRACE should therefore be interpreted together with careful evaluation design.

Important concerns include:

- **translator confounding**: translated features may encode site, stain, scanner, or acquisition artifacts;
- **prior hallucination**: the translator may produce plausible but task-misaligned proxy structure;
- **evaluation circularity**: downstream gains may be inflated by leakage or model selection on evaluation data.

TRACE is therefore best used alongside:

- strict train/validation/test separation,
- frozen translator evaluation,
- site- or batch-aware splits where relevant,
- calibration against measured modalities when available.

# Citation

If you use TRACE in your work, please cite the associated manuscript:

**Saisan, P., & Patel, S. P. (2026).**  
*H\&E-to-Molecular Translators as a Computational Primitive for Biomarker Discovery: Learnability Gains Under Conserved Information Ceilings.*  
bioRxiv.

## BibTeX

```bibtex
@article{saisan2026trace,
  author  = {Saisan, Payam and Patel, Sandip Pravin},
  title   = {H\&E-to-Molecular Translators as a Computational Primitive for Biomarker Discovery: Learnability Gains Under Conserved Information Ceilings},
  journal = {bioRxiv},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/psaisan/TRACE}
}
```

# License

MIT License
