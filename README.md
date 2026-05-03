# TRACE: Translator Representation Analysis of Ceilings and Efficiency

**Preprint:** Saisan & Patel (2026). *Molecular Translators as a Computational
Primitive for Biomarker Discovery: Learnability Gains Under Conserved Information
Ceilings.* bioRxiv. https://doi.org/10.64898/2026.04.27.720188

If you use TRACE in your work, please cite the preprint above — BibTeX at the bottom.

---

<img src="Images/Trace1.png" style="border: 0;" />

---

High-fidelity molecular translator systems — tools that transform routine H&E slides
into virtual molecular maps — are emerging as powerful primitives for biomarker
modeling and discovery, particularly as a way to engage the field's recurring
prediction plateau.

Their apparent success, however, invites a consequential misconception. As the line
between virtual and measured molecular maps begins to blur, virtual fidelity drifts
toward an intuitive assumption of newly recovered molecular information.

**That assumption is wrong.**

A molecular translator is a deterministic map of morphology. However faithful, it
cannot introduce new slide-specific information at inference. The H&E deployment
ceiling is conserved.

That is not a negative result.

Translators still carry a potentially transformative advantage — as force multipliers
on biomarker prediction learning. But translator-driven learnability gains arise in
the *absence* of new information, producing an **information–performance paradox**
whose resolution has real consequences for how the field moves forward and invests
its resources.

Without formal scaffolding, these gains will be routinely conflated with added
information — misdirecting developmental effort in precision medicine where
molecularly actionable decisions depend on predictors learned from pathology. The
field risks exhausting resources against a ceiling it cannot see.

---

The underlying problem is not specific to pathology. Wherever a computational
surrogate stands in for a richer or more costly measurement at deployment — virtual
staining, spatial transcriptomics imputation, single-cell modality transfer, remote
sensing proxies — the same paradox applies: apparent performance gains arising not
from new information but from better-organized existing information. TRACE was
designed to be agnostic to domain; the ceiling–gap decomposition and ARC diagnostic
apply to any paired-data-trained proxy deployed in place of a direct measurement.

---

## Why TRACE is here now?

Timing of TRACE's development is tied to an emerging innovation in molecular data
analytics. Translated molecular intermediates are emerging as a potentially
game-changing **computational primitive** for biomarker modeling and discovery.
TRACE is built for studying this setting: when a downstream target $Y$ may be
better predicted indirectly through a translated representation $h(X)$ derived
from $X$, rather than directly from its original deployable representation $X$.

Many biomarker pipelines now follow the pattern:

```text
deployable representation X  →  translator  →  translated representation h(X)
```

| Deployable representation $X$ | Translated representation $h(X)$ |
|---|---|
| H&E slide embeddings | predicted gene expression |
| H&E embeddings | predicted proteomics |
| H&E embeddings | predicted immune signatures |
| morphology features | predicted pathway activity |
| pathology foundation-model tokens | cross-modal biological embeddings |

The systems that anchor TRACE's empirical validation are
[MISO](https://www.nature.com/articles/s41467-025-66691-y) (*Nat. Commun.* 2025),
which translates H&E to spatial gene expression, and
[GigaTIME](https://www.cell.com/cell/fulltext/S0092-8674(25)01312-1) (*Cell* 2025),
which translates H&E to virtual multiplex immunofluorescence. GigaTIME alone was
applied to 14,256 patients across 51 hospitals — a scale of virtual molecular
mapping previously infeasible without translation. A [Microsoft Research blog
post](https://www.microsoft.com/en-us/research/blog/gigatime-scaling-tumor-microenvironment-modeling-using-virtual-population-generated-by-multimodal-ai/)
and [interactive demo](https://labs.ai.azure.com/projects/gigatime/) provide
accessible entry points to GigaTIME.

These are exactly the systems whose gains TRACE is designed to interpret correctly.

---

## What TRACE Does

TRACE provides the mathematical framework and computational tools to resolve this
paradox operationally.



**Ceiling–gap decomposition** formally separates the Bayes-optimal information
ceiling (modality-limited, irreducible) from finite-sample method gaps (recoverable
through better supervision, more labels, or improved architectures). Because a
deployed translator is a deterministic function of morphology, the data processing
inequality gives

```math
I(Y;\hat{Z}) \leq I(Y;X)
```

and accordingly

```math
\mathrm{AUC}^{*}(\hat{Z}) \leq \mathrm{AUC}^{*}(X)
```

Any practical gain from h(X) over X must therefore be interpreted as a learnability
effect, not as recovery of new deployment-time signal.


**Falsifiable signatures** distinguish method-limited from modality-limited regimes,
validated in controlled analytical experiments anchored to MISO and GigaTIME.




**The Advantage Representation Curve (ARC)** is the toolkit's primary diagnostic.

TRACE computes paired learning curves

```math
A_X(n) \equiv \mathrm{AUC}_n(X), \qquad A_H(n) \equiv \mathrm{AUC}_n(h(X))
```

and their difference

```math
\mathrm{ARC}(n) = A_H(n) - A_X(n)
```

indexed by label budget $n$. The shape of ARC across $n$ is typically far more
informative than any single-endpoint benchmark.

**Decision support** — TRACE turns ARC geometry into actionable interpretation:

| ARC pattern | Interpretation | Suggested action |
|---|---|---|
| Positive at small $n$, decaying toward zero | Translation improves sample efficiency | Collect more labels if feasible |
| Positive across the studied range | Translated representation retains practical value | Improve or exploit the translator |
| Near zero throughout | Translation adds little downstream value | Prefer direct learning on $X$ |
| Negative or unfavorably sign-reversing | Translation is lossy or distorting | Avoid translated representation for this task |

These are empirical reference regimes, not rigid bins. Real studies may fall between
them.

---

## Installation

```bash
git clone https://github.com/psaisan/TRACE
cd TRACE
pip install -e .
```

**Note:** Python's standard library includes a module named `trace`. Run notebooks
and scripts from the repository root to avoid import collisions.

---

## Getting Started

The recommended entry point is the notebook sequence in `Notebooks/`:

1. `00_overview_and_reference_scenarios.ipynb` — TRACE outputs and reference regimes
2. `01_sample_efficiency.ipynb` — positive low-label ARC, decaying toward zero
3. `02_persistent_advantage.ipynb` — sustained positive ARC across label budgets
4. `03_no_advantage.ipynb` — near-null ARC
5. `04_lossy_translation.ipynb` — negative ARC / failure regime
6. `05_custom_scenario_playground.ipynb` — user-defined scenarios

For paper-style synthetic outputs, start with notebook `00`.

---

## Practical Workflow

A typical applied setting:

```text
X = H&E embeddings
H = translated features (e.g. predicted gene expression from an H&E→RNA translator)
Y = downstream label (e.g. mutation status, response class)
```

Provide `X`, `H`, and `y`; specify a label-budget grid `n_grid`; fit and evaluate
paired downstream models across repeated subsamples. TRACE returns three coordinated
outputs for each run: paired learning curves with uncertainty bands, the ARC curve,
and a regime-score panel quantifying similarity to the four canonical regimes.

<div align="center">
  <img src="Images/Trace2.png" alt="TRACE reference outputs across four canonical ARC regimes" width="90%" />
</div>

*Figure 11 from the accompanying manuscript. Each row shows paired learning curves
(left), the ARC (middle), and regime scores (right) for the four canonical regimes:
quick-gain, sustained-gain, neutral, and impaired.*

For a fully executable demonstration reproducing all Figure 11 panels with a single
command. If you are reviewer request `trace_reviewer_demo.py` - to be included in the repository root after manuscript review completion.

---

## Citation

```bibtex
@article{saisan2026moleculartranslators,
  author  = {Saisan, Payam A. and Patel, Sandip Pravin},
  title   = {Molecular Translators as a Computational Primitive for Biomarker
             Discovery: Learnability Gains Under Conserved Information Ceilings},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.04.27.720188},
  url     = {https://doi.org/10.64898/2026.04.27.720188}
}
```

---

## License

MIT
