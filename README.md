
<img src="Images/Trace1.png" style="border: 0;" />

##

<div align="center">
  <img src="Images/Trace2.png" alt="TRACE overview" />
</div>

##




# **TRACE** : (**T**ranslation-based **R**epresentation **A**dvantage **C**haracterization and **E**valuation)

A minmalistic diagnostic tool for deciding **when a translated representation (h(X)) is actually useful for prediction compared to the original deployable representation (X).**

TRACE implements the **Advantage Representation Curve (ARC)** introduced in the accompanying manuscript.

---

TRACE is the **official companion repository** for the paper:

**H&E-to-Molecular Translators as a Computational Primitive for Biomarker Discovery: Learnability Gains Under Conserved Information Ceilings**  
**Payam Saisan, Sandip Pravin Patel**  
**bioRxiv** *(link coming soon)*

TRACE is both:

- a **simulation and analysis framework** for studying the paper’s central computational question, and
- a **practical diagnostic tool** for deciding when a translated representation \(h(X)\) is actually useful for downstream prediction compared to the original deployable representation \(X\).

The central question is:

> If a model trained on \(h(X)\) outperforms a model trained on \(X\), is that because translation truly helps downstream learning, or because it only provides a temporary advantage in the low-label regime?

TRACE answers this by analyzing **learning curves across label budgets**, rather than relying on a single benchmark AUC.

---

# Why TRACE exists

Many biomarker pipelines now follow the pattern

deployable representation  X   →   translator   →   translated representation  h(X)


Examples include:

| Deployable representation (X)     | Translated representation (h(X))  |
| --------------------------------- | --------------------------------- |
| H&E slide embeddings              | predicted gene expression         |
| H&E embeddings                    | predicted proteomics              |
| H&E embeddings                    | predicted immune signatures       |
| morphology features               | predicted pathway activity        |
| pathology foundation-model tokens | cross-modal biological embeddings |

Researchers often observe:

$
model(h(X)) > model(X)
$

But that single comparison is not enough.

A higher AUC from (h(X)) does **not** by itself tell you whether:

* translation provides a genuine practical advantage,
* the gain is mainly a **finite-sample learnability effect**,
* direct prediction from (X) will catch up once more labels are available,
* or the apparent gain reflects confounding, shortcut structure, or evaluation artifacts.

TRACE is built to distinguish those regimes.

---

# Core idea: advantage as a function of label budget

TRACE treats performance as a function of training size.

[
A_X(n) = \mathrm{AUC}(n, X)
]

[
A_H(n) = \mathrm{AUC}(n, h(X))
]

where:

* (X) is the original deployable representation
* (h(X)) is the translated representation
* (n) is the number of labeled training samples

TRACE computes the **Advantage Representation Curve (ARC)**

[
\mathrm{ARC}(n) = A_H(n) - A_X(n)
]

The key point is that the **shape of ARC across (n)** is often more informative than any single endpoint metric.

---

# What TRACE tells you

| ARC pattern                             | Interpretation                                                      | Suggested action                 |
| --------------------------------------- | ------------------------------------------------------------------- | -------------------------------- |
| large ARC at small (n), shrinking later | translation improves sample efficiency                              | collect more labels              |
| ARC remains positive across (n)         | translated representation provides a persistent practical advantage | improve translator               |
| ARC (\approx 0)                         | translation adds little value                                       | use original representation      |
| curves flatten while ARC shrinks        | diminishing returns                                                 | rethink representation or target |

TRACE turns learning curves into **decision-oriented summaries**.

---

# Relationship to the paper

The paper studies a specific distinction:

* **deployment-time information ceiling**: the Bayes-optimal predictive limit available from the deployable representation (X)
* **finite-sample learnability**: how easily a downstream learner can extract useful prediction from limited labeled data

If the translator (\hat Z = h(X)) is deterministic after paired-data training, then it cannot create new deployment-time information. Accordingly,

[
I(Y;\hat Z) \le I(Y;X)
]

and, under the paper’s framing,

[
\mathrm{AUC}^*(\hat Z) \le \mathrm{AUC}^*(X)
]

So whenever prediction from (\hat Z) exceeds prediction from (X) in practice, that gain must be interpreted as a **learnability effect**, not as creation of new deployment-time signal.

TRACE is the executable framework for studying exactly that effect.

---

# Two uses of this repository

## 1. Paper companion / simulation framework

TRACE contains the code used to generate the paper’s synthetic experiments, figures, and summary analyses.

These experiments isolate the mechanism proposed in the manuscript:

> translator-derived representations can improve downstream prediction in label-limited settings by reorganizing existing information into a more learnable form, even when deterministic translation does not increase the underlying deployment-time ceiling.

## 2. Practical representation diagnostic

TRACE can also be used as a simple workflow for users who already have:

```text
X : deployable features
H : translated features = h(X)
Y : downstream labels
```

and want to answer:

> Should I train on (X) or on (h(X))?

---

# Typical user workflow

A common practical setting is:

```text
X = H&E embeddings
H = predicted gene expression from an H&E→RNA translator
Y = mutation status
```

Run TRACE:

```python
import trace

report = trace.run(
    X=X,
    H=H,
    y=Y,
    n_grid=[50, 100, 200, 500, 1000],
    model="logistic"
)

report.plot_learning_curves()
report.plot_arc()
report.summary()
```

TRACE produces outputs such as:

```text
learning_curves.png
arc_curve.png
trace_summary.json
trace_report.txt
```

These outputs help determine whether translation is useful mainly when labels are scarce, remains useful across scales, or adds little practical value.

---

# Example interpretation

Learning curves

```text
AUC
0.9 |        H
    |       /
0.8 |      /
    |     /
0.7 |    /
    |   /
0.6 |  / X
    +----------------
       label budget
```

ARC

```text
ARC
0.15 |\
     | \
0.10 |  \
     |   \
0.05 |    \
     +-----\-----------
           label budget
```

Interpretation:

```text
translation helps when labels are scarce
direct learning on X catches up with more data
```

Decision:

```text
if labels are scarce → use translation
if labels are plentiful → direct modeling on X may suffice
```

---

# Minimal toy example

This synthetic example mirrors the practical use case:

* a latent biological signal drives the label
* the deployable representation contains signal plus nuisance
* the translated representation suppresses some nuisance and is easier to learn from

```python
import numpy as np
import trace

N = 4000
d = 30

rng = np.random.default_rng(0)

# latent biological signal
z = rng.normal(size=(N, 1))

# binary downstream label
y = (z[:, 0] > 0).astype(int)

# nuisance variation in the deployable representation
nuisance = rng.normal(scale=2.0, size=(N, d))

# deployable representation X
X = z @ rng.normal(size=(1, d)) + nuisance + rng.normal(size=(N, d))

# translated representation H = h(X)
H = z @ rng.normal(size=(1, 8)) + 0.5 * rng.normal(size=(N, 8))

report = trace.run(
    X=X,
    H=H,
    y=y,
    n_grid=[50, 100, 200, 500, 1000]
)
```

This toy setup typically yields a **positive low-label ARC that shrinks as sample size grows**, illustrating the paper’s core regime: **finite-sample learnability gains under conserved information ceilings**.

---

# Simulation framework in the paper

The repository’s full simulation pipeline follows the same logic as the manuscript.

## 1. Sanity checks for deterministic translation

The first stage verifies that a frozen deterministic translator obeys the data-processing inequality and does not raise the Bayes-optimal deployment ceiling.

## 2. Linear-Gaussian benchmark

The second stage studies an analytically controlled setting with:

* low-dimensional biological signal
* structured nuisance
* Gaussian observation noise
* paired-data translator learning
* downstream comparison between raw (X) and translated (\hat Z)

This benchmark supports DPI checks, ceiling comparisons, learning curves, fidelity sweeps, and nuisance-dependent phase behavior.

## 3. Nonlinear latent-world benchmark

The third stage moves beyond the linear regime:

* labels arise from nonlinear latent biology
* the paired modality remains relatively aligned with task-relevant signal
* the deployable representation is higher-dimensional, nuisance-entangled, and more distorted

This creates the important regime in which (\hat Z) may be very useful at low label budgets while still being slightly lossy at large sample sizes.

## 4. Robustness sweeps and scalar summaries

The fourth stage evaluates robustness under structured nonlinear sweep families, including:

* translator lossiness
* nuisance entanglement in (X)
* paired-sample availability

and summarizes resulting learning curves using metrics such as:

* low-label gain
* crossover sample size
* tail-gap estimate

---

# Threat models and failure modes

Apparent gains from translated intermediates do not automatically imply biology-aligned or deployment-relevant signal. TRACE therefore emphasizes explicit shortcut and failure-mode analysis.

Main concerns include:

* **translator confounding**: translated features encode site, stain, scanner, or acquisition artifacts
* **prior hallucination**: the translator produces plausible but misleading proxy structure
* **evaluation circularity**: downstream gains are inflated by test leakage or hyperparameter selection on evaluation data

The paper and code therefore emphasize:

* site-stratified evaluation
* frozen translator selection
* fidelity degradation tests
* separation of translator training from downstream label evaluation
* calibration against measured modalities when available

---

# Repository structure

```text
trace/
    learning_curves.py
    arc.py
    diagnostics.py
    plotting.py
    workflow.py

examples/
    toy_simulation.ipynb

paper/
    figure_generation/
    sweep_scripts/
    analysis/

Images/
    Trace1.png
    Trace2.png

README.md
```

---

# Getting started

Install locally:

```bash
git clone https://github.com/psaisan/TRACE
cd TRACE
pip install -e .
```

Then run a basic example:

```python
import trace

report = trace.run(
    X=X,
    H=H,
    y=y,
    n_grid=[50, 100, 200, 500, 1000]
)
```

For the full paper simulations and figure-generation workflow, see the scripts in the repository.

---

# Citation

If you use TRACE in your work, please cite the associated paper:

**Saisan, P., & Patel, S. P. (2026).**
*H&E-to-Molecular Translators as a Computational Primitive for Biomarker Discovery: Learnability Gains Under Conserved Information Ceilings.*
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

---

# License

MIT License

