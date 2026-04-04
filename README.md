<img src="Images/Trace1.png" style="border: 0;" />

##

<div align="center">
  <img src="Images/Trace2.png" alt="TRACE overview" />
</div>

##

# TRACE
## Translator Representation Analysis of Ceilings and Efficiency

An open-source diagnostic toolkit for determining when a translated representation \(h(X)\) provides downstream predictive advantage over the original deployable representation \(X\), as a function of label budget.

TRACE implements the **Advantage Representation Curve (ARC)** introduced in the accompanying manuscript.

---

TRACE is the official companion repository for the paper:

**Molecular Translators as a Computational Primitive for Biomarker Discovery: Learnability Gains Under Conserved Information Ceilings**  
**Payam Saisan, Sandip Pravin Patel**  
**bioRxiv** *(link coming soon)*

TRACE serves two related purposes:

- a simulation and analysis framework for studying the paper’s central computational question;
- a practical diagnostic toolkit for comparing direct learning on \(X\) against translated learning on \(h(X)\) across label budgets.

The central question is:

> If a model trained on \(h(X)\) outperforms a model trained on \(X\), is that because translation truly helps downstream learning, or because it only provides a temporary advantage in the low-label regime?

TRACE addresses this by analyzing **paired learning curves across label budgets**, rather than relying on a single benchmark AUC.

---

# Why TRACE is here now?

Timing of TRACE's development is tied to an emerging innovation in molecular data analytics. Translated molecular intermediates like MISO and GigaTime are emerging as a potentially game changing **computational primitive** for biomarker modeling. TRACE is built for studying this setting: when a downstream target \(Y\) may be better predicted indirectly through a translated representation \(h(X)\) derived from \(X\) vs. directly from it's original deployable representation \(X\). 

In this view, a biomarker pipeline is organized as
 
Many biomarker pipelines now follow the pattern

```text
deployable representation X  ->  translator  ->  translated representation h(X)
```

Examples include:

| Deployable representation \(X\) | Translated representation \(h(X)\) |
|---|---|
| H&E slide embeddings | predicted gene expression |
| H&E embeddings | predicted proteomics |
| H&E embeddings | predicted immune signatures |
| morphology features | predicted pathway activity |
| pathology foundation-model tokens | cross-modal biological embeddings |

Researchers often observe

$$
\mathrm{model}(h(X)) > \mathrm{model}(X),
$$

but that single comparison is not enough.

A higher AUC from \(h(X)\) does **not** by itself tell you whether:

- translation provides a genuine practical advantage;
- the gain is mainly a **finite-sample learnability effect**;
- direct prediction from \(X\) will catch up once more labels are available;
- or the apparent gain reflects confounding, shortcut structure, or evaluation artifacts.

TRACE is built to distinguish those cases.

---

# Core idea: advantage as a function of label budget

TRACE treats predictive performance as a function of training size.

$$
A_X(n) \equiv \mathrm{AUC}_n(X), \qquad
A_H(n) \equiv \mathrm{AUC}_n(h(X)),
$$

where \(X\) is the original deployable representation, \(h(X)\) is the translated representation, and \(n\) is the number of labeled training samples.

TRACE computes the **Advantage Representation Curve (ARC)**

$$
\mathrm{ARC}(n) = A_H(n) - A_X(n).
$$

ARC is the expected generalization performance gap, indexed by label budget, between models trained on \(h(X)\) and models trained on \(X\).

The key point is that the **shape of ARC across \(n\)** is often more informative than any single endpoint metric.

---

# What TRACE tells you

TRACE uses four canonical ARC geometries as a compact diagnostic vocabulary:

| ARC pattern | Interpretation | Suggested action |
|---|---|---|
| positive at small \(n\), decaying toward zero | translation mainly improves sample efficiency | collect more labels if feasible |
| positive across the studied range | translated representation retains practical value | improve or exploit the translated representation |
| near zero throughout | translation adds little downstream value | prefer direct learning on \(X\) |
| negative throughout, or sign-reversing unfavorably | translation is lossy or distorting | avoid translated representation for this task |

These are empirical reference regimes, not rigid bins. Real studies may lie between them.

TRACE turns learning curves into **decision-oriented summaries**.

---

# Relationship to the paper

The accompanying paper studies a specific distinction:

- **deployment-time information ceiling**: the Bayes-optimal predictive limit available from the deployable representation \(X\);
- **finite-sample learnability**: how easily a downstream learner can extract useful predictive structure from limited labeled data.

If the translator \(\hat Z = h(X)\) is deterministic after paired-data training, then it cannot create new deployment-time information. Accordingly,

```math
I(Y;\hat Z) \le I(Y;X),
```

and, under the paper’s framing,


```math
\mathrm{AUC}^{*}(\hat{Z}) \le \mathrm{AUC}^{*}(X)
```


So whenever prediction from \(\hat Z\) exceeds prediction from \(X\) in practice, that gain must be interpreted as a **learnability effect**, not as creation of new deployment-time signal.

TRACE is the executable framework for studying exactly that distinction.

---

# Two uses of this repository

## 1. Paper companion / simulation framework

TRACE contains the code used to generate the paper’s synthetic experiments, figures, and summary analyses.

These experiments isolate the mechanism proposed in the manuscript:

> translator-derived representations can improve downstream prediction in label-limited settings by reorganizing existing information into a more learnable form, even when deterministic translation does not increase the underlying deployment-time ceiling.

## 2. Practical representation diagnostic

TRACE can also be used for studies that already have:

```text
X : deployable features
H : translated features = h(X)
Y : downstream labels
```

and want to answer:

> Should I train on \(X\) or on \(h(X)\)?

---

# Getting started

Clone the repository:

```bash
git clone https://github.com/psaisan/TRACE
cd TRACE
```

If the repository includes packaging metadata for editable installation, install locally with:

```bash
pip install -e .
```

If not, launch notebooks or scripts from the repository root so that local imports resolve correctly.

## Important naming note

Python already includes a standard-library module named `trace`. Because this repository also uses `trace/` as a package name, import behavior can depend on how Python is launched and what is on the path.

The most reliable ways to work with this repository are:

- run notebooks from the repository root;
- run scripts from the repository root;
- treat the notebooks in `Notebooks/` as the primary runnable examples.

If you encounter an import collision with the Python standard-library `trace` module, first check that you are launching from the project root and that the local package has been installed correctly if editable installation is available.

---

# Where to start

The recommended entry point is the notebook sequence in `Notebooks/`.

## Suggested order

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

The `Examples/` directory contains additional example notebooks and scripts, including lightweight smoke tests and paper-style sweep runs.

---

# Typical practical workflow

A common applied setting is:

```text
X = H&E embeddings
H = translated features, such as predicted gene expression from an H&E->RNA translator
Y = downstream task label, such as mutation status or response class
```

TRACE then estimates:

- paired learning curves from \(X\) and \(H\);
- the Advantage Representation Curve (ARC);
- compact summaries of low-label, late-label, and crossover behavior;
- regime-oriented interpretation.

At a high level, the analysis path is:

1. provide `X`, `H`, and `y`;
2. specify a label-budget grid `n_grid`;
3. fit and evaluate paired downstream models across repeated subsamples;
4. inspect:
   - learning curves,
   - ARC,
   - scalar summaries,
   - regime-like behavior.

Because the exposed programmatic interface may evolve, the notebooks in `Notebooks/` should be treated as the primary runnable examples for this repository.

---

# Example interpretation

A typical sample-efficiency pattern looks like this:

```text
Learning curves

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

```text
ARC

ARC
0.15 |\
     | \
0.10 |  \
     |   \https://github.com/psaisan/TRACE/tree/main
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
if labels are scarce -> use translation
if labels are plentiful -> direct modeling on X may suffice
```

---

# Example outputs

TRACE is organized around three coordinated outputs:

1. **paired learning curves**  
   \(A_X(n)\) and \(A_H(n)\), typically with uncertainty bands;

2. **ARC curve**  
   \(\mathrm{ARC}(n)=A_H(n)-A_X(n)\);

3. **compact summaries / regime scores**  
   low-label advantage, late-regime behavior, and approximate crossover scale.

These outputs support interpretations such as:

- translation helps mainly when labels are scarce;
- direct learning on \(X\) catches up with increasing label budget;
- translated learning remains advantageous across scale;
- translation is effectively neutral;
- translation is lossy or misleading.

---

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

---

# License

MIT License
