# TRACE
## Translator Representation Analysis of Ceilings and Efficiency

TRACE is an open-source diagnostic toolkit for a simple but important question:

**When a translated representation `h(X)` outperforms the original deployable representation `X`, is that a real downstream advantage, or mostly a low-label learnability effect?**

TRACE answers that question by comparing **paired learning curves across label budgets** and summarizing their difference with the **Advantage Representation Curve (ARC)**.

<p align="center">
  <img src="Images/trace_reference_scenarios_summary.png" alt="TRACE reference scenarios summary" width="950" />
</p>

<p align="center">
  <em>Reference output from <code>trace_reference_scenarios.py</code>. Each row shows paired learning curves, the corresponding ARC curve, and a regime-score meter for one synthetic scenario family.</em>
</p>

---

## Start here: `trace_reference_scenarios.py`

The clearest entry point to this repository is the reference scenario script:

```bash
python trace_reference_scenarios.py
```

This script is designed to be the fastest way to understand what TRACE produces and how to read it.

### What it does

1. Generates synthetic datasets from one consistent linear-Gaussian TRACE world.
2. Instantiates one reference scenario for each main regime:
   - `sample_efficiency`
   - `persistent_advantage`
   - `no_advantage`
   - `lossy_translation`
3. Runs TRACE-style learning-curve diagnostics on each scenario.
4. Computes a regime scoreboard for each scenario.
5. Produces full-panel figures and a compact summary grid.

### Dependencies

- `numpy`
- `matplotlib`
- `scikit-learn`

Install them with:

```bash
pip install numpy matplotlib scikit-learn
```

### Outputs

Running the script saves:

```text
figs/sample_efficiency_full_panel.png
figs/persistent_advantage_full_panel.png
figs/no_advantage_full_panel.png
figs/lossy_translation_full_panel.png
figs/trace_reference_scenarios_summary.png
```

It also prints a compact summary for each scenario, including quantities such as:

- low-label ARC
- tail ARC
- final performance gap
- crossover scale, when present
- regime-style classification

---

## How to read TRACE outputs

TRACE is organized around three coordinated views:

### 1. Paired learning curves

These compare downstream performance as a function of labeled sample size:

- `A_X(n)`: predictive performance from the original representation `X`
- `A_H(n)`: predictive performance from the translated representation `h(X)`

The key question is not just which curve is higher at one point, but **how the gap evolves as label budget increases**.

### 2. Advantage Representation Curve (ARC)

TRACE defines

$$
A_X(n) = \mathrm{AUC}_n(X), \qquad
A_H(n) = \mathrm{AUC}_n(h(X)),
$$

and then computes

$$
\mathrm{ARC}(n) = A_H(n) - A_X(n).
$$

ARC is the expected generalization-performance advantage of learning on the translated representation rather than the original one, indexed by label budget `n`.

### 3. Regime meter

The regime meter is a compact heuristic scoreboard over four reference patterns:

- **sample efficiency**
- **persistent advantage**
- **no advantage**
- **lossy translation**

These scores are **similarity scores, not calibrated probabilities**. They help turn curve shape into a structured diagnosis.

---

## The four reference regimes

| Regime | ARC pattern | Interpretation |
|---|---|---|
| **sample efficiency** | positive at low `n`, then shrinks toward zero | translation mainly helps when labels are scarce |
| **persistent advantage** | positive early and still positive at larger `n` | translated representation remains practically useful across the studied range |
| **no advantage** | ARC stays near zero | translation adds little downstream value |
| **lossy translation** | ARC is negative, or turns negative at scale | translation discards or distorts useful signal |

These are reference geometries, not rigid bins. Real studies may show mixed behavior.

---

## Why this matters

Many biomarker and multimodal pipelines now follow the pattern

```text
X  ->  translator  ->  h(X)
```

Examples include:

- H&E slide embeddings -> predicted gene expression
- H&E embeddings -> predicted proteomics
- morphology features -> predicted pathway activity
- pathology foundation-model tokens -> cross-modal biological embeddings

In these settings, it is common to observe that a model trained on `h(X)` beats a model trained on `X` at a fixed benchmark label count.

That single comparison is not enough.

A higher AUC from `h(X)` does **not** by itself tell you whether:

- translation provides a durable practical benefit;
- the gain is mostly a finite-sample learnability effect;
- direct learning on `X` will catch up with more labels;
- or the translated representation is actually lossy in the large-sample regime.

TRACE is built to separate those cases.

---

## Relationship to the paper

TRACE is the companion repository for:

**H&E-to-Molecular Translators as a Computational Primitive for Biomarker Discovery: Learnability Gains Under Conserved Information Ceilings**  
**Payam Saisan, Sandip Pravin Patel**  
**bioRxiv** *(link coming soon)*

The paper studies the distinction between:

- the **deployment-time information ceiling** available from the deployable representation `X`; and
- **finite-sample learnability**, meaning how easily a downstream learner can extract predictive structure from limited labeled data.

Under the paper's framing, if the deployed translator `h(X)` is deterministic after paired-data training, then it can reorganize information already present in `X`, but it cannot create new deployment-time information from outside `X`.

That means a practical gain from `h(X)` should be interpreted as a **learnability gain**, not evidence that the translated representation has created a higher underlying information ceiling.

TRACE is the executable framework for studying exactly that distinction.

---

## Practical use on your own data

TRACE is also a diagnostic workflow for real representation pairs.

A typical applied setup is:

```text
X = deployable features
H = translated features = h(X)
y = downstream labels
```

For example:

```text
X = H&E embeddings
H = predicted molecular features from an H&E-to-RNA translator
y = downstream task label
```

The TRACE workflow is:

1. provide `X`, `H`, and `y`;
2. choose a label-budget grid `n_grid`;
3. fit paired downstream models across repeated subsamples;
4. inspect the learning curves, ARC, and regime summary.

At a high level, the reference script follows exactly this pattern.

---

## Where to start in the repository

If you want the fastest overview, start with one of these:

1. `trace_reference_scenarios.py`  
   Script-based overview of the four canonical reference regimes.

2. `Notebooks/00_overview_and_reference_scenarios.ipynb`  
   Notebook version of the same story, with more room for inspection.

3. `Notebooks/01_sample_efficiency.ipynb`  
   Positive low-label ARC that decays toward zero.

4. `Notebooks/02_persistent_advantage.ipynb`  
   Positive ARC that remains clearly useful later.

5. `Notebooks/03_no_advantage.ipynb`  
   Near-null ARC across the studied range.

6. `Notebooks/04_lossy_translation.ipynb`  
   Harmful or lossy translation regime.

7. `Notebooks/05_custom_scenario_playground.ipynb`  
   User-defined scenarios and experimentation.

---

## Example interpretation

A sample-efficiency pattern means:

- `h(X)` is easier to learn from when labels are scarce;
- the advantage is real in the low-label regime;
- but direct learning on `X` may catch up as more labels arrive.

A persistent-advantage pattern means:

- the translated representation remains practically useful even at the right tail of the studied label range;
- improving the translator may be a worthwhile modeling direction.

A no-advantage pattern means:

- translation is largely unnecessary for the downstream task under the current setup.

A lossy-translation pattern means:

- the translation step is discarding or distorting task-relevant signal;
- direct learning on `X` is likely preferable.

---

## Citation

If you use TRACE in your work, please cite the associated manuscript:

**Saisan, P., & Patel, S. P. (2026).**  
*H\&E-to-Molecular Translators as a Computational Primitive for Biomarker Discovery: Learnability Gains Under Conserved Information Ceilings.*  
bioRxiv.

### BibTeX

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

## License

MIT License
