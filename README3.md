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
