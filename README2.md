# **TRACE** : (**T**ranslation-based **R**epresentation **A**dvantage **C**haracterization and **E**valuation)

A minmalistic diagnostic tool for deciding **when a translated representation (h(X)) is actually useful for prediction compared to the original deployable representation (X).**

TRACE implements the **Advantage Representation Curve (ARC)** introduced in the accompanying manuscript.

The goal is simple:

If a model trained on (h(X)) beats a model trained on (X), TRACE determines **whether that gain is real and persistent vs. a data artifact.**

---

# The problem TRACE solves

In many biomarker pipelines we start with deployable features such as

* H&E slide embeddings
* pathology foundation-model tokens
* morphological features
* imaging representations

and then build a **translator**

```
X  →  h(X)
```

Examples

| X (deployable)      | h(X) (translated)           |
| ------------------- | --------------------------- |
| H&E embeddings      | predicted gene expression   |
| H&E embeddings      | predicted proteomics        |
| H&E embeddings      | predicted immune signatures |
| morphology features | predicted pathway activity  |

Researchers often observe

```
model(h(X)) > model(X)
```

But a single benchmark AUC cannot answer the critical question:

**Is the translated representation actually better, or does it only help when labels are scarce?**

TRACE answers this using **learning curves across label budgets.**

---

# Core idea: the Advantage Representation Curve

Let

```
A_X(n) = AUC(n, X)
A_H(n) = AUC(n, h(X))
```

where

* (X) = deployable representation
* (h(X)) = translated representation
* (n) = number of labeled training samples

TRACE computes

```
ARC(n) = A_H(n) − A_X(n)
```

The **shape of ARC** reveals why translation appears to help.

---

# What TRACE tells you

| ARC pattern                           | Interpretation                         | Action            |
| ------------------------------------- | -------------------------------------- | --------------------------- |
| large ARC at small n, shrinking later | translation improves sample efficiency | collect more labels         |
| ARC remains positive across n         | representation truly helps             | improve translator          |
| ARC ≈ 0                               | translation adds little value          | use original representation |
| curves flatten                        | diminishing returns                    | rethink representation      |

TRACE converts learning curves into **actionable modeling decisions.**

---

# Typical user workflow

You already have

```
X : deployable features
H : translated features h(X)
Y : labels
```

Example:

```
X = H&E embeddings
H = predicted gene expression
Y = mutation status
```

Run TRACE:

```python
import trace

report = trace.run(
    X=X,
    H=H,
    y=Y,
    n_grid=[50,100,200,500,1000],
    model="logistic"
)

report.plot_learning_curves()
report.plot_arc()
report.summary()
```

TRACE outputs

```
learning_curves.png
arc_curve.png
trace_summary.json
trace_report.txt
```

---

# Example interpretation

Learning curves

```
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

```
ARC
0.15 |\
     | \
0.10 |  \
     |   \
0.05 |    \
     +-----\-----------
           label budget
```

Interpretation

```
translation helps when labels are scarce
direct learning on X catches up with more data
```

Decision

```
if labels are scarce → use translation
if labels are plentiful → direct modeling may suffice
```

---

# Minimal toy example

Synthetic scenario:

* latent biological signal
* deployable representation with nuisance variation
* translated representation that suppresses nuisance

```python
import numpy as np
import trace

N = 4000
d = 30

rng = np.random.default_rng(0)

z = rng.normal(size=(N,1))
y = (z[:,0] > 0).astype(int)

nuisance = rng.normal(scale=2.0,size=(N,d))

X = z @ rng.normal(size=(1,d)) + nuisance + rng.normal(size=(N,d))
H = z @ rng.normal(size=(1,8)) + 0.5*rng.normal(size=(N,8))

report = trace.run(X=X,H=H,y=y,n_grid=[50,100,200,500,1000])
```

This toy example typically produces a **large ARC at small label budgets that shrinks as data grows**, illustrating a **finite-sample learnability advantage**.

---

# Installation

```
pip install trace-diagnostics
```

or

```
git clone https://github.com/yourname/TRACE
cd TRACE
pip install -e .
```

---

# Repository structure

```
trace/
    learning_curves.py
    arc.py
    diagnostics.py
    plotting.py
    workflow.py

examples/
    toy_simulation.ipynb
```

---

# What TRACE is (and is not)

TRACE **is**

* a representation-advantage diagnostic
* a learning-curve analysis tool
* a decision aid for choosing between (X) and (h(X))

TRACE **is not**

* a translator training framework
* a causal inference tool
* a theoretical estimator of information limits

Its purpose is practical:

> Given (X) and (h(X)), determine whether using (h(X)) is actually worth it.

---

# Relationship to the paper

The accompanying manuscript introduces the theoretical framework explaining why translated representations can improve finite-sample learnability without increasing deployment-time information.

TRACE is the **executable diagnostic implementation** of that framework.

---

# License

MIT License


