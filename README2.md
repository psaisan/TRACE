TRACE

Toolkit for Representation Advantage Diagnostics via ARC

TRACE is a lightweight diagnostic toolkit for analyzing learning efficiency in representation-based biomarker prediction systems.

It implements the Advantage Representation Curve (ARC) introduced in the accompanying manuscript and provides practical diagnostics for evaluating whether a translated representation $h(X)$ provides meaningful advantages over a deployable representation $X$.

Rather than evaluating models at a single training size, TRACE analyzes learning curves as functions of label budget, allowing practitioners to distinguish between:

representation-driven gains

sample-efficiency improvements

diminishing returns from additional labels



---

Conceptual Overview

In most biomarker studies, predictive performance is summarized by a single endpoint metric such as ROC–AUC.

TRACE instead treats predictive performance as a function of labeled training size.

Let

A_X(n) = AUC(n, X)

A_H(n) = AUC(n, h(X))

where

$X$ is a deployable representation

$h(X)$ is a translated representation

$n$ is the number of labeled training samples


The central observable is the Advantage Representation Curve (ARC)

ARC(n) = A_H(n) - A_X(n)

ARC measures the downstream predictive advantage of the translated representation as a function of label budget.

The ARC corresponds to the vertical separation between learning curves for models trained on $X$ and $h(X)$.


---

What TRACE Does

TRACE estimates learning curves and computes ARC-based diagnostics that help determine whether gains from a translated representation are due to:

improved statistical efficiency

improved representation geometry

or simply additional labels


TRACE converts empirical learning curves into decision-oriented summaries.


---

Inputs

TRACE operates on representations, not raw imaging data.

A typical workflow begins after feature extraction or representation learning.


---

Deployable representation $X$

A feature matrix derived from deployable data.

Examples include

H&E image embeddings

histology patch features

pathology foundation-model tokens

morphological features

spatial transcriptomics embeddings

any deployable model input representation


Shape

X : (N_samples × D_features)


---

Translated representation $h(X)$

A representation obtained from a translator model applied to $X$.

Examples

predicted gene expression from H&E

predicted proteomic profiles

cross-modal embeddings

predicted molecular states


Shape

H = h(X) : (N_samples × P_features)


---

Target labels $Y$

A supervised prediction target.

Examples

mutation status

disease subtype

response class

survival bin

clinical phenotype


Shape

Y : (N_samples,)


---

Label-budget grid

A set of training sizes used to estimate learning curves.

Example

n_grid = [50, 100, 200, 500, 1000, 2000]


---

TRACE Workflow

The TRACE pipeline proceeds in four stages.


---

1. Estimate Learning Curves

TRACE trains downstream predictive models using increasing label budgets.

For each training size $n$

sample $n$ labeled examples

train model on representation $X$

train model on representation $h(X)$

evaluate AUC on held-out test data


This produces the learning curves

A_X(n)

A_H(n)


---

2. Compute ARC

TRACE computes the Advantage Representation Curve

ARC(n) = A_H(n) - A_X(n)

Interpretation

ARC(n) Meaning

> 0 translation helps
≈ 0 little difference
< 0 translation hurts


But the shape of ARC across n contains the real information.


---

3. Extract ARC Diagnostics

TRACE summarizes the ARC using several compact metrics.

Low-label advantage

Average ARC over small label budgets.

ARC_low

Measures whether translation improves sample efficiency.


---

Crossover estimate

Approximate label count where the advantage disappears.

n_cross

Indicates when direct learning from $X$ may catch up.


---

Integrated ARC

Total advantage across the practical label range.

ARC_int

Measures the overall value of translation.


---

Plateau heuristic

A heuristic based on learning-curve flattening.

Identifies regimes where additional labels may yield diminishing returns.


---

Decision Interpretation

ARC pattern Interpretation Suggested action

Large early ARC, early crossover translation improves sample efficiency collect more labels
Persistent positive ARC representation quality matters improve translator
ARC ≈ 0 little benefit from translation focus on direct modeling
Curves flatten while ARC shrinks diminishing returns consider alternative representations



---

Minimal Example

import trace

report = trace.run(
    X=X_features,
    H=translated_features,
    y=labels,
    n_grid=[50,100,200,500,1000],
    model="logistic"
)

report.plot_learning_curves()
report.plot_arc()
report.summary()


---

Outputs

TRACE produces

learning_curves.png
arc_curve.png
trace_summary.json
trace_report.txt


---

Scope

TRACE is a diagnostic toolkit, not a model-training framework.

It does not

train translators

determine true modality information ceilings

infer causal biological mechanisms


Instead, TRACE provides learning-curve diagnostics that help practitioners understand when representation translation meaningfully improves predictive modeling.


---

Repository Structure

trace/
    learning_curves.py
    arc.py
    diagnostics.py
    plotting.py
    workflow.py
examples/
    toy_simulation.ipynb
README.md


---

3. Commit the README
