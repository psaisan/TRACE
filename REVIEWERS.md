# TRACE Figure 11 Reviewer Reproduction Packet:
(Non public zip archive included for reviewers with via Journal Review)

README Link: https://github.com/psaisan/TRACE/blob/main/REVIEWERS.md

This packet reproduces the manuscript Figure 11 TRACE/ARC reference-regime panel from synthetic simulations only. No external data are required.

The packet includes both a command-line script and a Jupyter notebook wrapper:

- `reproduce_figure11.py` : standalone Python script.
- `reproduce_figure11.ipynb` : notebook wrapper that runs the same script and records wall-clock runtime.

## Installation with pip

From this packet directory:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Installation with conda

```bash
conda env create -f environment.yml
conda activate trace-figure11
```

## Reproduce Figure 11 from the command line

```bash
python reproduce_figure11.py --outdir figs
```

The script writes the summary panel, per-regime panels, metadata, and seed files into `figs/`.

## Reproduce Figure 11 from Jupyter

```bash
jupyter notebook reproduce_figure11.ipynb
```

Run the notebook cells in order. The notebook calls `main()` from `reproduce_figure11.py`, displays the generated summary panel, and writes runtime information to:

```text
figs/trace_notebook_runtime.json
```

## Output files

Expected outputs include:

```text
figs/trace_reference_scenarios_summary.png
figs/trace_reference_scenarios_summary.pdf
figs/trace_reference_scenarios_metadata.json
figs/trace_reference_frozen_seeds.json
figs/trace_notebook_runtime.json        # notebook run only
```

## Seed behavior

If `figs/trace_reference_frozen_seeds.json` is absent, the first run performs deterministic seed discovery and writes the seed file. Including that JSON in later runs makes reproduction fixed-seed and avoids seed discovery.

## Notes

The code uses synthetic data only. The variables follow the manuscript notation:

- $X$: deployable representation.
- $\hat Z$: translated representation, $\hat Z = h(X). $
- $y$: endpoint label.
- $ARC(n) = A_{\hat Z}(n) - A_X(n)$.
