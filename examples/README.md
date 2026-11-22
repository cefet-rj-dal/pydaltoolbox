# pydaltoolbox Examples

The examples now support two complementary entry points:

- a guided tutorial track for readers who want to learn the workflow step by step
- thematic collections for readers who want to inspect one family of methods at a time

If you are new to `pydaltoolbox`, start with the tutorials. If you already know the basic Experiment Line, jump directly to the thematic collections.

## Guided Tutorial Track

- [Tutorials](tutorial/README.md) - a structured path with semantic numbering blocks for foundations, classification study, workflow transfer, and extension.

## Thematic Example Collections

- [Transformations](transf/README.md) - sampling, cleaning, scaling, encoding, smoothing, balancing, feature selection, and dimensionality reduction.
- [Classification](classification/README.md) - foundations, core learner families, and model selection.
- [Regression](regression/README.md) - interpretable models, nonlinear learners, and tuning.
- [Clustering](clustering/README.md) - partition-based methods, density-based clustering, and tuning.
- [Pattern mining](patterns/README.md) - association rules, itemsets, and sequential patterns.
- [Graphics](graphics/README.md) - comparison, distribution, relationship, time-oriented, and export-focused visualizations.
- [Custom extensions](custom/README.md) - examples showing how to add new transformations and learners while preserving the DAL workflow.

## Documentation Design

The example corpus now follows a clearer didactic convention:

- numbering uses semantic gaps so related examples stay grouped by topic
- each subfolder `README` explains the intended reading order instead of acting only as a catalog
- the weaker examples are being brought closer to the same teaching style used in the strongest notebooks

## Generating the Processed Examples

Install the execution and optional method dependencies, then run the converter
from any directory:

```bash
pip install -e ".[docs,patterns,xgboost,viz]"
python notebooks/conversor.py
```

To rebuild only one branch or notebook, pass its path relative to `notebooks/`:

```bash
python notebooks/conversor.py graphics
python notebooks/conversor.py graphics/comparison/01-comparison-bar.md
```

The generated Markdown and its figures are written to the matching location
under `examples/`.
