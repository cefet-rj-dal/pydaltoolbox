# pydaltoolbox Examples

This folder holds the **processed documentation**: each file is a notebook from
[`notebooks/`](../notebooks/) executed end to end, with the real output (metrics,
tables) embedded. It is the Python counterpart of the R package's `examples/`.

These files are **generated** — do not edit them by hand. To rebuild:

```bash
pip install -e ".[docs]"
python tools/build_examples.py
```

Because every notebook is executed, a successful build also acts as a smoke test
over the public API.

## Guided tutorials

- [01 - First Experiment](tutorial/01-first-experiment.md) — the basic fit /
  predict / evaluate rhythm with a decision tree.
- [02 - Sampling Strategy](tutorial/02-sampling-strategy.md) — random vs.
  stratified sampling, k-fold, and building splits from folds.

More tutorials are added as each phase of the port lands (see
[planejamento.md](../planejamento.md)).
