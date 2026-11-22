# Custom Extension Examples

This folder explains how to extend `pydaltoolbox` without breaking the package workflow. The numbering now uses wide semantic gaps because each example belongs to a different extension family and should be read as a separate contract: transformation, classification, regression, and clustering.

The didactic idea is simple: once you understand the constructor and the required Python method for one family, the others become easier to adapt.

## Data Transformation

- [01-custom-transformation.md](01-custom-transformation.md) - create a custom transformation object and implement `transform()`.

## Custom Learners

- [10-custom-classification.md](10-custom-classification.md) - create a custom classifier that fits and predicts inside the DAL interface.
- [20-custom-regression.md](20-custom-regression.md) - create a custom regressor while preserving the same experiment contract.
- [30-custom-clustering.md](30-custom-clustering.md) - create a custom clustering method that integrates with unsupervised workflows.

## Base Contracts

These examples document the minimal interfaces that specialized implementations inherit from.

- [40-autoencoder-base-encoder.md](40-autoencoder-base-encoder.md) - inspect the encoder-only autoencoder base contract.
- [41-autoencoder-base-encoder-decoder.md](41-autoencoder-base-encoder-decoder.md) - inspect the encoder-decoder autoencoder base contract.
- [50-pattern-miner-base.md](50-pattern-miner-base.md) - inspect the common pattern-miner workflow and slots.
