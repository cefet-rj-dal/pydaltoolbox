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

## Transformations

One didactic example per transform (objective + reference + fit/transform/inverse):

- [01 - Min-Max Normalization](transf/01-minmax.md)
- [02 - Z-Score Normalization](transf/02-zscore.md)
- [03 - Missing-Value Removal](transf/03-na-removal.md)
- [04 - Simple Imputation](transf/04-imputation.md)
- [05 - Outlier Removal (Boxplot/IQR)](transf/05-outliers-boxplot.md)
- [06 - Outlier Removal (Gaussian/3-sigma)](transf/06-outliers-gaussian.md)
- [07 - Categorical Mapping (one-hot)](transf/07-categ-mapping.md)
- [08 - PCA](transf/08-pca.md)

## Classification

One didactic example per learner (objective + reference + fit/predict/evaluate):

- [01 - Decision Tree](classification/01-decision-tree.md)
- [02 - CART Tree (rpart)](classification/02-rpart.md)
- [03 - k-Nearest Neighbors](classification/03-knn.md)
- [04 - Naive Bayes](classification/04-naive-bayes.md)
- [05 - Logistic Regression / GLM](classification/05-logistic-glm.md)
- [06 - Multinomial Logistic](classification/06-multinomial.md)
- [07 - Elastic-Net Logistic](classification/07-elastic-net.md)
- [08 - Support Vector Machine](classification/08-svm.md)
- [09 - Random Forest](classification/09-random-forest.md)
- [10 - Bagging](classification/10-bagging.md)
- [11 - Boosting](classification/11-boosting.md)
- [12 - Multi-Layer Perceptron](classification/12-mlp.md)
- [13 - Majority Baseline](classification/13-majority-baseline.md)
- [90 - Comparing Learners](classification/90-comparing-learners.md) — model
  selection: the whole family under one uniform API.

## Regression

One didactic example per regressor (objective + reference + fit/predict/evaluate):

- [01 - Linear Regression](regression/01-linear.md)
- [02 - Decision Tree](regression/02-decision-tree.md)
- [03 - k-Nearest Neighbors](regression/03-knn.md)
- [04 - Random Forest](regression/04-random-forest.md)
- [05 - Support Vector Regression](regression/05-svm.md)
- [06 - Multi-Layer Perceptron](regression/06-mlp.md)
- [90 - Comparing Models](regression/90-comparing-regressors.md) — the whole family
  ranked by R2 under one uniform API.

More tutorials are added as each phase of the port lands (see
[planejamento.md](../planejamento.md)).
