---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Principal Component Analysis (DtPca)

## About the transformation
- `DtPca`: Principal Component Analysis projects correlated variables onto orthogonal components ordered by explained variance. You can let the tool pick the number of components via an elbow heuristic or set it explicitly.

## Didactic goal
Focus on what changes in the dataset after each operation.

## Environment setup
```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Fit PCA using the target column as reference and transform
```python
iris = dal.datasets.iris()
mypca = dal.DtPca("Species").fit(iris)
mypca.transform(iris).head()
```

## Manually set the number of components and repeat
```python
dal.DtPca("Species", components=3).fit(iris).transform(iris).head()
```

## References
- Pearson, K. (1901). *On lines and planes of closest fit to systems of points in space.*
- Hotelling, H. (1933). *Analysis of a complex of statistical variables into principal components.*
