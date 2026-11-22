# Feature Generation (FeatureGeneration)

## About the transformation
- `FeatureGeneration`: creates new attributes from named expressions over existing columns.

## Didactic goal
Reinforce that preprocessing is not only about cleaning or scaling. Feature generation changes the representation itself and can make patterns easier for downstream learners to capture.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Create area and ratio features


```python
iris = dal.datasets.iris()
gen = dal.FeatureGeneration(
    sepal_area=lambda df: df["sepal_length"] * df["sepal_width"],
    petal_area=lambda df: df["petal_length"] * df["petal_width"],
    sepal_ratio="sepal_length / sepal_width",
)
gen.transform(iris).head()
```




       sepal_length  sepal_width  petal_length  petal_width Species  sepal_area  petal_area  sepal_ratio
    0           5.1          3.5           1.4          0.2  setosa       17.85        0.28     1.457143
    1           4.9          3.0           1.4          0.2  setosa       14.70        0.28     1.633333
    2           4.7          3.2           1.3          0.2  setosa       15.04        0.26     1.468750
    3           4.6          3.1           1.5          0.2  setosa       14.26        0.30     1.483871
    4           5.0          3.6           1.4          0.2  setosa       18.00        0.28     1.388889



## What to observe
- Generated attributes become regular columns in the transformed dataset.
- This is often one of the most impactful preprocessing steps in practice.
