# Outlier Removal — Boxplot Rule (OutliersBoxplot)

## About the transformation
- `OutliersBoxplot`: identifies outliers by the boxplot rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR) and removes them from numeric attributes.

## Environment setup


```python
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Remove outliers via the boxplot rule and inspect the result


```python
iris = dal.datasets.iris().copy()
model = dal.OutliersBoxplot(alpha=1.5).fit(iris)
clean = model.transform(iris)
print("rows before:", len(iris), " after:", len(clean))
clean.head()
```

    rows before: 150  after: 146
    




       sepal_length  sepal_width  petal_length  petal_width Species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa



## References
- Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.
