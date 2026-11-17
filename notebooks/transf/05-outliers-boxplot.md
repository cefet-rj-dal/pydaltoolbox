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

# Outlier Removal - Boxplot / IQR (OutliersBoxplot)

Tukey's rule flags a value as an outlier when it falls below `Q1 - alpha*IQR` or above
`Q3 + alpha*IQR`, where `IQR = Q3 - Q1`. `alpha=1.5` corresponds to the standard
boxplot whiskers; `alpha=3` targets only extreme values. `fit` learns the per-column
thresholds and `transform` drops rows that violate them -- it makes no distributional
assumption, which is why it is a robust first choice.

**Reference:** Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.

```python
import pydaltoolbox as dal

iris = dal.datasets.iris().copy()
iris.loc[0, "sepal_width"] = 100.0   # inject a gross outlier
```

```python
model = dal.OutliersBoxplot(alpha=1.5).fit(iris)
clean = model.transform(iris)
print("before:", len(iris), " after:", len(clean), " max sepal_width:", clean["sepal_width"].max())
```
