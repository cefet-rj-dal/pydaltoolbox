# Outlier Removal - Gaussian / 3-sigma (OutliersGaussian)

Under a Gaussian assumption, values outside `mean +/- alpha*sd` are treated as
outliers. `alpha=3` is the classic three-sigma rule (about 99.7% of a normal
distribution lies within it). `fit` learns the per-column mean and standard deviation
and `transform` drops rows that fall outside. Use it when the features are roughly
bell-shaped; otherwise the IQR rule is safer.

**Reference:** Pukelsheim, F. (1994). *The Three Sigma Rule.* The American
Statistician 48(2):88-91.


```python
import pydaltoolbox as dal

iris = dal.datasets.iris().copy()
iris.loc[0, "sepal_width"] = 100.0   # inject a gross outlier
```


```python
model = dal.OutliersGaussian(alpha=3.0).fit(iris)
clean = model.transform(iris)
print("before:", len(iris), " after:", len(clean), " max sepal_width:", clean["sepal_width"].max())
```

    before: 150  after: 149  max sepal_width: 4.4
    
