# Custom Clustering

A custom clusterer implements `fit` (storing the training data so the evaluation metrics can
use it) and `cluster` (returning labels). Here the custom clusterer wraps scikit-learn's BIRCH
backend under the DAL `Clusterer` contract.

## A custom clusterer subclassing `Clusterer`


```python
from sklearn.cluster import Birch

import pydaltoolbox as dal


class ClusterBirchCustom(dal.Clusterer):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def fit(self, data, **kwargs):
        data = self._prepare_fit(data)  # stores train_data for the evaluation metrics
        self.model = Birch(n_clusters=self.k).fit(data.to_numpy(float))
        return self

    def cluster(self, data, **kwargs):
        return self.model.predict(self._features(data))
```

## Same clustering workflow


```python
iris = dal.datasets.iris()
x = iris.drop(columns="Species")
ref = iris["Species"]

model = ClusterBirchCustom(k=3).fit(x)
clu = model.cluster(x)
model.evaluate(clu, ref)["metrics"]
```




        metric      value      goal      type
    0     wcss  94.141592  minimize  internal
    1  entropy   0.515579  minimize  external



## What to observe
- `fit` calls `self._prepare_fit(data)` so the internal metrics can see the training data.
- `cluster` returns labels; everything else (evaluation) is inherited.

## References
- Zhang, T., Ramakrishnan, R., and Livny, M. (1996). *BIRCH: An Efficient Data Clustering
  Method for Very Large Databases.*
