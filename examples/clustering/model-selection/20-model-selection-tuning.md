# Model Selection — Tuning (CluTune)

## About the utility
- `CluTune`: selects hyperparameters for a clustering method. Clustering is unsupervised, so it fits the clusterer for each configuration and scores it with the model's intrinsic metric, then the model's selector picks the best configuration.
- In this example the tuned base model is `ClusterKmeans`.

## Didactic goal
Preserve the same clustering line of experiment and treat tuning as a change in configuration rather than as a different workflow.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal
from pydaltoolbox.clustering.metrics import metric_entropy, metric_silhouette, selector_best

warnings.filterwarnings("ignore")
```

## Load data; keep reference labels only for interpretation


```python
iris = dal.datasets.iris()
x = iris.drop(columns="Species")
ref = iris["Species"]
x.head()
```




       sepal_length  sepal_width  petal_length  petal_width
    0           5.1          3.5           1.4          0.2
    1           4.9          3.0           1.4          0.2
    2           4.7          3.2           1.3          0.2
    3           4.6          3.1           1.5          0.2
    4           5.0          3.6           1.4          0.2



## Model configuration and tuning setup
The example tunes `k` against a compact internal criterion (`silhouette`) while still reporting an external interpretation score (`entropy`) against `iris["Species"]`.


```python
base_model = dal.ClusterKmeans(seed=42)
base_model.metric = metric_silhouette
base_model.selector = selector_best
base_model.eval_internal = [metric_silhouette]
base_model.eval_external = [metric_entropy]

model = dal.CluTune(base_model, ranges={"k": range(2, 11)})
```

## Fit the tuned model and inspect the selected configuration


```python
model = model.fit(x)
model.k
```




    2



## Generate cluster labels with the selected configuration


```python
clu = model.cluster(x)
pd.Series(clu).value_counts().sort_index()
```




    0    97
    1    53
    Name: count, dtype: int64



## Evaluate the tuned partition


```python
model.evaluate(clu, ref)["metrics"]
```




           metric     value      goal      type
    0  silhouette  0.681046  maximize  internal
    1     entropy  0.757101  minimize  external



## What to observe
- The workflow is still configure, fit, cluster, and evaluate.
- Tuning changes how the clustering configuration is chosen, not how the experiment is read.

## References
- Satopaa, V., Albrecht, J., Irwin, D., and Raghavan, B. (2011). *Finding a "Kneedle" in a Haystack.*
- Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.*
- Shannon, C. E. (1948). *A mathematical theory of communication.*
