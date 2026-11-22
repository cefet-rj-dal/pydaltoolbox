# Custom Evaluation (clustering metrics)

## About the utility
- Clustering models in `pydaltoolbox` carry their own default internal and external evaluation metrics.
- Those defaults come from the class constructor, and they can be replaced when the analyst wants a narrower or different evaluation view.

## Didactic goal
Show that metric customization is optional, inspect the defaults of a clustering class, and then replace them explicitly in a controlled example.

## Environment setup


```python
import warnings

import pydaltoolbox as dal
from pydaltoolbox.clustering.metrics import metric_entropy, metric_silhouette

warnings.filterwarnings("ignore")
```

## Load data; keep reference labels only for external interpretation


```python
iris = dal.datasets.iris()
x = iris.drop(columns="Species")
ref = iris["Species"]
```

## Fit ClusterKmeans and inspect its default evaluation set


```python
model_default = dal.ClusterKmeans(k=3, seed=42).fit(x)
clu_default = model_default.cluster(x)
{
    "internal": [m.__name__ for m in model_default.eval_internal],
    "external": [m.__name__ for m in model_default.eval_external],
}
```




    {'internal': ['metric_silhouette',
      'metric_davies_bouldin',
      'metric_calinski_harabasz'],
     'external': ['metric_entropy', 'metric_purity', 'metric_adjusted_rand_index']}



## Run the default evaluation


```python
model_default.evaluate(clu_default, ref)["metrics"]
```




                    metric       value      goal      type
    0           silhouette    0.552819  maximize  internal
    1       davies_bouldin    0.661972  minimize  internal
    2    calinski_harabasz  561.627757  maximize  internal
    3              entropy    0.393886  minimize  external
    4               purity    0.893333  maximize  external
    5  adjusted_rand_index    0.730238  maximize  external



## Customize to keep only one internal and one external metric
`silhouette` gives a compact internal summary of cohesion and separation, while `entropy` shows how mixed each cluster is relative to a reference partition.


```python
model_custom = dal.ClusterKmeans(k=3, seed=42)
model_custom.eval_internal = [metric_silhouette]
model_custom.eval_external = [metric_entropy]
model_custom = model_custom.fit(x)
clu_custom = model_custom.cluster(x)
{
    "internal": [m.__name__ for m in model_custom.eval_internal],
    "external": [m.__name__ for m in model_custom.eval_external],
}
```




    {'internal': ['metric_silhouette'], 'external': ['metric_entropy']}



## Evaluate again after the customization


```python
model_custom.evaluate(clu_custom, ref)["metrics"]
```




           metric     value      goal      type
    0  silhouette  0.552819  maximize  internal
    1     entropy  0.393886  minimize  external



## Available metric functions


```python
import pydaltoolbox.clustering.metrics as metrics

[n for n in dir(metrics) if n.startswith("metric_")]
```




    ['metric_adjusted_rand_index',
     'metric_calinski_harabasz',
     'metric_davies_bouldin',
     'metric_entropy',
     'metric_loglik',
     'metric_noise_points',
     'metric_purity',
     'metric_silhouette',
     'metric_wcss',
     'metric_withinerror']



## What to observe
- The workflow does not change: configure, fit, cluster, and evaluate.
- What changes is only the content of `eval_internal` and `eval_external`.
- The default metric set is class-specific; concrete classes override the base defaults with more suitable metrics.

## References
- MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations.*
- Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.*
- Shannon, C. E. (1948). *A mathematical theory of communication.*
