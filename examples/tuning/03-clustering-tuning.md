# Clustering Tuning (CluTune)

Clustering is unsupervised, so `CluTune` does not use cross-validation. Instead it fits
the clusterer for each configuration and scores it with the model's intrinsic metric
(for k-means, the silhouette), then the model's selector picks the best configuration.
`fit` returns the tuned, fitted clusterer.

**Reference:** Satopaa, V., Albrecht, J., Irwin, D., Raghavan, B. (2011). *Finding a
"Kneedle" in a Haystack: Detecting Knee Points in System Behavior.*


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

tune = dal.CluTune(dal.ClusterKmeans(seed=42), ranges={"k": range(2, 8)})
model = tune.fit(x)
print("chosen k:", model.k)
```

    chosen k: 2
    


```python
labels = model.cluster(x)
model.evaluate(labels, iris["Species"])["metrics"]
```




                    metric       value      goal      type
    0           silhouette    0.681046  maximize  internal
    1       davies_bouldin    0.404293  minimize  internal
    2    calinski_harabasz  513.924546  maximize  internal
    3              entropy    0.757101  minimize  external
    4               purity    0.666667  maximize  external
    5  adjusted_rand_index    0.539922  maximize  external


