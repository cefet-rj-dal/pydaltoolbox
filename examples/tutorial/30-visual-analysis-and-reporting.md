# Tutorial 30 - Visual Analysis and Reporting

A workflow should not end with a metric table. Visual inspection helps understand the data and
communicate results. This tutorial uses one exploratory plot and one compact report table.
(The plot needs the optional `[viz]` extra.)

## An exploratory scatter plot


```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
long = pd.DataFrame({
    "x": iris["sepal_length"],
    "value": iris["sepal_width"],
    "variable": iris["Species"],
})
ax = dal.plot_scatter(long, label_x="sepal_length", label_y="sepal_width")
ax.figure
```




    
![png](output_1_0.png)
    




    
![png](output_1_1.png)
    


## A compact comparison report


```python
slevels = list(iris["Species"].cat.categories)
split = dal.SampleStratified("Species", seed=1).train_test(iris)
train, test = split["train"], split["test"]

models = {
    "majority": dal.ClaMajority("Species", slevels),
    "tree": dal.ClaDtree("Species", slevels, random_state=1),
}
report = []
for name, model in models.items():
    model.fit(train)
    pred = model.predict(test.drop(columns="Species"))
    m = model.evaluate(test["Species"], pred)["metrics"].copy()
    m.insert(0, "model", name)
    report.append(m)
pd.concat(report, ignore_index=True)
```




          model  accuracy  TP  TN  FP  FN  precision  recall  sensitivity  specificity   f1
    0  majority  0.333333  10   0  20   0   0.333333     1.0          1.0          0.0  0.5
    1      tree  0.966667  10  20   0   0   1.000000     1.0          1.0          1.0  1.0



## What to observe
- The plot supports interpretation before model choice, not only after modeling.
- The compact report makes different learners easier to compare in a reproducible way.
- Visualization and reporting are part of the experiment design.
