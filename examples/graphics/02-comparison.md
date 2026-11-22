# Comparison Plots (bar, lollipop, pie, radar)

These `plot_*` functions compare magnitudes across categories. They mirror the R
`daltoolbox` ggplot2 graphics on matplotlib and need the optional `[viz]` extra
(`pip install pydaltoolbox[viz]`). Each returns a `matplotlib.axes.Axes`.

**Reference:** Cleveland, W. S., & McGill, R. (1984). *Graphical perception.* JASA, 79(387).


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()

# per-species summaries
one = iris.groupby("Species", as_index=False)["sepal_length"].mean()
two = iris.groupby("Species", as_index=False)[["sepal_length", "sepal_width"]].mean()
```

    C:\Users\eduar\AppData\Local\Temp\ipykernel_21212\4051515343.py:6: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      one = iris.groupby("Species", as_index=False)["sepal_length"].mean()
    C:\Users\eduar\AppData\Local\Temp\ipykernel_21212\4051515343.py:7: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      two = iris.groupby("Species", as_index=False)[["sepal_length", "sepal_width"]].mean()
    

## Bar

`plot_bar` takes a two-column (category, value) table.


```python
ax = dal.plot_bar(one, colors="#2a78d6", label_y="mean sepal length")
ax.figure
```




    
![png](output_3_0.png)
    




    
![png](output_3_1.png)
    


## Grouped bar

`plot_groupedbar` places several series side by side per category.


```python
ax = dal.plot_groupedbar(two)
ax.figure
```




    
![png](output_5_0.png)
    




    
![png](output_5_1.png)
    


## Stacked bar

`plot_stackedbar` stacks the series within each category.


```python
ax = dal.plot_stackedbar(two)
ax.figure
```




    
![png](output_7_0.png)
    




    
![png](output_7_1.png)
    


## Lollipop

`plot_lollipop` is a sparser bar chart: a stem, a circle, and the rounded value.


```python
ax = dal.plot_lollipop(one, max_value_gap=0.2)
ax.figure
```




    
![png](output_9_0.png)
    




    
![png](output_9_1.png)
    


## Pie

`plot_pieplot` shows part-to-whole composition.


```python
ax = dal.plot_pieplot(one)
ax.figure
```




    
![png](output_11_0.png)
    




    
![png](output_11_1.png)
    


## Radar

`plot_radar` draws a multivariate profile over radial axes (at least three variables).


```python
import pandas as pd

profile = pd.DataFrame({
    "name": ["Petal.L", "Petal.W", "Sepal.L", "Sepal.W"],
    "value": [
        iris["petal_length"].mean(),
        iris["petal_width"].mean(),
        iris["sepal_length"].mean(),
        iris["sepal_width"].mean(),
    ],
})
ax = dal.plot_radar(profile, colors="#2a78d6")
ax.figure
```




    
![png](output_13_0.png)
    




    
![png](output_13_1.png)
    

