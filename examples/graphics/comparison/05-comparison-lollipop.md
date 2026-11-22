# Lollipop Chart (plot_lollipop)

## About the chart
- Lollipop: alternative to bars; emphasizes values with a marker and a stem, a lighter visual for comparisons.

## Environment setup


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Aggregate by species


```python
agg = iris.groupby("Species", as_index=False)["sepal_length"].mean()
agg
```

    C:\Users\eduar\AppData\Local\Temp\ipykernel_30744\2622842540.py:1: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      agg = iris.groupby("Species", as_index=False)["sepal_length"].mean()
    




          Species  sepal_length
    0      setosa         5.006
    1  versicolor         5.936
    2   virginica         6.588



## Build a lollipop chart


```python
ax = dal.plot_lollipop(agg, max_value_gap=0.2)
ax.figure
```




    
![png](output_5_0.png)
    




    
![png](output_5_1.png)
    


## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
- Cleveland, W. S. (1994). *The Elements of Graphing Data* (2nd ed.). Hobart Press.
