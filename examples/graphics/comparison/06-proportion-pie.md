# Pie Chart (plot_pieplot)

## About the chart
- Pie: represents proportions of a total. Use sparingly and with few categories when angles are easy to compare.

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

    C:\Users\eduar\AppData\Local\Temp\ipykernel_32424\2622842540.py:1: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      agg = iris.groupby("Species", as_index=False)["sepal_length"].mean()
    




          Species  sepal_length
    0      setosa         5.006
    1  versicolor         5.936
    2   virginica         6.588



## Build a pie chart


```python
ax = dal.plot_pieplot(agg, colors=["#2a78d6", "#eb6834", "#1baf7a"])
ax.figure
```




    
![png](output_5_0.png)
    




    
![png](output_5_1.png)
    


## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
- Tufte, E. R. (2001). *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press.
