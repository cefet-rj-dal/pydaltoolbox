# Bar Chart with Error Bars (plot_bar)

## About the chart
- Bars with error bars: add uncertainty/variability (e.g., standard deviation) on top of a bar chart.

## Environment setup


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Aggregate mean and standard deviation by species


```python
agg = iris.groupby("Species", as_index=False).agg(
    mean=("sepal_length", "mean"),
    sd=("sepal_length", "std"),
)
agg
```

    C:\Users\eduar\AppData\Local\Temp\ipykernel_12448\2568368498.py:1: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      agg = iris.groupby("Species", as_index=False).agg(
    




          Species   mean        sd
    0      setosa  5.006  0.352490
    1  versicolor  5.936  0.516171
    2   virginica  6.588  0.635880



## Build the bars and overlay error bars


```python
ax = dal.plot_bar(agg[["Species", "mean"]], colors="#2a78d6")
ax.errorbar(range(len(agg)), agg["mean"], yerr=agg["sd"],
            fmt="none", ecolor="darkred", capsize=4, linewidth=1.1)
ax.figure
```




    
![png](output_5_0.png)
    




    
![png](output_5_1.png)
    


## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
