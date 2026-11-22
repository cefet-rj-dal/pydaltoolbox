# Correlation Heatmap (plot_correlation)

## About the chart
- `plot_correlation`: correlation heatmap for numeric variables.

## Didactic goal
Show a compact view of pairwise association before model fitting -- useful for redundancy inspection and feature screening.

## Environment setup


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Correlation heatmap with hierarchical reordering


```python
ax = dal.plot_correlation(iris.drop(columns="Species"), reorder="hclust")
ax.figure
```




    
![png](output_3_0.png)
    




    
![png](output_3_1.png)
    


## References
- Friendly, M. (2002). *Corrgrams: Exploratory displays for correlation matrices.*
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
