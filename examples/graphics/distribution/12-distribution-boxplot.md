# Boxplot (plot_boxplot)

## About the chart
- Boxplot: summarizes a distribution via quartiles and highlights outliers; comparable across groups.

## Environment setup


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Boxplot per numeric column (plain, then coloured)


```python
ax = dal.plot_boxplot(iris.drop(columns="Species"))
ax.figure
```




    
![png](output_3_0.png)
    




    
![png](output_3_1.png)
    


## References
- Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
