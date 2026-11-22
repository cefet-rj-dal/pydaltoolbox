# Boxplot by Class (plot_boxplot_class)

## About the chart
- `plot_boxplot_class`: boxplots of one numeric attribute grouped by a class label.

## Didactic goal
Complement the plain boxplot with a grouped version -- often the more useful view when the question is how the distribution changes across classes.

## Environment setup


```python
import pydaltoolbox as dal

iris = dal.datasets.iris()
```

## Grouped boxplot of sepal width by species


```python
ax = dal.plot_boxplot_class(iris[["sepal_width", "Species"]], class_label="Species")
ax.figure
```




    
![png](output_3_0.png)
    




    
![png](output_3_1.png)
    


## References
- Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.
