# Radar Chart (plot_radar)

## About the chart
- Radar: displays multiple numeric variables on radial axes from a common origin; useful for comparative profiles.

## Environment setup


```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```


```python
iris = dal.datasets.iris()
```

## Mean per numeric variable, shaped as name/value


```python
means = iris.drop(columns="Species").mean()
profile = pd.DataFrame({"name": means.index, "value": means.to_numpy()}).sort_values("name")
profile
```




               name     value
    2  petal_length  3.758000
    3   petal_width  1.199333
    0  sepal_length  5.843333
    1   sepal_width  3.057333



## Build a radar chart


```python
ax = dal.plot_radar(profile, colors="#2a78d6")
ax.figure
```




    
![png](output_6_0.png)
    




    
![png](output_6_1.png)
    


## What to observe
- With four variables, the profile is a quadrilateral; in general it is an n-sided polygon, where n is the number of variables.
- `plot_radar` already sets the full drawing area.

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
- Wilkinson, L. (2005). *The Grammar of Graphics* (2nd ed.). Springer.
