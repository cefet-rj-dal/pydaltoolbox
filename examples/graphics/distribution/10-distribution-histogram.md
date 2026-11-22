# Histogram (plot_hist)

## About the chart
- Histogram: distributes observations into bins along the x-axis; useful to visualize frequency and skewness.

## Environment setup


```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```

## Generate variables with distinct distributions


```python
rng = np.random.default_rng(1)
example = pd.DataFrame({
    "exponential": rng.exponential(1.0, 10000),
    "uniform": rng.uniform(2.5, 3.5, 10000),
    "normal": rng.normal(5.0, 1.0, 10000),
})
example.head()
```




       exponential   uniform    normal
    0     1.073029  2.807068  3.867724
    1     0.308453  3.062785  5.051074
    2     5.375437  3.443158  5.637334
    3     0.366427  3.028611  4.689365
    4     0.115362  2.517028  4.907643



## Histogram of the exponential variable


```python
ax = dal.plot_hist(example[["exponential"]], label_x="exponential", color="#2a78d6")
ax.figure
```




    
![png](output_5_0.png)
    




    
![png](output_5_1.png)
    


## References
- Freedman, D., and Diaconis, P. (1981). *On the histogram as a density estimator.*
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
