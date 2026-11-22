# Time Series with Forecast (plot_ts_pred)

## About the chart
- Time series with fit and forecast: shows observed values, model fit, and forecast horizon for visual comparison.

## Environment setup


```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```

## Synthetic noisy series; fit and forecast windows


```python
x = np.arange(0, 10.25, 0.25)
y = np.sin(x) + np.random.default_rng(0).normal(0, 0.1, len(x))
adjust = np.sin(x[:35])
prediction = np.sin(x[35:])
```

## Plot observed series with fit and forecast


```python
ax = dal.plot_ts_pred(y=y, yadj=adjust, ypred=prediction)
ax.figure
```




    
![png](output_5_0.png)
    




    
![png](output_5_1.png)
    


## References
- Hyndman, R. J., and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
