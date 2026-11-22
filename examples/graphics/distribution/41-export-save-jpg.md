# Exporting a Chart to JPG/PNG

## About the feature
- Chart saving: export a figure to raster formats (JPG, PNG) with `Figure.savefig`.

## Environment setup


```python
import numpy as np
import pandas as pd

import pydaltoolbox as dal
```


```python
import os
import tempfile
```

## Build a chart


```python
x = np.arange(0, 10, 0.25)
serie = pd.DataFrame({"x": x, "sin": np.sin(x), "cosine": np.cos(x) + 5})
ax = dal.plot_series(serie)
ax.figure
```




    
![png](fig/41-export-save-jpg/output_4_0.png)
    




    
![png](fig/41-export-save-jpg/output_4_1.png)
    


## Save it as JPG and PNG


```python
for ext in ("jpg", "png"):
    path = os.path.join(tempfile.gettempdir(), f"series.{ext}")
    ax.figure.savefig(path, dpi=100, bbox_inches="tight")
    print("saved", path)
```

    saved C:/Users/eduar/AppData/Local/Temp/series.jpg
    saved C:/Users/eduar/AppData/Local/Temp/series.png
    

## References
- Wickham, H. (2016). *ggplot2: Elegant Graphics for Data Analysis.* Springer.
