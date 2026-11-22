"""Structure plots: hierarchical-clustering dendrogram."""
from __future__ import annotations

import numpy as np

from ._helpers import TEXT_SECONDARY, new_axes


def plot_dendrogram(hc, labels=True, label_size: float = 3, title=None):
    """Dendrogram of a hierarchical clustering.

    Theory:
        Agglomerative clustering repeatedly merges the two closest clusters; the
        dendrogram records that history as a tree whose join heights encode the
        dissimilarity at each merge. Cutting the tree at a height yields a flat
        partition, so the plot both summarises structure and guides the choice of
        the number of clusters.

    ``hc`` is a SciPy linkage matrix (from ``scipy.cluster.hierarchy.linkage``).
    ``labels`` toggles leaf labels (or supply a list of leaf labels); passing an
    object with a ``linkage_`` / ``children_`` attribute is also accepted.

    References:
        Ward, J. H. (1963). Hierarchical grouping to optimize an objective
            function. JASA, 58(301), 236-244.
        Wickham, H. (2016). ggplot2. Springer.

    Porting:
        Mirrors ``grf_dendrogram.R``. Fidelity: adapted.
        - Takes a SciPy linkage matrix (the Python analogue of R's ``hclust``)
          and renders with ``scipy.cluster.hierarchy.dendrogram``.
    """
    from scipy.cluster.hierarchy import dendrogram

    linkage_matrix = np.asarray(hc)
    leaf_labels = None
    if isinstance(labels, (list, tuple, np.ndarray)):
        leaf_labels = list(labels)
        no_labels = False
    else:
        no_labels = not labels

    ax = new_axes()
    dendrogram(linkage_matrix, ax=ax, labels=leaf_labels, no_labels=no_labels,
               color_threshold=0, above_threshold_color="#333333",
               leaf_font_size=label_size * 2.845)
    ax.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, rotation=90)
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontweight="bold")
    return ax
