"""
    Utility functions for visualization of 2D matrices.

    @Author: Sohrab Salehi (sohrab.salehi@columbia.edu)
"""
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import numpy as np
import scanpy as sc
import anndata

def fast_heat(X):
    N, _ = X.shape
    coeff = 3
    fig, ax = plt.subplots(figsize=(coeff+1, coeff))
    #ax.set_title(f'cov {N} x {N}')
    cmap =  'viridis'
        
    img = ax.matshow(X, aspect="auto", vmin=np.min(X), vmax=np.max(X), cmap = cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(img, cax)
    plt.tight_layout()

def get_clust_ass(mat, linkage, n_clusters):
    n_clust = np.minimum(n_clusters, mat.shape[0])
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clust).fit(mat)
    return np.squeeze(clustering.labels_)

def agg_cluster(mat, linkage = 'ward', n_clusters_row=10, n_clusters_col=10):
    # linkage in ("ward", "average", "complete", "single"):
    row_ass = get_clust_ass(mat, linkage, n_clusters_row)
    col_ass = get_clust_ass(mat.T, linkage, n_clusters_col)
    return row_ass, col_ass

def fast_heat_plus(X, **kwargs):
    row_ass, col_ass = agg_cluster(X, **kwargs)
    fast_heat(X[np.argsort(row_ass), :][:, np.argsort(col_ass)])

def scanpy_cluster(mat, clust=None, is_counts=True):
    xdata = anndata.AnnData(mat)
    if clust is not None:
        xdata.obs['cluster'] = [f'c{i}' for i in clust]
    if is_counts:
        if not np.all(mat >= 0):
            raise ValueError('Counts matrix must be non-negative.')
        xdata.layers['counts'] = xdata.X.copy()
        sc.pp.normalize_total(xdata)
        sc.pp.log1p(xdata)
    
    sc.tl.pca(xdata)
    sc.pp.neighbors(xdata)
    sc.tl.umap(xdata)
    sc.tl.leiden(xdata)
    return(xdata)