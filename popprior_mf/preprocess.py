"""
    Preporcess the RNA data

    1. Load, handle genomic coordinates, 
"""

import anndata
import itertools
import networkx as nx
import scanpy as sc
import scglue
from matplotlib import rcParams
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# check the package works
from scglue import genomics
genomics.check_deps("bedtools")


def filter_coordinateless_genes(rna):
    vv = rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].isna()
    v1 = np.logical_or(vv.chrom, vv.chromStart)
    v2 = np.logical_or(v1, vv.chromEnd)
    vv = (v2 == True)
    drop_gene_names = vv.index.values[vv]
    len(drop_gene_names)
    drop_gene_indicator = np.in1d(rna.var_names, drop_gene_names)
    rna = rna[:, ~drop_gene_indicator].copy()
    return(rna)


def preprocess_rna(rnaDir, gtfPath, nCells=-1, min_genes=100, min_cells=3, min_counts=3, n_top_genes=2000, pct_mito=40):
    """
    Preprocess the RNA data
    Load, handle genomic coordinates, normalize and scale, compute PCA
    Only keeps higly variable genes (based on seurat v3 definition)
    Remove genes with unknown genomic coordinates
    TODO: inspect the removed genes

    input:
        rnaDir: directory containing the RNA data
        gtfPath: path to the GTF file
        nCells: number of cells to keep (if -1, keep all)

    output:
        rna: anndata.AnnData object
    """
    
    rna = sc.read_10x_mtx(rnaDir)
    if nCells > -1:
        sc.pp.subsample(rna,  n_obs=nCells)

    # Find gene coordinates & remove genes with no coordinates
    scglue.data.get_gene_annotation(rna, gtf=gtfPath, gtf_by="gene_name")
    rna = filter_coordinateless_genes(rna)

    # Minimal filtering...
    rna.var['mt'] = rna.var_names.str.startswith('MT-')
    # 1. Filter mitochondrial genes
    sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    plt.hist(rna.obs["pct_counts_mt"], bins=100)
    plt.savefig("../data/hist_pct_mt.png")
    rna = rna[rna.obs["pct_counts_mt"] < pct_mito]
    # 2. Filter cells with too few genes
    sc.pp.filter_cells(rna, min_genes=min_genes)
    # 3. Filter genes with too few counts, or expressed in too few cells
    sc.pp.filter_genes(rna, min_counts=min_counts, min_cells=min_cells) 

    # Find highly variable genes and only keep them
    # Stuart, Tim, et al. "Comprehensive integration of single-cell data." Cell 177.7 (2019): 1888-1902.
    # 1. Using count data, compute mean and variance of each gene
    # 2. Fit a loess with span .3 to variance against mean plot (function sigma(mean_j) for gene j)
    # 3. Compute the regularized z_score as z_ij = (x_ij - mean_j) / sigma(mean_j), then clip by sqrt(N) for N the number fo cells
    # 4. Rank by regularized z_score
    sc.pp.highly_variable_genes(rna, n_top_genes=n_top_genes, flavor="seurat_v3", subset=True)
    rna.layers["counts"] = rna.X.copy()
    # Normalize, log transform and scale (changes .X)
    # Normalize each cell by the total counts in that cell, multiplying by median total counts per cell (so all cells have same total counts)
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    # Unit variance and zero mean
    # TODO: not allowed! Move this after train test split!
    #sc.pp.scale(rna)

    return(rna)


def main(argv):
    """
    Main function
    """

    print(f"argv: {argv}")
    rnaDir = argv[0]
    outDir = argv[1]
    nCells = int(argv[2])
    gtfPath = argv[3]

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Write the command line arguments to a file
    with open(os.path.join(outDir, "argv.txt"), "w") as f:
        for arg in argv:
            f.write(f"{arg}\n")

    rna = preprocess_rna(gtfPath=gtfPath, rnaDir=rnaDir, nCells=nCells)
    anndata.AnnData.write_h5ad(rna, os.path.join(outDir, "rna_full.h5ad"))

if __name__ == "__main__":
    main(sys.argv[1:])
    
    
 

    
