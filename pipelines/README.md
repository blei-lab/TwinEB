# Pipelines

This directory contains nextflow scripts to run all experiments that were studied in this manuscript.
The directory structure is as follows:

```
├── README.md
├── bmf.nf
├── dataset_path.csv
├── gmf
├── nextflow.config
├── pmf
└── run.sh
```

## run.sh

Script to run the nextflow pipeline that will run all experiments for this manuscript. 
These include 4 datasets, over many configurations of two models.

NB: please set the paths according to your platform. 
NB: there are about 65,520 experiments that this script will attempt to run. ## bmf.nf

The nextflow script. 
It will generate the split/train/validation split for the given dataset, runs the model for each configuration in the `grid_search.csv` file, and gather the results in the `summary.csv.gz` file.## dataset_path.csv

Absolute paths to the datasets that were studied in this manuscript.
## gmf

Contains `grid_search.csv` with all model specifications (configurations), values of the parameters used to run each the Gaussian matrix factorization experiments. ## nextflow.config

An example config file for nextflow. ## pmf

Contains `grid_search.csv` with all model specifications (configurations), values of the parameters used to run each the Poisson matrix factorization experiments. 

## README.md

This file.