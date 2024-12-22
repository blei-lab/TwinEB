# Notebooks


Jupyter notebooks to create simulated data (Figure 1), preprocess the data (Figures 2 & 3), and plotting (Figures 1-3 and all figures in the supplement).


```
├── README.md
├── gene_expression_preprocess.ipynb
├── movielens_100k_preprocess.ipynb
├── movielens_1m_preprocess.ipynb
├── goodbooks.ipynb
├── goodbooks_preprocess.ipynb
├── usrers_artists_preprocess.ipynb
```


## Preprocessing

The following notebooks contain code used to preprocess the raw input files:

```
gene_expression_preprocess.ipynb
movielens_100k_preprocess.ipynb
movielens_1m_preprocess.ipynb
usrers_artists_preprocess.ipynbgoodbooks_preprocess.ipynb
```

The output of these scripts can then be used to generate the train/validation/test split via the `setup_data` command in the `driver.py` script.

