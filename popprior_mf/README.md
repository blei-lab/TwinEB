# Population Prior for Matrix Factorization

This is an explanation for the algorithms developed in the Population Prior for Matrix Factorization manuscript. 

We discuss the source code, a sample dataset, how to preprocess your own dataset, and how to train, and validate and test the models.
We start with a short vignettes and defer the details to the end of the document.


## Vignettes

We give a short vignettes on how to install the package and train/validate/test a model.
The `driver.py` script houses tasks necessary for generating a train/validate/test split for the dataset (task `setup_data`) and to run the model (task `run_model`).

### Installing the package

We recommend setting up a virtual environment.
We require `python 3.8` or higher.

```
# Install Python 3.8
conda create -n py38 python=3.8
conda activate py38

# Set up a virtual environment and activate it
python3.8 -m venv testpp
source testpp/bin/activate
```


Then install the requirements of the package via `pip` as follows:

```
# Upgrade to the latest version of pip
pip install --upgrade pip
pip install --upgrade setuptools
pip install -r requirements.txt
```

### Running the code

The `driver.py` file implements the modules for training, validation and testing the model. 
A sample run for the dataset that accompanies this code base (with train/validation/test split built-in) is as follows:

```
python driver.py run_model -i ../data/ml-100k_0.2_1.0.pkl -o ../results/may21 -l 10 -m 1000 -f PPCAEBTwinPlus -b 128 -rlr 1e-3 -clr 1e-1 -tol 10 -k 70 -psi 100 -n 10 -c 50 
```

This command will write the results to `../results/may21/Experiment***/config.yaml`. 


### Data

We downloaded this dataset from its [permanent url](https://grouplens.org/datasets/movielens/100k/). 

See the Jupiter notebook `movielens_100k.ipynb` for the preprocessing of this dataset into an anndata object that can be used with the `setup_data` to create the train/validation/test split. 

Assuming that the results of the `movielens_100k.ipynb` is available under the `../data/ml100k_preprocessed.h5ad` directory, we can holdout 20 per cent of the rows, also masking 20 per cent of the elements as follows:

```
python driver.py setup_data -i
./driver.py setup_data -i ../data/synthetic/pmf/pmf_6000_4000_64_1000_3_sparse_0.1.h5ad -o ../data/synthetic/pmf/processed/normalized -p .2 -f True -l 1
```


### Nextflow

See `pipelines/run.sh` for an example run. 
Ensure that you have `nextflow` executable downloaded and a compatible java for your platform:

```
cd ~
# Download nextflow
wget https://github.com/nextflow-io/nextflow/releases/download/v22.10.8/nextflow
# Grant executable permission 
chmod +x nextflow
```

Run the pipeline, assuming that the path to the input data is under `input_paths="path/to/input.csv"` and the parameters of the model specified in `param_sweeps="path/to/param_sweeps.csv"`

```
# Ensure a correct java version is available
module load java/11.0.12
./nextflow run bmf.nf -w ./work --is_testing=false --rna_paths=$input_paths --sweep_path=$param_sweeps -with-report report.html -with-trace
```
Examples for each file is included in the `pipelines` directory.


## More details


### Create 

```
usage: driver.py [-h] [--filePath FILEPATH] [--saveDir SAVEDIR] [--holdoutPortion HOLDOUTPORTION] [--force FORCE] [--cacheDir CACHEDIR] [--correlationCutOff CORRELATIONCUTOFF] [--holdoutRows HOLDOUTROWS]
                 [--ignore_pca IGNORE_PCA]

```


### Running the code

```
python driver.py run_model -i ../data/ml-100k_0.2_1.0.pkl -o ../results/may21 -l 10 -m 100 -f PPCAEBTwinPlus -b 128 -rlr 1e-3 -clr 1e-1 -tol 10 -k 10 -n 10 -c 50 
```

For a list and detail of the available arguments, use `python driver.py run_model -h`:

```

