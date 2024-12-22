# !/bin/bash

# Bash file to run nextflow pipelines one after the other
# Will cd into each directory, run the corresponding nextflow pipeline, and repeat

home_dir='path/to/home/dir'

# Please download nextflow and give it executable permissions
# cd $home_dir
# wget https://github.com/nextflow-io/nextflow/releases/download/v22.10.8/nextflow
# chmod +x nextflow

# The directories to run
dirs=(
pmf
gmf
)

# Adjust these paths accordingly
# where the pmf and gmf directories are located
batch_dir='path/to/batch/dir'
# where the origial preprocessed data is located (without the train, valid, and test splits)
data_set_paths='path/to/code/pipelines/dataset_path.csv'
# where the processed data with the train, valid, and test splits are located
train_valid_test_data_path='path/to/data/dir/'

# define an index counter named i starting at zero
i=0
# For each element of dirs, run the nextflow pipeline
for dir in "${dirs[@]}"
do
    echo "Running for $dir"
    # cd into the directory
    cd $batch_dir/$dir
    # Copy nextflow and nextflow.config, or download it
    cp $home_dir/nextflow .
    cp $home_dir/nextflow.config .
    # Ensure the right jave version is loaded
    module load java/11.0.12
    # Run the nextflow pipeline
    cmd_str="./nextflow run bmf.nf -w ./work --is_testing=false --rna_paths=$data_set_paths --sweep_path=$batch_dir/$dir/grid_search.csv --ignore_cache=false --cache_data_dir=$train_valid_test_data_path -profile ceto_gpu -with-report report.html -with-trace"
    # Run the command
    $cmd_str
done