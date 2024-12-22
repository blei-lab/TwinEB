#!/usr/bin/env python
"""
    Set of tools to generate files for a parameter sweep.

    @Author: Sohrab Salehi sohrab.salehi@columbia.edu
"""

import random
import string
import argparse
import sys
import os
import pandas as pd
import yaml
import itertools
from utils import str2bool
from tqdm import tqdm


# ---------------- Utils ----------------
def create_random_str(n=6):
    letters = string.ascii_lowercase
    str_part = ''.join(random.choice(letters) for i in range(n))
    num_part = ''.join(random.choice(string.digits) for i in range(n))
    return num_part + str_part


def product_dict(**kwargs):
    """Create the cartesian product of a dictionary of lists"""
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

# convert a tuple into one dict
def tuple_to_dict(tup):
    """
    Convert a tuple to a dict. 

    Useful when converting the resutls of product_dict, consisting of multiple dictionaries into one dictionary.
    """
    res = {}
    for i in range(len(tup)):
        if type(tup[i]) == dict:
            for key, value in tup[i].items():
                res[key] = value
        else:
            for key, value in tup[i][0].items():
                res[key] = value
    return res


def handle_decoupled(params, yaml_shared):
    # Gather all non_coupled params
    non_coupled_params = {}
    for param in params:
        if param['coupled'] is False:
            non_coupled_params[param['name']] = param['range']
    # Add them to the shared params and expand
    if len(non_coupled_params) > 0:
        tmp_shared = {**yaml_shared.copy(), **non_coupled_params}
    else:
        tmp_shared = yaml_shared.copy()
    expanded = list(product_dict(**tmp_shared))
    return expanded


def handle_coupled(params, expanded):
    """
    Parses the coupled params and expands the list of dictionaries. 

    Coupled params are useful when only specicic pairs of settings are interesting (it avoids a full cartesian product). 
    Example:
    - name: row_prior_vals
      range: [[.1, .1], [.01, .03], [0.1, 0.3], [1, 3], [10, 30], [100, 30]]
      coupled: true
      type: float
      name1: row_prior_concentration
      name2: row_prior_rate
    """
    tmp = []
    # Gather all coupled params
    coupled_params = []
    for param in params:
        if param['coupled'] is True:
            for j in range(len(param['range'])):
                coupled_params.append({param['name1']: param['range'][j][0], param['name2']: param['range'][j][1]})
            
    if len(coupled_params) > 0:
        tmp = list(itertools.product(*[expanded, coupled_params]))
        for i in range(len(tmp)):
            tmp[i] = tuple_to_dict(tmp[i])
    else:
        tmp = expanded
    return tmp


def convert_params_to_csv(res, outPath):
    """
    Combine all into one dataframe and save to csv

    NB: Replaces NaN with 1
    """
    df = pd.DataFrame()
    # Concatenate all the models
    for key, val in res.items():
        df = pd.concat([df, pd.DataFrame(val)])

    # Set int columns to int
    # 'latent_dim', 'seed', 'num_samples', 'row_learning_rate', 'column_learning_rate', 'max_steps', 'batch_size', 'n_llhood_samples', 'row_prior_scale', 'factor_model', 'row_prior_concentration', 'row_prior_rate', 'num_pseudo_obs', 'num_pseudo_obs_global']
    # 'latent_dim', 'seed', 'num_samples', 'max_steps', 'batch_size', 'n_llhood_samples', 'num_pseudo_obs', 'num_pseudo_obs_global']
    # replace all NaN's with 1
    df = df.fillna(1)

    int_columns = ['latent_dim', 'seed', 'num_samples', 'max_steps', 'batch_size', 'n_llhood_samples', 'num_pseudo_obs', 'num_pseudo_obs_global']
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    df['id'] = df.apply(lambda x: create_random_str(), axis=1)
    df['id'] = df['factor_model'].astype(str) + '_' +  df['id'].astype(str)
    assert df.duplicated(['id']).sum() == 0, 'There are duplicates in the id column'
    df.to_csv(outPath, index=False)
    print('Saved to {}'.format(outPath))
    # print the number of rows
    print('Number of rows: {}'.format(df.shape[0]))


def dict_from_yaml_path(path):
    """Helper function to read a yaml file and return the dict"""
    with open(path) as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_dict


def generate_params_from_paths(specifi_path, shared_path, **kwargs):
    """Helper function to generate the params from the paths"""
    yaml_specific = dict_from_yaml_path(specifi_path)
    yaml_shared = dict_from_yaml_path(shared_path)
    return generate_params(yaml_specific, yaml_shared)


def print_params(params, n=3):
    """
    Prints the params

    Args:
        n: number of params to print per model
    """
    for key, val in params.items():
        print('#-----------------------')
        print(key)
        print('#-----------------------')
        for i in range(len(val)):
            if i < n:
                print(val[i])


def generate_params(specifi_path, shared_path, output_path=None, do_print=True):
    """
    Generates the params for the grid search.

    For the specific model, first build the non composite ones, then, add one for each element of the composie ones

    NB: special character is `|` in the model name. Useful to add one off configurations for a model.
    NB: will drop param named dummy. Useful for just adding a param.
    """
    params_specific = dict_from_yaml_path(specifi_path)
    params_shared = dict_from_yaml_path(shared_path)

    res = {}
    for model, params in tqdm(params_specific.items()):
        print(model)
        # Handle decoupled params
        expanded = handle_decoupled(params, params_shared)
        # Handle coupled params
        res[model] = handle_coupled(params, expanded)

    # Add the key as model to each entry of the dict
    for model, params in res.items():
        for i in range(len(params)):
            params[i]['factor_model'] = model

            # Drop special characters (anything after | in the model name)
            params[i]['factor_model'] = params[i]['factor_model'].split('|')[0]

    # Drop the dummy column
    for model, params in res.items():
        for i in range(len(params)):
            if 'dummy' in params[i]:
                del params[i]['dummy']

    if do_print:
        print_params(res)

    if output_path is None:
        # Get the parent directory of the specific path
        parent_dir = os.path.dirname(specifi_path)
        #output_path = os.path.join(parent_dir, 'grid_search.csv.gz')
        output_path = os.path.join(parent_dir, 'grid_search.csv')

    convert_params_to_csv(res, output_path)


def read_default_yaml():
    """Reads the default yaml files and returns the dicts"""
    # parse this into yaml
    specifi_path = '../pipeline/batches/local_test/specific.yaml'
    shared_path = '../pipeline/batches/local_test/shared.yaml'
    # load the files and return the dicts, and close the file
    with open(specifi_path) as f:
        yaml_specific = yaml.load(f, Loader=yaml.FullLoader)
    with open(shared_path) as f:
        yaml_shared = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_specific, yaml_shared
    
    


def concat_grid_search(path1, path2, output_path=None):
    """
    Merges two grid search files.

    NB: Will drop duplicates
    """
    df1 = pd.read_csv(path1)
    df1['id'] = df1['id'].astype(str) + '_0'
    df2 = pd.read_csv(path2)
    df2['id'] = df2['id'].astype(str) + '_1'
    df = pd.concat([df1, df2])
    df.to_csv(output_path, index=False)
    print('Saved to {}'.format(output_path))
    print('Number of rows: {}'.format(df.shape[0]))


if __name__ == '__main__':
    """
    Main function to generate the params for the grid search.

    The params are read from the specific and shared yaml files.
    The params are then expanded and saved to a csv file.
    """

    def add_default_switches(parser):
        parser.add_argument('--output', '-o', type=str, help='The path to the output csv file. Default will save a file named grid_search.csv.gz in the same directory as the specific yaml file.', required=False)
        parser.add_argument("--verbose", '-v', type=str2bool, nargs='?', const=True, default=True,  help="Print the params. Default is True.")
        return parser

    task_name = sys.argv[1]
    if task_name == 'generate_params':
        parser = argparse.ArgumentParser(description='Generate the params for the grid search from shared and specific yaml files.')
        parser.add_argument('--specific', '-s', type=str, help='The path to the specific yaml file.')
        parser.add_argument('--shared', '-sh', type=str, help='The path to the shared yaml file.')
        parser = add_default_switches(parser)

        args = parser.parse_args(sys.argv[2:])
        generate_params(args.specific, args.shared, output_path=args.output, do_print=args.verbose)
    elif task_name == 'generate_from_dir':
        # Expect the specific and shared yaml files to be in the same directory
        parser = argparse.ArgumentParser(description='Generate the params for the grid search from shared and specific yaml files under the given input direcotry.')
        parser.add_argument('--input', '-i', type=str, help='The path to the input directory.')
        parser = add_default_switches(parser)

        # Create the outputs
        args = parser.parse_args(sys.argv[2:])
        specific_path = os.path.join(args.input, 'specific.yaml')
        shared_path = os.path.join(args.input, 'shared.yaml')
        # Check that the files exist
        assert os.path.exists(specific_path), f'The specific yaml file does not exist ({specific_path})'
        assert os.path.exists(shared_path), f'The shared yaml file does not exist ({shared_path})'
        generate_params(specific_path, shared_path, output_path=args.output, do_print=args.verbose)
    elif task_name == 'concat':
        parser = argparse.ArgumentParser(description='Concatenate two grid search files.')
        parser.add_argument('--path1', '-p1', type=str, help='The path to the first grid search file.', required=True)
        parser.add_argument('--path2', '-p2', type=str, help='The path to the second grid search file.', required=True)
        parser.add_argument('--output', '-o', type=str, help='The path to the output csv file.', required=True)
        args = parser.parse_args(sys.argv[2:])
        concat_grid_search(args.path1, args.path2, args.output)
    else:
        raise ValueError(f'Invalid task name ({task_name})')
