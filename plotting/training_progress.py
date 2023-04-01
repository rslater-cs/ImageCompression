import pandas as pd
import matplotlib.pyplot as plt
import os

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Optional

axis_index = {'epoch':0, 'loss':1, 'psnr':2}

def load_dataframes(tables: List[Path]):
    dataframes = []
    for table in tables:
        frame = pd.read_csv(table)
        dataframes.append(frame)

    return dataframes

def display_data(tables: List[Path], y_axis_index: int, x_axis_index: int, y_name: str, x_name: str, title: str, axis_labels = None, y_bounds: Optional[Tuple[float, float]] = None, x_bounds: Optional[Tuple[float, float]] = None, save_path = None):
    tables: List[pd.DataFrame] = load_dataframes(tables)

    for table in tables:
        table_len = len(table.columns)
        indexes = list(range(table_len))
        indexes.remove(y_axis_index)
        indexes.remove(x_axis_index)
        table.drop(labels=table.columns[indexes], axis=1)

    joined_table = pd.DataFrame()
    joined_table[x_name] = tables[0][x_name]
    for i, table in enumerate(tables):
        joined_table[f'{y_name}_{i}'] = table[y_name]

    for i in range(len(tables)):
        if axis_labels != None:
            label = axis_labels[i]
        else:
            label = axis_labels[i]
        plt.plot(joined_table[x_name], joined_table[f'{y_name}_{i}'], label=label)

    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.ylim(y_bounds)
    plt.xlim(x_bounds)
    plt.legend(loc='lower right')

    if save_path != None:
        plt.savefig(save_path)

    plt.show()

def clean_params(params):
    cparams = []

    for i in range(len(params)):
        num = ''.join(params[i])
        cparams.append(num)

    return cparams

def get_parameters(params):
    parameters = []
    for i in range(len(params)//4):
        start = 4*i
        parameters.append(f'e{params[start]}_t{params[start+1]}_w{params[start+2]}_d{params[start+3]}')
    return parameters

def get_folder_names(params):
    folder_names = []
    for i in range(len(params)//4):
        start = 4*i
        folder_names.append(f'..\saved_models\SwinCompression_e{params[start]}_t{params[start+1]}_w{params[start+2]}_d{params[start+3]}')
    return folder_names

def validate_folders(folder_names):
    for folder in folder_names:
        if not os.path.exists(folder):
            raise Exception(f'{folder} does not exist')
        
def get_files(folders, file_name):
    files = []
    for folder in folders:
        files.append(Path(f'{folder}\{file_name}.csv'))

    return files

if __name__ == '__main__':
    # arguments: file type (valid, train), folders (list(list(hyperparameters))), metric (loss, psnr)
    parser = ArgumentParser()
    parser.add_argument("-v", "--valid", dest="valid", help="Whether to look at train or valid data", type=bool)
    parser.add_argument("-l", "--loss", dest="loss", help="Whether to look at psnr or loss data", type=bool)
    parser.add_argument("-f", "--folders", dest="param_set", help="A list of the model parameters to be looked at", type=list, nargs='*')
    parser.add_argument("-n", "--name", dest="name", help="Name of plot", type=str)

    args = vars(parser.parse_args())

    if(len(args['param_set']) == 0):
        raise Exception("You must provide at least on set of model parameters")
    if(len(args['param_set']) % 4 != 0):
        raise Exception("Each folder must be described by 4 parameters (embed, transfer, window, depth)")
    
    args['param_set'] = clean_params(args['param_set'])
    print(args['param_set'])
    labels = get_parameters(args['param_set'])
    print(labels)
    folders = get_folder_names(args["param_set"])
    print(folders)
    validate_folders(folders)

    file = "valid" if args['valid'] else "train"

    metric = 'loss' if args['loss'] else "psnr"

    y_name = f'{file}_{metric}'

    x_axis = 'epoch'

    files = get_files(folders, file)
    print(files)

    # data = load_dataframes(files)
    display_data(files, axis_index[metric], axis_index[x_axis], y_name=y_name, x_name=x_axis, axis_labels=labels ,title='', save_path=f'./plots/{args["name"]}.png')

