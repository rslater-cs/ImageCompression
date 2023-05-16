import pandas as pd
import matplotlib.pyplot as plt
import os

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Optional

axis_index = {'epoch':0, 'loss':1, 'psnr':2}

# Load multiple dataframes where multiple graphs to be displayed
def load_dataframes(tables: List[Path]) -> List[pd.DataFrame]:
    dataframes = []
    for table in tables:
        frame = pd.read_csv(table)
        dataframes.append(frame)

    return dataframes


if __name__ == '__main__':
    # arguments: file type (valid, train), folders (list(list(hyperparameters))), metric (loss, psnr)
    parser = ArgumentParser()
    # parser.add_argument("-l", "--loss", dest="loss", help="Whether to look at psnr or loss data", type=bool)
    parser.add_argument("-f", "--folder", dest="folder", help="The path of the model to be displayed", type=str)
    parser.add_argument("-l", "--loss", dest="is_loss", help="Wether to use loss as metric", action="store_true")
    parser.add_argument("-n", "--name", dest="name", help="Name of plot", type=str)

    args = vars(parser.parse_args())

    metric = 'loss' if args['is_loss'] else "psnr"

    # Load the training and validation data
    files = [f'{args["folder"]}/train.csv', f'{args["folder"]}/valid.csv']

    data = load_dataframes(files)

    x_axis = data[0]['epoch']

    training_data = data[0][f'train_{metric}']
    validation_data = data[1][f'valid_{metric}']

    # plot the training loss/psnr and validation loss/psnr
    plt.plot(x_axis, training_data, label=f'train {metric}')
    plt.plot(x_axis, validation_data, label=f'validation {metric}')
    plt.legend(loc='lower right')

    plt.xlabel("epoch")
    plt.ylabel(f"{metric}")

    if args['is_loss']:
        plt.title("Loss During Training")
    else:
        plt.title("Peak Signal-to-Noise Ratio During Training")

    plt.savefig(f'./plotting/plots/{args["name"]}.png')