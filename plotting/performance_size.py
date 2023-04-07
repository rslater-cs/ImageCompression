import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Optional

ROOT_FOLDER = Path("./stages/")

def get_score(model_folder):
    data = pd.read_csv(model_folder / 'test.csv')
    return float(data.iloc[0]['test_psnr'])

def get_size(model_folder):
    file = open(model_folder / 'model_info.txt')
    data = file.readline()
    file.close()
    data = data.strip('total parameters: ')
    return int(data)

def get_stats(stage):
    stage_folder = ROOT_FOLDER / stage
    models = len(os.listdir(ROOT_FOLDER / stage))

    psnr_scores = []
    model_sizes = []
    for i in range(models):
        psnr_scores.append(get_score(stage_folder / str(i)))
        model_sizes.append(get_size(stage_folder / str(i)))

    return psnr_scores, model_sizes

def get_text_loc(psnr_scores, model_sizes):
    points = np.array(list(zip(psnr_scores, model_sizes)))

    for i in range(points.shape[0]):
        dist = np.sqrt(np.sum(np.square(points-points[i]), axis=1))
        rel = np.where(dist < 20000, np.ones_like(dist), np.zeros_like(dist))
        rel[i] = 0
        print(rel)

def plot_stats(psnr_scores, model_sizes, name):
    size_min = 40
    size_max = 400

    min_size = min(model_sizes)
    max_size = max(model_sizes)

    sizes = []
    for i in range(len(model_sizes)):
        size = ((model_sizes[i]-min_size)/(max_size-min_size))*(size_max-size_min)+size_min
        sizes.append(size)
        model_sizes[i] = model_sizes[i] / 1000000.0

    colors = cm.rainbow(np.linspace(0, 1, len(model_sizes)))

    print(model_sizes)

    plt.scatter(model_sizes, psnr_scores, c=colors, s=80)

    for i in range(len(model_sizes)):
        plt.text(model_sizes[i]+0.2, psnr_scores[i]+0.3, str(i))

    plt.xlim((min(model_sizes)-1, max(model_sizes)+1))
    plt.ylim((min(psnr_scores)-2, max(psnr_scores)+2))

    plt.savefig(f'./plotting/plots/{name}.png')
    plt.show()

if __name__ == '__main__':
    # arguments: file type (valid, train), folders (list(list(hyperparameters))), metric (loss, psnr)
    parser = ArgumentParser()
    parser.add_argument("-s", "--stage", dest="stage", help="Whether to look at train or valid data", type=int)
    parser.add_argument("-n", "--name", dest="name", help="File name to save results under", type=str)


    args = vars(parser.parse_args())

    psnr_scores, model_sizes = get_stats(str(args['stage']))

    get_text_loc(psnr_scores, model_sizes)

    # plot_stats(psnr_scores, model_sizes, args['name'])



    

