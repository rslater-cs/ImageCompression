import os
from pathlib import Path
import torch
import shutil
from decimal import Decimal, getcontext
from typing import List
from copy import deepcopy

getcontext().prec = 1000

NETWORK_NAME = 'SwinCompression'

def make_path(data_dir):
    if(not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    else:
        if(not os.path.exists(f'{data_dir}/checkpoint.pt')):
            shutil.rmtree(data_dir)
            os.mkdir(data_dir)

    return Path(data_dir)


def save_model(model, path: Path, in_progress=False):
    if(in_progress):
        encoder_path = path / "encoder_progress.pt"
        decoder_path = path / "decoder_progress.pt"
    else:
        encoder_path = path / "encoder.pt"
        decoder_path = path / "decoder.pt"

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)

    return path

def to_decimal(arr: List):
    for i, val in enumerate(arr):
        arr[i] = Decimal(val)

    return arr

def make_probabilities(arr: List[Decimal]):
    total = Decimal(0.0)

    for val in arr:
        total += val

    for i in range(len(arr)):
        arr[i] = arr[i]/total

    return arr

def cumsum(arr: Decimal):
    total = Decimal(0.0)
    result = [Decimal(0.0)]

    for val in arr:
        result.append(result[-1]+val)

    return result

def arithmetic_encode(features: torch.Tensor):
    features = features.flatten()

    result_table = torch.bincount(features)

    freq_table = to_decimal(result_table.tolist())

    p = make_probabilities(freq_table)

    pcum = cumsum(p)

    features = features.tolist()

    lower = Decimal(0.0)
    width = Decimal(1.0)

    for value in features:
        lower = (width*pcum[value])+lower
        width = width*p[value]

    message = lower+width/2
    return message, result_table

def search_sorted(arr: List[Decimal], item: Decimal, lower: Decimal, width: Decimal):
    index = 0
    
    # maybe item is rarely less than rel_element
    for element in arr[1:]:
        rel_element = (element*width)+lower
        if rel_element >= item:
            return index
        index += 1
    return index-1

def arithmetic_decode(freq_table: torch.Tensor, message: Decimal):
    freq_sum = torch.sum(freq_table)

    features = torch.empty(freq_sum).type(torch.uint8)

    p = make_probabilities(to_decimal(freq_table.tolist()))
    pcum = cumsum(p)

    lower = Decimal(0.0)
    width = Decimal(1.0)

    for i in range(freq_sum):
        symbol = search_sorted(pcum, message, lower, width)

        lower = (width*pcum[symbol])+lower
        width = width*p[symbol]

        features[i] = symbol

    return features



