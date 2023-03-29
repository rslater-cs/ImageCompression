import torch
from decimal import Decimal, getcontext
from typing import List

getcontext().prec = 1000

class AC():

    def encode(self, data: torch.Tensor):
        data = data.flatten()
        freq_map = self.create_map(data)
        probs = self.create_probs(freq_map)


    def create_map(self, data: torch.Tensor):
        val_map = dict()

        for item in data:
            if item in val_map:
                val_map[item] += 1
            else:
                val_map[item] = 1

        return val_map

    def create_probs(self, freqs):
        probs = torch.empty(len(freqs))

        total = torch.sum(torch.tensor(freqs.values()))

        i = 0
        for key, value in freqs:
            probs[i] = value/total
            i+=1

        return probs

    def calc_interval(probs: torch.Tensor, prob_index: int):
        S = torch.sum(probs[0:prob_index])
        R = torch.sum(probs)
        
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