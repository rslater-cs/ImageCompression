import torch
from model_scripts import data_saver
from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-r", "--res", dest="resolution", help="The height and width", type=int)
parser.add_argument("-t", "--transfer_dim", dest="channels", help="The number of channels to be encoded", type=int)
args = vars(parser.parse_args())

M = (args["resolution"]*args["resolution"]*args["channels"])
split = 1.0
N = int(split*M)

print("string length", N)
print("chunks =", M//N)

test_data = (torch.randn((N,))*255).type(torch.uint8)

frequency_table = test_data.bincount()
probability_table = frequency_table/torch.sum(frequency_table)
probability_table = probability_table.tolist()

test_data = test_data.tolist()

print(test_data[-10:])

cum_freq = prob_to_cum_freq(probability_table, N)

encoder = RangeEncoder(f'./saved_images/testfile_{args["resolution"]}_{args["channels"]}.bin')
encoder.encode(test_data, cum_freq)
encoder.close()

decoder = RangeDecoder(f'./saved_images/testfile_{args["resolution"]}_{args["channels"]}.bin')
decoded_data = decoder.decode(len(test_data), cum_freq)
decoder.close()

print(decoded_data[-10:])

correct = True
for i in range(len(test_data)):
    if test_data[i] != decoded_data[i]:
        correct = False
        break

print("Successful compression =", correct)