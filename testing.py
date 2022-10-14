from bz2 import compress
from models.SwinCompression import Encoder, Decoder
from data.patch_loader import PatchSet
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from data.patch_loader import PatchSet
from data.model_saver import save_model
from torch.utils.data import DataLoader
from models.ConvCompression import ConvCompression
from tqdm import tqdm
import numpy as np
from time import time

LAYERS = 3
REDUCTION_FACTOR = 2**3

FRAME_SIZE = np.asarray([1280, 720])
PATCH_SIZE = np.asarray([1280, 720])

VALID_PATCH = FRAME_SIZE / PATCH_SIZE

BATCH_SIZE = 1

if(np.all(VALID_PATCH % 1 != 0)):
    raise Exception("Frame of {}x{} cannot be split into even patches of {}x{}".format(FRAME_SIZE[0], FRAME_SIZE[1], PATCH_SIZE[0], PATCH_SIZE[1]))

VALID_REDUCTION = PATCH_SIZE / REDUCTION_FACTOR

if(np.all(VALID_REDUCTION % 1 != 0)):
    raise Exception("Patch of {}x{} cannot be reduced by factor {}".format(PATCH_SIZE[0], PATCH_SIZE[1], REDUCTION_FACTOR))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Using", device)

encoder = Encoder(embed_dim=24, output_dim=1, patch_size=[2,2], depths=[2,2,2,2], num_heads=[2,2,2,2], window_size=[2,2])
decoder = Decoder(input_embed_dim=1, embed_dim=192, patch_size=[2,2], depths=[2,2,2,2], num_heads=[2,2,2,2], window_size=[2,2])

encoder = encoder.to(device)
encoder.eval()

decoder = decoder.to(device)
decoder.eval()

patch_dataset = PatchSet(FRAME_SIZE, PATCH_SIZE, "movies\\1280_720\\nuclearFamily.mp4")
patch_loader = DataLoader(patch_dataset, batch_size=BATCH_SIZE)

patch_iter = iter(patch_loader)

for i in range(5):
    inputs, labels = patch_iter.next()
    inputs, labels = inputs.to(device), labels.to(device)
    start_encoder = time()
    compressed = encoder(inputs)
    print(compressed.shape)
    start_decoder = time()
    decompressed = decoder(compressed)
    end_decoder = time()
    print(decompressed.shape)

    print("ENCODER TIME:", start_decoder-start_encoder)
    print("DECODER TIME", end_decoder-start_decoder)

    print("Network done")