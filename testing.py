from models.SwinCompression import Encoder
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

model = Encoder()
print(model)
model = model.to(device)
model.eval()

patch_dataset = PatchSet(FRAME_SIZE, PATCH_SIZE, "movies\\1280_720\\nuclearFamily.mp4")
patch_loader = DataLoader(patch_dataset, batch_size=BATCH_SIZE)

inputs, labels = iter(patch_loader).next()
inputs, labels = inputs.to(device), labels.to(device)

outputs = model(inputs)

print("Network done")

print(outputs.shape)