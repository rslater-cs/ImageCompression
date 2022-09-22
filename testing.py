from requests import delete
import torch
from torch.optim import Adam, SGD
from models.ConvCompression import ConvCompression
import numpy as np
from PIL import Image
from torchvision import transforms
from data.patch_loader import PatchSet
from torch.utils.data import DataLoader
from time import time

LAYERS = 3
REDUCTION_FACTOR = 2**3

FRAME_SIZE = np.asarray([1280, 720])
PATCH_SIZE = np.asarray([80, 80])

patch_dataset = PatchSet(FRAME_SIZE, PATCH_SIZE, "movies\\nuclearFamily_Trim.mp4")
patch_loader = DataLoader(patch_dataset, batch_size=1)


input, label = iter(patch_loader).next()

print(input.shape)

model = ConvCompression()
model.eval()

output = model(input)

print(output.shape)