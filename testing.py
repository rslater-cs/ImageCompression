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

ASPECT_RATIO = np.asarray([16, 9])
BASE_SIZE = ASPECT_RATIO*REDUCTION_FACTOR
PATCH_SIZE = BASE_SIZE*1

dataset = PatchSet(patch_size=PATCH_SIZE)
dataloader = DataLoader(dataset, batch_size=16)