from requests import delete
import torch
from torch.optim import Adam, SGD
from models.ConvCompression import ConvCompression
import numpy as np
from PIL import Image
from torchvision import transforms
from data.test_data import TestSet
from torch.utils.data import DataLoader
from time import time

ASPECT_RATIO = (16, 9)
BASE_SIZE = (ASPECT_RATIO[0]*8, ASPECT_RATIO[1]*8)
PATCH_SIZE = (BASE_SIZE[0]*1, BASE_SIZE[1]*1)

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(torch.cuda.get_device_name(device))

dataset = TestSet(PATCH_SIZE[0], PATCH_SIZE[1])
dataloader = DataLoader(dataset, batch_size=len(dataset)//2)

print("PatchSize", PATCH_SIZE)
print("DataSize", len(dataset))
print("Batch Size", len(dataset)/2)

model = ConvCompression()
model = model.to(device)
model.eval()

prev = time()

for x in range(100):
    for patch in iter(dataloader):
        patch = patch.to(device)
        outputs = model(patch)

post = time()

print(outputs.shape)
print((post-prev)/100)