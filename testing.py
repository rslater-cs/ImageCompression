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
PATCH_SIZE = (BASE_SIZE[0]*3, BASE_SIZE[1]*3)

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

dataset = TestSet(PATCH_SIZE[0], PATCH_SIZE[1])
dataloader = DataLoader(dataset, batch_size=len(dataset)//8)

model = ConvCompression()
model = model.to(device)
model.eval()

data = iter(dataloader).next()
data = data.to(device)
print(data.shape)

prev = time()
outputs = model(data)
post = time()

print(outputs.shape)
print(post-prev)