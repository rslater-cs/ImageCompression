import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from data.patch_loader import PatchSet
from torch.utils.data import DataLoader
from models.ConvCompression import ConvCompression
from tqdm import tqdm
import numpy as np
from time import time

EPOCHS = 10

LAYERS = 3
REDUCTION_FACTOR = 2**3

FRAME_SIZE = np.asarray([1280, 720])
ASPECT_RATIO = np.asarray([16, 9])
BASE_SIZE = ASPECT_RATIO*REDUCTION_FACTOR
PATCH_SIZE = BASE_SIZE*1

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Using", device)

model = ConvCompression()
model = model.to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

patch_dataset = PatchSet(FRAME_SIZE, PATCH_SIZE)
patch_loader = DataLoader(patch_dataset, batch_size=512)

for epoch in range(EPOCHS):
    with tqdm(patch_loader, unit="batch") as tepoch:
        data_speed = time()
        for data in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            inputs = data.to(device)

            # print("Data Speed", time()-data_speed)

            optimizer.zero_grad()

            model_speed = time()
            outputs = model(inputs)
            # print("Model speed", time()-model_speed)
            loss = criterion(outputs, inputs)

            prop_speed = time()
            loss.backward()
            optimizer.step()
            # print("Prop Speed", time()-prop_speed)

            tepoch.set_postfix(loss=loss.item())

            data_speed = time()

            # if(index % 200000):
            #     print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss * 1000000 / 200000:.3f}')
            #     running_loss = 0.0

torch.save(model.state_dict(), '.\saved_models\compressionnet.pth')