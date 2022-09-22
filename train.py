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
PATCH_SIZE = np.asarray([80, 80])

VALID_PATCH = FRAME_SIZE / PATCH_SIZE

BATCH_SIZE = 768

if(np.all(VALID_PATCH % 1 != 0)):
    raise Exception("Frame of {}x{} cannot be split into even patches of {}x{}".format(FRAME_SIZE[0], FRAME_SIZE[1], PATCH_SIZE[0], PATCH_SIZE[1]))

VALID_REDUCTION = PATCH_SIZE / REDUCTION_FACTOR

if(np.all(VALID_REDUCTION % 1 != 0)):
    raise Exception("Patch of {}x{} cannot be reduced by factor {}".format(PATCH_SIZE[0], PATCH_SIZE[1], REDUCTION_FACTOR))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Using", device)

model = ConvCompression()
print(model)
model = model.to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

patch_dataset = PatchSet(FRAME_SIZE, PATCH_SIZE, "movies\\nuclearFamily_Trim.mp4")
patch_loader = DataLoader(patch_dataset, batch_size=BATCH_SIZE)

for epoch in range(EPOCHS):
    with tqdm(patch_loader, unit="batch") as tepoch:
        data_speed = time()
        for inputs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            inputs, labels = inputs.to(device), labels.to(device)

            # print("Data Speed", time()-data_speed)

            optimizer.zero_grad()

            model_speed = time()
            outputs = model(inputs)
            # print("Model speed", time()-model_speed)
            loss = criterion(outputs, labels)

            prop_speed = time()
            loss.backward()
            optimizer.step()
            # print("Prop Speed", time()-prop_speed)

            tepoch.set_postfix(loss=loss.item())

            data_speed = time()

            # if(index % 200000):
            #     print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss * 1000000 / 200000:.3f}')
            #     running_loss = 0.0

torch.save(model.state_dict(), '.\saved_models\compressionnet_trimmed_black.pth')