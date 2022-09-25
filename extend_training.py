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

EPOCHS = 1

LAYERS = 3
REDUCTION_FACTOR = 2**3

FRAME_SIZE = np.asarray([1280, 720])
PATCH_SIZE = np.asarray([80, 80])

VALID_PATCH = FRAME_SIZE / PATCH_SIZE

BATCH_SIZE = 600

MODEL_PATH = ".\\saved_models\\compressionnet_1\\compressionnet.pth"

if(np.all(VALID_PATCH % 1 != 0)):
    raise Exception("Frame of {}x{} cannot be split into even patches of {}x{}".format(FRAME_SIZE[0], FRAME_SIZE[1], PATCH_SIZE[0], PATCH_SIZE[1]))

VALID_REDUCTION = PATCH_SIZE / REDUCTION_FACTOR

if(np.all(VALID_REDUCTION % 1 != 0)):
    raise Exception("Patch of {}x{} cannot be reduced by factor {}".format(PATCH_SIZE[0], PATCH_SIZE[1], REDUCTION_FACTOR))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Using", device)

model = torch.load(MODEL_PATH).to(device)
print(model)
model.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

patch_dataset = PatchSet(FRAME_SIZE, PATCH_SIZE, "movies\\nuclearFamily_Trim.mp4")
patch_loader = DataLoader(patch_dataset, batch_size=BATCH_SIZE)

for epoch in range(EPOCHS):
    with tqdm(patch_loader, unit="batch") as tepoch:
        data_speed = time()
        for inputs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

saved_path = torch.save(model, MODEL_PATH)

print("Model saved at:", saved_path)