from torch import cuda
import torch.optim as optim
from torch.nn import MSELoss
from data.patch_loader import PatchSet
from data.model_saver import save_model
from torch.utils.data import DataLoader
from models import ConvCompression, SwinCompression
from tqdm import tqdm
import numpy as np
from time import time

NETWORK_TYPE = "SwinCompression"
# NETWORK_TYPE = "ConvCompression"

EPOCHS = 15

LAYERS = 3
REDUCTION_FACTOR = 2**3

FRAME_SIZE = np.asarray([1280, 720])
PATCH_SIZE = np.asarray([1280, 720])

VALID_PATCH = FRAME_SIZE / PATCH_SIZE

BATCH_SIZE = 5

if(np.all(VALID_PATCH % 1 != 0)):
    raise Exception("Frame of {}x{} cannot be split into even patches of {}x{}".format(FRAME_SIZE[0], FRAME_SIZE[1], PATCH_SIZE[0], PATCH_SIZE[1]))

VALID_REDUCTION = PATCH_SIZE / REDUCTION_FACTOR

if(np.all(VALID_REDUCTION % 1 != 0)):
    raise Exception("Patch of {}x{} cannot be reduced by factor {}".format(PATCH_SIZE[0], PATCH_SIZE[1], REDUCTION_FACTOR))

device = "cuda:0" if cuda.is_available() else "cpu"

print("Using", device)

compressor = SwinCompression.FullSwinCompressor(embed_dim=24, output_dim=1, patch_size=[2,2], depths=[2,2,2,2], num_heads=[2,2,2,2], window_size=[2,2])

criterion = MSELoss()
optimizer = optim.Adam(compressor.parameters(), lr=1e-5)

patch_dataset = PatchSet(FRAME_SIZE, PATCH_SIZE, "movies\\nuclearFamily_Trim.mp4")
patch_loader = DataLoader(patch_dataset, batch_size=BATCH_SIZE)

for epoch in range(EPOCHS):
    with tqdm(patch_loader, unit="batch") as tepoch:
        data_speed = time()
        for inputs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output_images = compressor(inputs)
            loss = criterion(output_images, labels)

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

saved_path = save_model(compressor.encoder, compressor.decoder, NETWORK_TYPE)

print("Model saved at:", saved_path)