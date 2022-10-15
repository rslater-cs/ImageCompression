from time import time

from torch import cuda
import torch.optim as optim
from torch.nn import MSELoss, LayerNorm
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from data.patch_loader import PatchSet
from data.model_saver import save_model
from models import ConvCompression, SwinCompression
from model_analyser import model_requirements

NETWORK_TYPE = "SwinCompression"
# NETWORK_TYPE = "ConvCompression"

EPOCHS = 1

FRAME_SIZE = np.asarray([1024, 576])
PATCH_SIZE = np.asarray([1024, 576])

BATCH_SIZE = 1

MOVIE_PATH = "C:\\Users\\ryans\\OneDrive - University of Surrey\\Documents\\Computer Science\\Modules\\Year3\\FYP\\MoviesDataset\\DVU_Challenge\\Movies\\1024_576\\nuclearFamily.mp4"

device = "cuda:0" if cuda.is_available() else "cpu"

print("Using", device)

compressor = SwinCompression.FullSwinCompressor(embed_dim=16, transfer_dim=1, patch_size=[2,2], depths=[2,2,2,4,6,2], num_heads=[2,2,2,2,2,2], window_size=[2,2])
# compressor = ConvCompression.FullConvConvCompressor(32, 1, 6)
compressor = compressor.to(device)
print(compressor)
param_count = model_requirements.get_parameters(compressor)
print("TOTAL PARAMETERS:", f'{param_count:,}')

criterion = MSELoss()
optimizer = optim.Adam(compressor.parameters(), lr=1e-5)

patch_dataset = PatchSet(FRAME_SIZE, PATCH_SIZE, MOVIE_PATH)
patch_loader = DataLoader(patch_dataset, batch_size=BATCH_SIZE)

for epoch in range(EPOCHS):
    with tqdm(patch_loader, unit="batch") as tepoch:
        data_speed = time()
        for inputs in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            inputs = inputs.to(device)

            optimizer.zero_grad()

            output_images = compressor(inputs)
            loss = criterion(output_images, inputs)

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

saved_path = save_model(compressor.encoder, compressor.decoder, NETWORK_TYPE)

print("Model saved at:", saved_path)