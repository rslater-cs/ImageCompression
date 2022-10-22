from torch import cuda
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_loading import frame_loader, imageframe_loader, cifar_10
from models import ConvCompression, SwinCompression
from model_analyser import model_requirements, model_saver
import ssl

import time

ssl._create_default_https_context = ssl._create_unverified_context

EPOCHS = 20
BATCH_SIZE = 16
MOVIE_PATH = "E:\\Programming\\Datasets\\MoviesDataset\\DVU_Challenge\\Movies\\512_288_IMS"

device = "cuda:0" if cuda.is_available() else "cpu"

print("Using", device)

compressor = SwinCompression.FullSwinCompressor(embed_dim=16, transfer_dim=16, patch_size=[2,2], depths=[2,2,2,6,2], num_heads=[2,2,2,2,2], window_size=[2,2])
# compressor = ConvCompression.FullConvConvCompressor(embed_dim=16, transfer_dim=48, reduction_layers=4)
compressor = compressor.to(device)
param_count = model_requirements.get_parameters(compressor)
print("TOTAL PARAMETERS:", f'{param_count:,}')

time.sleep(5)

criterion = MSELoss()
optimizer = optim.Adam(compressor.parameters(), lr=1e-5)

# dataset = imageframe_loader.ImageSet(MOVIE_PATH)
# dataset = frame_loader.FrameSet(MOVIE_PATH)
# dataset = cifar_10.CIFAR()
data_loader = DataLoader(dataset.trainset, batch_size=BATCH_SIZE, shuffle=dataset.shufflemode)

save_path = model_saver.get_path(type=compressor.network_type)

for epoch in range(EPOCHS):
    with tqdm(data_loader, unit="batch") as tepoch:
        for inputs, _ in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            inputs = inputs.to(device)
            outputs = inputs.clone()

            optimizer.zero_grad()

            output_images = compressor(inputs)
            loss = criterion(output_images, outputs)

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

        print("Progress saved at:", model_saver.save_model(compressor, save_path, in_progress=True))

saved_path = model_saver.save_model(compressor, save_path)

print("Model saved at:", saved_path)