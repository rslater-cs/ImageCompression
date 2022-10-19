from torch import cuda
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_loading import frame_loader, imageframe_loader, cifar_10
from models import ConvCompression, SwinCompression
from model_analyser import model_requirements, model_saver

EPOCHS = 5
BATCH_SIZE = 1
MOVIE_PATH = "C:\\Users\\ryans\\OneDrive - University of Surrey\\Documents\\Computer Science\\Modules\\Year3\\FYP\\MoviesDataset\\DVU_Challenge\\Movies\\1024_576_IMS\\nuclearFamily"

device = "cuda:0" if cuda.is_available() else "cpu"

print("Using", device)

compressor = SwinCompression.FullSwinCompressor(embed_dim=16, transfer_dim=1, patch_size=[2,2], depths=[2,2,4,6], num_heads=[2,2,2,2], window_size=[2,2])
# compressor = ConvCompression.FullConvConvCompressor(32, 1, 4)
compressor = compressor.to(device)
param_count = model_requirements.get_parameters(compressor)
print(compressor)
print("TOTAL PARAMETERS:", f'{param_count:,}')

criterion = MSELoss()
optimizer = optim.Adam(compressor.parameters(), lr=1e-5)

# dataset = imageframe_loader.ImageSet(MOVIE_PATH)
# dataset = frame_loader.FrameSet(MOVIE_PATH)
dataset = cifar_10.CIFAR()
data_loader = DataLoader(dataset.trainset, batch_size=BATCH_SIZE, shuffle=dataset.shufflemode)

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

saved_path = model_saver.save_model(compressor.encoder, compressor.decoder, compressor.network_type)

print("Model saved at:", saved_path)