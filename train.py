from torch import cuda, log10
from math import ceil
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_loading import imagenet
from model_analyser import model_requirements, model_saver

import time

def pSNR(mse):
    psnr = 10*log10(1.0**2/mse)
    
    return psnr

def start_session(model, epochs, batch_size, subset):

    device = "cuda:0" if cuda.is_available() else "cpu"

    print("Using", device)

    model = model.to(device)
    param_count = model_requirements.get_parameters(model)
    print("TOTAL PARAMETERS:", f'{param_count:,}')

    time.sleep(1)

    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    dataset = imagenet.IN(portion=subset)
    data_loader = DataLoader(dataset.trainset, batch_size=batch_size, shuffle=dataset.shufflemode)
    data_length = len(dataset.trainset)

    save_path = model_saver.get_path(type=model.network_type)

    for epoch in range(epochs):
        total_psnr = 0.0
        with tqdm(data_loader, unit="batch") as tepoch:
            for inputs, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs = inputs.to(device)
                outputs = inputs.clone()

                optimizer.zero_grad()

                output_images = model(inputs)
                loss = criterion(output_images, outputs)
                psnr = pSNR(loss).item()

                loss.backward()
                optimizer.step()

                tepoch.set_postfix({"loss":loss.item(), "pSNR":psnr})
                
                total_psnr += psnr

        print("\n\n\t\tpSNR: {}\n\n".format(total_psnr/ceil(data_length/batch_size)))

        print("Progress saved at:", model_saver.save_model(model, save_path, in_progress=True))

    saved_path = model_saver.save_model(model, save_path)

    print("Model saved at:", saved_path)