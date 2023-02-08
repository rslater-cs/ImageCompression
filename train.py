from torch import cuda, log10
from math import ceil
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_loading import imagenet, cifar_10
from model_analyser import model_requirements, model_saver

import os

import time

def device_info(save_dir, data_dir):
    file = open(save_dir+"/gpu_info.txt", 'w', newline="\n")
    file.write("\nHas Cuda:\n")
    file.write(str(cuda.is_available()))
    file.write("\nCuda device count:\n")
    file.write(str(cuda.device_count()))
    file.write("\nCurrent Device:\n")
    file.write(str(cuda.current_device()))
    file.write("\nCurrent Device ID:\n")
    file.write(str(cuda.get_device_name(cuda.current_device())))
    file.write("\nDevice Names:\n")
    for i in range(cuda.device_count()):
        file.write(str(cuda.get_device_name(i)))
        file.write("\n")
    file.close()

def print_status(save_dir, message):
    file = open(save_dir+"/output.txt", 'w', newline="\n")
    file.write(f'{message}')
    file.close()

def log(save_dir, message):
    file = open(save_dir+"/output.txt", 'a', newline="\n")
    file.write(f'\n{message}\n')
    file.close()

def reset_log(save_dir):
    file = open(save_dir+"/output.txt", 'w', newline="\n")
    file.write(f'')
    file.close()

def pSNR(mse):
    psnr = 10*log10(1.0**2/mse)
    
    return psnr

def start_session(model, epochs, batch_size, save_dir, data_dir):
    reset_log(save_dir)

    # does get cuda:0
    device = "cuda:0" if cuda.is_available() else "cpu"

    device_info(save_dir, data_dir)

    print("Using", device)

    model = model.to(device)
    param_count = model_requirements.get_parameters(model)
    print("TOTAL PARAMETERS:", f'{param_count:,}')

    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # dataset = imagenet.IN(portion=subset)
    dataset = imagenet.IN(data_dir)
    data_loader = DataLoader(dataset.trainset, batch_size=batch_size, shuffle=dataset.shufflemode)
    data_length = len(dataset.trainset)

    save_path = model_saver.get_path(save_dir)

    for epoch in range(epochs):
        total_psnr = 0.0
        print_every = 1000
        current_batch = 0
        start = time.time()
        with tqdm(data_loader, unit="batch") as tepoch:
            for inputs, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                if(current_batch % print_every == 0):
                    log(save_dir, f'Current Batch: {current_batch},\n       \
                        Total Batches: {tepoch.total},\n Elapsed Time: {time.time()-start}')
                    start = time.time()

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

            current_batch += 1

        print_status(save_dir, "\n\n\t\tpSNR: {}\n\n".format(total_psnr/ceil(data_length/batch_size)))

        log("Progress saved at:", model_saver.save_model(model, save_path, in_progress=True))

    saved_path = model_saver.save_model(model, save_path)

    log("Model saved at:", saved_path)