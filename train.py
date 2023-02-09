from torch import cuda, log10
from math import ceil
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from loggers.printing import Printer, Status
from loggers.metrics import MetricLogger

from tqdm import tqdm

from data_loading import imagenet, cifar_10
from model_analyser import model_requirements, model_saver

import os

import time

def device_info(save_dir):
    writer = Printer(save_dir, name="gpu_info")
    writer.print("Has Cuda:")
    writer.print(str(cuda.is_available()))
    writer.print("Cuda Device Count:")
    writer.print(str(cuda.device_count()))
    writer.print("Current Device:")
    writer.print(str(cuda.current_device()))
    writer.print("Current Device ID:")
    writer.print(str(cuda.get_device_name(cuda.current_device())))
    writer.print("Device Names:")

    for i in range(cuda.device_count()):
        writer.print(str(cuda.get_device_name(i)))
    writer.close()

def pSNR(mse):
    psnr = 10*log10(1.0**2/mse)
    
    return psnr

def start_session(model, epochs, batch_size, save_dir, data_dir):

    # does get cuda:0
    device = "cuda:0" if cuda.is_available() else "cpu"

    device_info(save_dir)

    print("Using", device)

    model = model.to(device)
    param_count = model_requirements.get_parameters(model)
    print("TOTAL PARAMETERS:", f'{param_count:,}')

    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # dataset = imagenet.IN(portion=subset)
    dataset = imagenet.IN(data_dir)
    data_loader = DataLoader(dataset.trainset, batch_size=batch_size, shuffle=dataset.shufflemode)
    train_len = len(dataset.trainset)
    valid_len = len(dataset.validset)

    log = Printer(save_dir)
    status = Status(save_dir)
    training_log = MetricLogger(save_dir, name='train', size=ceil(dataset.trainset/batch_size))
    valid_log = MetricLogger(save_dir, name='valid', size=ceil(valid_len/batch_size))

    for epoch in range(epochs):
        total_psnr = 0.0
        total_loss = 0.0
        with tqdm(data_loader, unit="batch") as tepoch:
            current_b = 0
            print_every = 300
            start = time.time()
            for inputs, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                current_b += 1

                if(time.time()-start >= print_every):
                    log.print(f'Epoch {epoch}/{epochs}({current_b/tepoch.total}%)')
                    log.print(f'Elapsed Time: {time.time()-start}')
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
                total_loss += loss.item()

            current_batch += 1

        # print_status(save_dir, "\n\n\t\tpSNR: {}\n\n".format(total_psnr/ceil(data_length/batch_size)))
        training_log.put(epoch, total_loss, total_psnr)

        status.print(f'Progress saved at:, {model_saver.save_model(model, save_dir, in_progress=True)}')

    saved_path = model_saver.save_model(model, save_dir)

    log.print("Final model saved at:", saved_path)