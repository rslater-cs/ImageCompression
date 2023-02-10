from torch import cuda, log10
from math import ceil
import torch.optim as optim
from torch.nn import MSELoss, Module
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

def pSNR(mse):
    psnr = 10*log10(1.0**2/mse)
    
    return psnr

def train(model: Module, optimizer: optim.Optimizer, criterion: MSELoss, tepoch: tqdm, log: Printer, device, epoch: int, epochs: int):
    model.requires_grad_(True)
    total_psnr = 0.0
    total_loss = 0.0
    current_b = 0
    print_every = 300
    start = time.time()
    curr = time.time()
    for inputs, _ in tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        current_b += 1

        if(time.time()-curr >= print_every):
            curr = time.time()
            log.print(f'Epoch {epoch}/{epochs}({100*current_b/tepoch.total}%)')
            log.print(f'Elapsed Time: {curr-start}')

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
    
    return total_psnr, total_loss

def valid(model: Module, criterion: MSELoss, batches, device):
    model.requires_grad_(False)
    total_psnr = 0.0
    total_loss = 0.0

    for inputs, _ in iter(batches):
        inputs = inputs.to(device)
        outputs = inputs.clone()

        output_images = model(inputs)
        loss = criterion(output_images, outputs)
        psnr = pSNR(loss).item()

        total_psnr += psnr
        total_loss += loss.item()

    return total_psnr, total_loss


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
    trainloader = DataLoader(dataset.trainset, batch_size=batch_size, shuffle=dataset.shufflemode)
    validloader = DataLoader(dataset.validset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(dataset.testset, batch_size=batch_size, shuffle=False)
    train_len = len(dataset.trainset)
    valid_len = len(dataset.validset)
    test_len = len(dataset.testset)

    log = Printer(save_dir)
    status = Status(save_dir)
    training_log = MetricLogger(save_dir, name='train', size=ceil(train_len/batch_size))
    valid_log = MetricLogger(save_dir, name='valid', size=ceil(valid_len/batch_size))

    for epoch in range(epochs):
        with tqdm(trainloader, unit="batch") as tepoch:
            tr_psnr, tr_loss = train(model, optimizer, criterion, tepoch, log, device, epoch, epochs)
            v_psnr, v_loss = valid(model, criterion, validloader, device)

            training_log.put(epoch, tr_loss, tr_psnr)
            valid_log.put(epoch, v_loss, v_psnr)

        status.print(f'Progress saved at:, {model_saver.save_model(model, save_dir, in_progress=True)}')

    tst_psnr, tst_loss = valid(model, criterion, testloader, device)

    batches = ceil(test_len/batch_size)
    status.print(f'Loss: {tst_loss/batches}, PSNR: {tst_psnr/batches}')

    saved_path = model_saver.save_model(model, save_dir)

    log.print(f'Final model saved at: {saved_path}')

    training_log.close()
    valid_log.close()