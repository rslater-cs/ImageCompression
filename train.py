from torch import cuda, log10, save, load
from math import ceil
import torch.optim as optim
from torch.nn import MSELoss, Module, DataParallel
from torch.utils.data import DataLoader
from loggers.printing import Printer, Status
from loggers.metrics import MetricLogger

from tqdm import tqdm

from data_loading import imagenet, cifar_10
from model_scripts import model_requirements, model_saver

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

def train(model: Module, optimizer: optim.Optimizer, criterion: MSELoss, tepoch: tqdm, path, device, epoch: int, epochs: int):
    model.requires_grad_(True)
    log = Printer(path, name = "current_epoch")
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
            log.print(f'Epoch {epoch+1}/{epochs}({100*current_b/tepoch.total}%)')
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


def start_session(model: Module, epochs, batch_size, save_dir, data_dir):

    # does get cuda:0
    base_device = "cuda:0" if cuda.is_available() else "cpu"
    devices = [i in range(cuda.device_count())]

    device_info(save_dir)

    print("Using Devices", devices)

    model = DataParallel(model, device_ids=devices)
    model.train()
    param_count = model_requirements.get_parameters(model)
    print("TOTAL PARAMETERS:", f'{param_count}')

    param_doc = Printer(save_dir, name="model_info", mode='w')
    param_doc.print(f'total parameters: {param_count}')

    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    mode = 'w'
    if(os.path.exists(f'{save_dir}/checkpoint.pt')):
        checkpoint = load(f'{save_dir}/checkpoint.pt')
        start_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        mode = 'a'

    # dataset = imagenet.IN(portion=subset)
    dataset = imagenet.IN(data_dir)
    trainloader = DataLoader(dataset.trainset, batch_size=batch_size, shuffle=dataset.shufflemode)
    validloader = DataLoader(dataset.validset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(dataset.testset, batch_size=batch_size, shuffle=False)

    train_len = len(dataset.trainset)
    train_batches = ceil(train_len/batch_size)

    valid_len = len(dataset.validset)
    valid_batches = ceil(valid_len/batch_size)

    test_len = len(dataset.testset)
    test_batches = ceil(test_len/batch_size)

    log = Printer(save_dir, mode=mode)
    status = Status(save_dir)
    training_log = MetricLogger(save_dir, name='train', size=train_batches, mode=mode)
    valid_log = MetricLogger(save_dir, name='valid', size=valid_batches, mode=mode)
    test_log = MetricLogger(save_dir, name="test", size=test_batches, mode=mode)

    for epoch in range(start_epoch, epochs):
        with tqdm(trainloader, unit="batch") as tepoch:
            tr_psnr, tr_loss = train(model, optimizer, criterion, tepoch, save_dir, base_device, epoch, epochs)
            v_psnr, v_loss = valid(model, criterion, validloader, base_device)

            training_log.put(epoch, tr_loss, tr_psnr)
            valid_log.put(epoch, v_loss, v_psnr)

            log.print(f'Epoch {epoch}: Loss = {tr_loss/train_batches}, PSNR = {tr_psnr/train_batches}')
            log.print(f'Valid Score: Loss = {v_loss/valid_batches}, PSNR = {v_psnr/valid_batches}')

        # status.print(f'Progress saved at:, {model_saver.save_model(model, save_dir, in_progress=True)}')
        save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optimizer.state_dict()
        }, f'{save_dir}/checkpoint.pt')

    model.eval()
    tst_psnr, tst_loss = valid(model, criterion, testloader, base_device)

    status.print(f'Loss: {tst_loss/test_batches}, PSNR: {tst_psnr/test_batches}')
    test_log.put(0, tst_loss, tst_psnr)

    save({
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict()
    }, f'{save_dir}/final_model.pt')

    os.remove(f'{save_dir}/checkpoint.pt')

    log.print(f'Final model saved at: {save_dir}')


