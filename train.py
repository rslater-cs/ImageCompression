from torch import cuda, save, load, no_grad
import torch.optim as optim
from torch.nn import MSELoss, Module, DataParallel
from torch.utils.data import DataLoader

from data_scripts.loggers.printing import Printer, Status
from data_scripts.loggers.metrics import MetricLogger
from data_scripts import imagenet
from data_scripts.data_saver import save_gpu_stats
from models import model_requirements, metrics

from tqdm import tqdm

from math import ceil

import os
import time

def train(model: Module, optimizer: optim.Optimizer, tepoch: tqdm, path, device, epoch: int, epochs: int):
    log = Printer(path, name = "current_epoch")

    mse = MSELoss()
    average_psnr = metrics.AverageMetric(metrics.PSNR(), batches=len(tepoch))
    average_loss = metrics.AverageMetric(MSELoss(), batches=len(tepoch))

    print_every = 300
    start = time.time()
    curr = time.time()
    for inputs, _ in tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        if(time.time()-curr >= print_every):
            curr = time.time()
            log.print(f'Epoch {epoch+1}/{epochs}({100*average_loss.current/tepoch.total}%)')
            log.print(f'Elapsed Time: {curr-start}')

        inputs = inputs.to(device)
        outputs = inputs.clone()

        optimizer.zero_grad()

        output_images = model(inputs)
        loss = mse(output_images, outputs)
        avg_psnr = average_psnr(output_images, outputs).item()
        avg_loss = average_loss(output_images, outputs).item()

        loss.backward()
        optimizer.step()

        tepoch.set_postfix({"loss":average_loss, "pSNR":avg_psnr})
    
    return avg_psnr, avg_loss

def valid(model: Module, batches, device):
    average_loss = metrics.AverageMetric(MSELoss(), batches=len(batches))
    average_psnr = metrics.AverageMetric(metrics.PSNR(), batches=len(batches))
    with no_grad():
        for inputs, _ in iter(batches):
            inputs = inputs.to(device)
            outputs = inputs.clone()

            output_images = model(inputs)
            avg_psnr = average_psnr(output_images, outputs).item()
            avg_loss = average_loss(output_images, outputs).item()

        return avg_psnr, avg_loss


def start_session(model: Module, epochs, batch_size, save_dir, data_dir):
    save_gpu_stats(save_dir)

    # does get cuda:0
    base_device = "cuda:0" if cuda.is_available() else "cpu"
    devices = [i for i in range(cuda.device_count())]

    model = model.to(base_device)
    model.train()

    param_count = model_requirements.get_parameters(model)
    param_doc = Printer(save_dir, name="model_info", mode='w')
    param_doc.print(f'total parameters: {param_count}')
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    mode = 'w'
    if(os.path.exists(f'{save_dir}/checkpoint.pt')):
        checkpoint = load(f'{save_dir}/checkpoint.pt')
        start_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        mode = 'a'
    
    model = DataParallel(model, device_ids=devices)

    dataset = imagenet.IN(data_dir)
    trainloader = DataLoader(dataset.trainset, batch_size=batch_size, shuffle=dataset.shufflemode)
    train_len = len(dataset.trainset)
    train_batches = ceil(train_len/batch_size)

    validloader = DataLoader(dataset.validset, batch_size=batch_size, shuffle=False)
    valid_len = len(dataset.validset)
    valid_batches = ceil(valid_len/batch_size)

    testloader = DataLoader(dataset.testset, batch_size=batch_size, shuffle=False)
    test_len = len(dataset.testset)
    test_batches = ceil(test_len/batch_size)
    
    status = Status(save_dir)
    training_log = MetricLogger(save_dir, name='train', size=train_batches, mode=mode)
    valid_log = MetricLogger(save_dir, name='valid', size=valid_batches, mode=mode)
    test_log = MetricLogger(save_dir, name="test", size=test_batches, mode=mode)

    for epoch in range(start_epoch, epochs):
        with tqdm(trainloader, unit="batch") as tepoch:
            avg_psnr, avg_loss = train(model, optimizer, tepoch, save_dir, base_device, epoch, epochs)
            v_avg_psnr, v_avg_loss = valid(model, validloader, base_device)

            training_log.put(epoch, avg_loss, avg_psnr)
            valid_log.put(epoch, v_avg_loss, v_avg_psnr)

        save({
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optim': optimizer.state_dict()
        }, f'{save_dir}/checkpoint.pt')

    model.eval()
    tst_avg_psnr, tst_avg_loss = valid(model, testloader, base_device)

    status.print(f'Loss: {tst_avg_loss}, PSNR: {tst_avg_psnr}')
    test_log.put(0, tst_avg_loss, tst_avg_psnr)

    save({
        'encoder': model.module.encoder.state_dict(),
        'decoder': model.module.decoder.state_dict()
    }, f'{save_dir}/final_model.pt')

    os.remove(f'{save_dir}/checkpoint.pt')


