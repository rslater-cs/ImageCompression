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

import os
import time

# training loop
def train(model: Module, optimizer: optim.Optimizer, tepoch: tqdm, path, device, epoch: int, epochs: int):
    log = Printer(path, name = "current_epoch")

    # Setup loss and averaging metrics
    mse = MSELoss()
    average_psnr = metrics.AverageMetric(metrics.PSNR(), batches=len(tepoch))
    average_loss = metrics.AverageMetric(MSELoss(), batches=len(tepoch))

    print_every = 300
    start = time.time()
    curr = time.time()
    for inputs, _ in tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        # Logs progress every 5 minutes
        if(time.time()-curr >= print_every):
            curr = time.time()
            log.print(f'Epoch {epoch+1}/{epochs}({100*average_loss.current/tepoch.total}%)')
            log.print(f'Elapsed Time: {curr-start}')

        inputs = inputs.to(device)
        outputs = inputs.clone()

        optimizer.zero_grad()

        # Calculate loss and metrics
        output_images = model(inputs)
        loss = mse(output_images, outputs)
        avg_psnr = average_psnr(output_images, outputs)
        avg_loss = average_loss(output_images, outputs)

        loss.backward()
        optimizer.step()

        tepoch.set_postfix({"loss":avg_loss, "pSNR":avg_psnr})
    
    return avg_psnr, avg_loss

# validation loop
def valid(model: Module, batches, device):
    # setup average metrics
    average_loss = metrics.AverageMetric(MSELoss(), batches=len(batches))
    average_psnr = metrics.AverageMetric(metrics.PSNR(), batches=len(batches))
    with no_grad():
        for inputs, _ in iter(batches):
            inputs = inputs.to(device)
            outputs = inputs.clone()

            output_images = model(inputs)
            avg_psnr = average_psnr(output_images, outputs)
            avg_loss = average_loss(output_images, outputs)

        return avg_psnr, avg_loss


def start_session(model: Module, epochs, batch_size, save_dir, data_dir):
    # log gpu informations
    save_gpu_stats(save_dir)

    # check if GPU exists and get all available GPUs
    base_device = "cuda:0" if cuda.is_available() else "cpu"
    devices = [i for i in range(cuda.device_count())]

    # model needs to be in a base device for multi-GPU support
    model = model.to(base_device)
    model.train()

    # log the model parameter count
    param_count = model_requirements.get_parameters(model)
    param_doc = Printer(save_dir, name="model_info", mode='w')
    param_doc.print(f'total parameters: {param_count}')
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # if a checkpoint exists then continue the training cycle from the last step
    start_epoch = 0
    mode = 'w'
    if(os.path.exists(f'{save_dir}/checkpoint.pt')):
        checkpoint = load(f'{save_dir}/checkpoint.pt')
        start_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        mode = 'a'
    
    # setup model for use with multiple GPUs
    model = DataParallel(model, device_ids=devices)

    # load imagenet
    dataset = imagenet.IN(data_dir)

    trainloader = DataLoader(dataset.trainset, batch_size=batch_size, shuffle=dataset.shufflemode)
    validloader = DataLoader(dataset.validset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(dataset.testset, batch_size=batch_size, shuffle=False)
    
    # setup logging
    status = Status(save_dir)
    training_log = MetricLogger(save_dir, name='train', mode=mode)
    valid_log = MetricLogger(save_dir, name='valid', mode=mode)
    test_log = MetricLogger(save_dir, name="test", mode=mode)

    # Training cycle for certain amount of epochs
    for epoch in range(start_epoch, epochs):
        with tqdm(trainloader, unit="batch") as tepoch:
            avg_psnr, avg_loss = train(model, optimizer, tepoch, save_dir, base_device, epoch, epochs)
            v_avg_psnr, v_avg_loss = valid(model, validloader, base_device)

            # log model performance
            training_log.put(epoch, avg_loss, avg_psnr)
            valid_log.put(epoch, v_avg_loss, v_avg_psnr)

        # save a checkpoint at end of every epoch
        save({
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optim': optimizer.state_dict()
        }, f'{save_dir}/checkpoint.pt')

    # get test metrics 
    model.eval()
    tst_avg_psnr, tst_avg_loss = valid(model, testloader, base_device)

    status.print(f'Loss: {tst_avg_loss}, PSNR: {tst_avg_psnr}')
    test_log.put(0, tst_avg_loss, tst_avg_psnr)

    # save final version of the encoder and decoder
    save({
        'encoder': model.module.encoder.state_dict(),
        'decoder': model.module.decoder.state_dict()
    }, f'{save_dir}/final_model.pt')

    # remove checkpoint file so training won't be continued if model is to be trained again from the begining
    os.remove(f'{save_dir}/checkpoint.pt')


