from train import valid
from data_scripts import imagenet
from data_scripts.loggers.metrics import MetricLogger
import torch
from torch.utils.data import DataLoader
from models.SwinCompression import PublishedCompressor
from argparse import ArgumentParser, ArgumentTypeError
import os

expand_hyperparameters = {
    'e':'embed_dim',
    't':'transfer_dim',
    'w':'window_size',
    'd':'depth'
}

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise ArgumentTypeError(f"Path does not exist: {path}")
    
def get_hyperparameters(dir: str):
    folder = dir.split(os.sep)[-1]
    segments = folder.split('_')[1:]

    parameters = dict()
    for segment in segments:
        parameters[expand_hyperparameters[segment[0]]] = int(segment[1:])

    return parameters

# Load model parameters into quantised version of compression network, compare the quantisation vs no quantisation psnr value
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_dir", dest="model_dir", help="The location of the model to be teted", type=str)
    parser.add_argument("-i", "--imagenet", dest="imagenet", help="The location where imagenet is stored", type=str)

    args = vars(parser.parse_args())

    params = get_hyperparameters(args['model_dir'])

    depths = [2]*params['depth']
    depths[-2] = 6

    heads = [4]*params['depth']

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = PublishedCompressor(embed_dim=params['embed_dim'], 
        transfer_dim=params['transfer_dim'], 
        patch_size=[2,2], 
        depths=depths, 
        num_heads=heads, 
        window_size=[params['window_size'], params['window_size']], 
        dropout=0.5)
    model.requires_grad_(False)
    model = model.to(device)

    model_params = torch.load(f'{args["model_dir"]}/final_model.pt', map_location=device)

    model.encoder.load_state_dict(model_params['encoder'])
    model.decoder.load_state_dict(model_params['decoder'])
    model.eval()
    model.requires_grad_(False)

    dataset = imagenet.IN(args['imagenet']).testset
    dataloader = DataLoader(dataset, 32, shuffle=False)

    avg_psnr, avg_loss = valid(model, dataloader, device)

    test_log = MetricLogger(args['model_dir'], name="test_quantised", mode='w')
    test_log.put(0, avg_loss, avg_psnr)




