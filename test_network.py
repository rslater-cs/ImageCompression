from argparse import ArgumentParser, ArgumentTypeError
import os
import torch
from torchvision import transforms
from models.SwinCompression import Quantise8, DeQuantise8
from data_scripts import imagenet
from models import SwinCompression
# from models import SwinCompression_old as SwinCompression
from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq
from typing import Tuple
from PIL import Image

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
    
def preprocess(image: torch.Tensor):
    image[image > 1.0] = 1.0
    image[image < 0.0] = 0.0

    return image
    
def compress_image(image: torch.Tensor, transform: torch.nn.Module, folder_name: str = 'image_folder', filename: str = 'image'):
    dir = f'./saved_images/{folder_name}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    quantise = Quantise8()
    encoder = RangeEncoder(f'{dir}/{filename}_message.bin')

    transformed_image = transform(image)
    shape = transformed_image.shape

    quantised_image, min_x, max_x = quantise(transformed_image)
    quantised_image = quantised_image.reshape(quantised_image.shape[1])

    frequency_table: torch.Tensor = quantised_image.bincount()
    probability_table = frequency_table/torch.sum(frequency_table)
    probability_table = probability_table.tolist()
    cum_freq = prob_to_cum_freq(probability_table, quantised_image.shape[0])
    data = quantised_image.tolist()
    encoder.encode(data, cum_freq)
    encoder.close()

    torch.save(
        {
            'freq': frequency_table,
            'min': min_x,
            'max': max_x
        },
        f'{dir}/{filename}_metadata.pt')
    
    return shape, f'{dir}/{filename}'
    
def decompress_image(transform: torch.nn.Module, shape: torch.Size, dir: str, device) -> torch.Tensor:
    decoder = RangeDecoder(f'{dir}_message.bin')
    dequanise = DeQuantise8()

    metadata = torch.load(f'{dir}_metadata.pt', map_location=device)
    frequency_table = metadata['freq']
    min_x = metadata['min']
    max_x = metadata['max']

    data_length = torch.sum(frequency_table)
    probability_table = frequency_table/data_length
    probability_table = probability_table.tolist()
    cum_freq = prob_to_cum_freq(probability_table, data_length)

    data = decoder.decode(data_length, cum_freq)
    decoder.close()

    quantised_data = torch.tensor([data]).to(device)
    dequantised_data = dequanise(quantised_data, min_x, max_x, shape)
    
    transformed_image = transform(dequantised_data)

    return transformed_image

def no_quantisation(image: torch.Tensor, encoder: torch.nn.Module, decoder: torch.nn.Module):
    features = encoder(image)
    decoded_image = decoder(features)

    return decoded_image

def get_image(dataset) -> Tuple[torch.Tensor, int]:
    index = torch.randint(low=0, high=len(dataset), size=(1,)).item()
    image = torch.unsqueeze(dataset[index][0], 0)
    return image, index

def save_images(original: torch.Tensor, reconstruct_q: torch.Tensor, reconstruct: torch.Tensor, dir):
    toImage = transforms.ToPILImage()

    B, C, H, W = original.shape
    original = original.reshape(C, H, W)
    reconstruct = reconstruct.reshape(C, H, W)
    reconstruct_q = reconstruct_q.reshape(C, H, W)

    reconstruct = preprocess(reconstruct)
    reconstruct_q = preprocess(reconstruct_q)

    original = toImage(original)
    reconstruct = toImage(reconstruct)
    reconstruct_q = toImage(reconstruct_q)

    original.save(f'{dir}_orig.png')
    reconstruct.save(f'{dir}_recon.png')
    reconstruct_q.save(f'{dir}_reconq.png')

def get_hyperparameters(dir: str):
    folder = dir.split(os.sep)[-1]
    segments = folder.split('_')[1:]

    parameters = dict()
    for segment in segments:
        parameters[expand_hyperparameters[segment[0]]] = int(segment[1:])

    return parameters

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_dir", dest="model_dir", help="The location of the model to be teted", type=dir_path)
    parser.add_argument("-s", "--seed", dest="manual_seed", help="The seed to be used when picking a random image", type=int, default=-1)
    parser.add_argument("-i", "--imagenet", dest="imagenet", help="The location where imagenet is stored", type=dir_path)
    parser.add_argument("-n", "--number_images", dest="images", help="How many images should be loaded", type=int)

    args = vars(parser.parse_args())

    params = get_hyperparameters(args['model_dir'])

    depths = [2]*params['depth']
    depths[-2] = 6

    heads = [4]*params['depth']

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = SwinCompression.FullSwinCompressor(embed_dim=params['embed_dim'], 
        transfer_dim=params['transfer_dim'], 
        patch_size=[2,2], 
        depths=depths, 
        num_heads=heads, 
        window_size=[params['window_size'], params['window_size']], 
        dropout=0.5)
    model.requires_grad_(False)
    model = model.to(device)

    model_params = torch.load(f'{args["model_dir"]}/final_model.pt', map_location=device)
    
    encoder_model = model.encoder
    encoder_model.load_state_dict(model_params['encoder'])
    
    decoder_model = model.decoder
    decoder_model.load_state_dict(model_params['decoder'])

    dataset = imagenet.IN(args['imagenet']).testset

    if args['manual_seed'] != -1:
        torch.manual_seed(args['manual_seed'])

    for i in range(args['images']):
        image, index = get_image(dataset)
        image = image.to(device)
        folder_name = f'e{params["embed_dim"]}_t{params["transfer_dim"]}_w{params["window_size"]}_d{params["depth"]}'
        shape, dir = compress_image(image, encoder_model, folder_name=folder_name, filename=f'image{i}_index{index}')
        new_image = decompress_image(decoder_model, shape, dir, device)
        new_image_nq = no_quantisation(image, encoder_model, decoder_model)
        save_images(image, new_image, new_image_nq, dir)



