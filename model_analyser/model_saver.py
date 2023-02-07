import os
from pathlib import Path
import torch
import shutil

NETWORK_NAME = 'SwinCompression'

def get_path(data_dir):
    networks = [(os.path.join(data_dir, NETWORK_NAME), int(file.split("_")[1]))\
        for file in os.listdir(data_dir) if (os.path.isdir(os.path.join(data_dir, NETWORK_NAME)) and NETWORK_NAME in file)]

    # and type in name

    networks.sort(key= lambda x: x[1])

    if(len(networks) == 0):
        max_id = -1
    else:
        max_id = networks[-1][1]

    network_id = max_id+1

    path = os.path.join(data_dir, "{}_{}".format(NETWORK_NAME, network_id))

    if(not os.path.exists(path)):
        os.mkdir(path)

    return Path(path)


def save_model(model, path: Path, in_progress=False):
    if(in_progress):
        encoder_path = path / "encoder_progress.pt"
        decoder_path = path / "decoder_progress.pt"
    else:
        encoder_path = path / "encoder.pt"
        decoder_path = path / "decoder.pt"

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)

    return path