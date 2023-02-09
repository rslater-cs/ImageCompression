import os
from pathlib import Path
import torch
import shutil

NETWORK_NAME = 'SwinCompression'

def make_path(data_dir):
    if(not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    else:
        os.rmdir(data_dir)
        os.mkdir(data_dir)

    return Path(data_dir)


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