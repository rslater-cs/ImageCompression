import os
from pathlib import Path
import torch
import shutil

def get_path(type):
    root = ".\\saved_models\\"
    networks = [(os.path.join(root, name), int(name.split("_")[1]))\
        for name in os.listdir(".\\saved_models\\") if (os.path.isdir(os.path.join(root, name)) and type in name)]

    networks.sort(key= lambda x: x[1])

    if(len(networks) == 0):
        max_id = -1
    else:
        max_id = networks[-1][1]

    network_id = max_id+1

    path = os.path.join(root, "{}_{}".format(type, network_id))

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

    code_copy_path = Path(".\\models") / "{}.py".format(model.network_type)
    code_paste_path = path / "{}.py".format(model.network_type)

    if(not os.path.exists(code_paste_path)):
        shutil.copyfile(code_copy_path, code_paste_path)

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)

    return path