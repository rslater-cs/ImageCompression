import os
import torch

def save_model(model):
    root = ".\\saved_models\\"
    basename = "compressionnet"
    networks = [(os.path.join(root, name), int(name.split("_")[1]))\
        for name in os.listdir(".\\saved_models\\") if os.path.isdir(os.path.join(root, name))]

    networks.sort(key= lambda x: x[1])

    if(len(networks) == 0):
        max_id = -1
    else:
        max_id = networks[-1][1]

    network_id = max_id+1

    path = os.path.join(root, "{}_{}".format(basename, network_id))

    if(not os.path.exists(path)):
        os.mkdir(path)

    path = os.path.join(path, "{}{}".format(basename, ".pth"))

    torch.save(model, path)

    return path