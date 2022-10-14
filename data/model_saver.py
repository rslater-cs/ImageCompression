import os
import torch

def save_model(encoder, decoder, type):
    root = ".\\saved_models\\"
    encoder_name = "{}_encoder".format(type)
    decoder_name = "{}_decoder".format(type)
    networks = [(os.path.join(root, name), int(name.split("_")[1]))\
        for name in os.listdir(".\\saved_models\\") if os.path.isdir(os.path.join(root, name))]

    networks.sort(key= lambda x: x[1])

    if(len(networks) == 0):
        max_id = -1
    else:
        max_id = networks[-1][1]

    network_id = max_id+1

    path = os.path.join(root, "{}_{}".format(type, network_id))

    if(not os.path.exists(path)):
        os.mkdir(path)

    encoder_path = os.path.join(path, "{}{}".format(encoder_name, ".pth"))
    decoder_path = os.path.join(path, "{}{}".format(decoder_name, ".pth"))

    torch.save(encoder, encoder_path)
    torch.save(decoder, decoder_path)

    return path