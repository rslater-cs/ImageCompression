from argparse import ArgumentParser, ArgumentTypeError
import os

from models import SwinCompression
from train import start_session
from model_scripts.data_saver import make_path

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise ArgumentTypeError(f"Path does not exist: {path}")

class Args():
    epochs: int = 20
    batch_size: int = 8

    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("-s", "--save_dir", dest="save_dir", help="The location where models should be saved", type=dir_path)
        parser.add_argument("-i", "--imagenet", dest="imagenet_dir", help="The location that imagenet is stored", type=dir_path)

        parser.add_argument("-e", "--epochs", dest="epochs", help="Number of epochs to be run", type=int)
        parser.add_argument("-b", "--batch", dest="batch_size", help="Size of data batches", type=int)

        parser.add_argument("-m", "--embed", dest="embed_dim", help="What dimensionality to embed the patches into", type=int)
        parser.add_argument("-t", "--transfer", dest="transfer_dim", help="The dimensionality of the compressed features", type=int)
        parser.add_argument("-w", "--window", dest="window_size", help="The size of the attension windows", type=int)
        parser.add_argument("-d", "--depth", dest="depth", help="The amount of reduction layers", type=int)

        self.args = vars(parser.parse_args())

if __name__ == "__main__":
    args = Args().args

    depths = [2]*args['depth']
    depths[-2] = 6

    heads = [4]*args['depth']

    print(depths)    

    compressor = SwinCompression.FullSwinCompressor(embed_dim=args['embed_dim'], 
        transfer_dim=args['transfer_dim'], 
        patch_size=[2,2], 
        depths=depths, 
        num_heads=heads, 
        window_size=[args['window_size'], args['window_size']], 
        dropout=0.5)

    full_save_path = f'{args["save_dir"]}/SwinCompression_e{args["embed_dim"]}_t{args["transfer_dim"]}_w{args["window_size"]}_d{args["depth"]}'
    full_save_path = make_path(full_save_path)

    start_session(model=compressor, epochs=args['epochs'], batch_size=args['batch_size'], save_dir=full_save_path, data_dir=args['imagenet_dir'])

# session.py -s ./saved_models -i ./ -e 1 -b 1 -m 12 -t 16 -w 2 -d 3
# Need to create a script to setup restoration from checkpoint