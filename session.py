from argparse import ArgumentParser

from models import ConvCompression, SwinCompression
from train import start_session

class Args():
    epochs: int = 20
    batch_size: int = 8
    model_type: str 
    subset: int = 1

    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("-e", "--epochs", dest="epochs", help="Number of epochs to be run", type=int)
        parser.add_argument("-b", "--batch", dest="batch_size", help="Size of data batches", type=int)
        parser.add_argument("-m", "--model", dest="model_type", help="Which model to run, swin or conv", type=str, required=True)
        parser.add_argument("-s", "--subset", dest="subset", help="Reduction factor of imagenet dataset", type=int)

        args = vars(parser.parse_args())

        if(args["epochs"] != None):
            self.epochs = args["epochs"]

        if(args["batch_size"] != None):
            self.batch_size = args["batch_size"]

        if(args["subset"] != None):
            self.subset = args["subset"]

        self.model_type = args["model_type"]

if __name__ == "__main__":
    args = Args()

    if(args.model_type == "swin"):
        compressor = SwinCompression.FullSwinCompressor(embed_dim=48, transfer_dim=16, patch_size=[2,2], depths=[4,4,6,4], num_heads=[4,4,4,4,4], window_size=[4,4])
    elif(args.model_type == "conv"):
        compressor = ConvCompression.FullConvConvCompressor(embed_dim=128, transfer_dim=16, reduction_layers=4)

    start_session(model=compressor, epochs=args.epochs, batch_size=args.batch_size, subset=args.subset)

# --model swin --epochs 5 --batch 128