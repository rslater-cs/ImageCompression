from ctypes import resize
from tkinter import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import numpy as np

# ImageCompression -> AI -> Python -> Programming
REL_PATH_TO_DATASETS = "../../../"

class IN():

    def __init__(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.trainset = ImageFolder(root=REL_PATH_TO_DATASETS+'Datasets/ImageNet/train', transform=transform)
        # self.validset = ImageNet(root='E:\Programming\Datasets\train_blurred', train=False, download=True, transform=transform)
        self.shufflemode = True