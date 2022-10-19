from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np

from pathlib import Path
import imageio
import cv2

BLOCK_SIZE = 100

class ImageSet():
    def __init__(self, image_folder):
        self.shufflemode = True
        
        toTensor = transforms.ToTensor()

        print("Loading Started")

        folder_path = Path(image_folder)

        self.trainset = ImageFolder(folder_path, transform=toTensor)

        print("Total Samples", len(self.images))

        print("Loading Complete")