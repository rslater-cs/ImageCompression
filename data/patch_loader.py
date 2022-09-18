from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from pathlib import Path
import os
import os.path
import time
import sys
import random

PATH_ROOT = Path(os.path.curdir) / "data" / "patches"

class PatchSet(Dataset):
    def __init__(self, patch_size):
        self.PILtoTensor = transforms.ToTensor()
        self.path = PATH_ROOT / "{}_{}".format(patch_size[0], patch_size[1])

        self.content = []

        print("Loading Started")

        for root, dir, files in os.walk(self.path):
            if(len(dir) == 0):
                for file in files:
                    self.content.append(Path(root) / Path(file))

        random.shuffle(self.content)

        print("Loading Complete")

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        return self.PILtoTensor(Image.open(self.content[index]))