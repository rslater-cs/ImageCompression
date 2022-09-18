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
PATCH_CACHE_SIZE = 100000

class PatchSet(Dataset):
    def __init__(self, patch_size):
        self.PILtoTensor = transforms.ToTensor()
        self.path = PATH_ROOT / "{}_{}".format(patch_size[0], patch_size[1])
        self.patch_size = patch_size
        self.progress = 0

        self.content = []

        print("Loading Started")

        for root, dir, files in os.walk(self.path):
            if(len(dir) == 0):
                for file in files:
                    self.content.append(Path(root) / Path(file))

        random.shuffle(self.content)

        self.cache = torch.zeros((PATCH_CACHE_SIZE, 3, self.patch_size[1], self.patch_size[0]))
        self.load_patches(self.progress)

        print("Loading Complete")

    def load_patches(self, start):
        size = min(len(self.content)-start, PATCH_CACHE_SIZE)
        for i in range(size):
            self.cache[i] = self.PILtoTensor(Image.open(self.content[start+i]))

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        if(index > self.progress+PATCH_CACHE_SIZE):
            self.progress += PATCH_CACHE_SIZE
            self.load_patches(self.progress)

        return self.cache[index % PATCH_CACHE_SIZE]