from unittest.mock import patch
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
import imageio
import cv2

PATH_ROOT = Path(os.path.curdir) / "data"
FRAME_CACHE_SIZE = 2

class PatchSet(Dataset):
    def __init__(self, frame_size, patch_size, movie_path):
        self.toTensor = transforms.ToTensor()
        self.path = PATH_ROOT
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.patch_x = frame_size[0] // patch_size[0]
        self.patch_y = frame_size[1] // patch_size[1]
        self.patches_per_frame = self.patch_x*self.patch_y
        self.patch_cache_size = FRAME_CACHE_SIZE*self.patches_per_frame
        self.progress = 0
        self.cache_refreshes = 0

        print("Loading Started")

        clip_path = Path(PATH_ROOT) / Path(movie_path)

        self.movie = imageio.get_reader(clip_path)

        cap = cv2.VideoCapture(str(clip_path))
                    
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.length = int(self.length)

        self.cache = torch.zeros((self.patch_cache_size, 3, self.patch_size[1], self.patch_size[0]))
        self.load_patches()

        print("Total Samples", self.patches_per_frame*self.length)

        print("Loading Complete")

    def load_patches(self):
        size = min(self.length-self.progress, FRAME_CACHE_SIZE)
        indexes = torch.randperm(size*self.patches_per_frame)
        i = self.progress
        j = 0
        
        while(i-(FRAME_CACHE_SIZE*self.cache_refreshes) < size):
            frame = self.toTensor(self.movie.get_data(i))

            for l in range(self.patch_x):
                for k in range(self.patch_y):
                    self.cache[indexes[j]] = frame[:, k*self.patch_size[1]:k*self.patch_size[1]+self.patch_size[1], l*self.patch_size[0]:l*self.patch_size[0]+self.patch_size[0]]
                    j += 1

            i += 1

        self.progress = i

    def __len__(self):
        return (self.patches_per_frame*self.length)

    def __getitem__(self, index):
        patch = self.toTensor(self.movie.get_data(index))

        return patch, patch.clone()