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

PATH_ROOT = Path(os.path.curdir) / "data" / "movies"
FRAME_CACHE_SIZE = 350

class PatchSet(Dataset):
    def __init__(self, frame_size, patch_size, movie_path):
        self.toTensor = transforms.ToTensor()
        self.path = PATH_ROOT
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.patches_per_frame = (frame_size[0] // patch_size[0])**2
        self.patch_cache_size = FRAME_CACHE_SIZE*self.patches_per_frame
        self.patch_x = frame_size[0] // patch_size[0]
        self.patch_y = frame_size[1] // patch_size[1]
        self.progress = 0
        self.cache_refreshes = 0

        self.length = 0

        print("Loading Started")

        clip_path = Path(PATH_ROOT) / Path(movie_path)
        self.movie_names.append(clip_path)

        reader = imageio.get_reader(clip_path)
        self.content.append(reader)

        cap = cv2.VideoCapture(str(clip_path))
                    
        self.length += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        self.length = int(self.length)

        self.cache = torch.zeros((self.patch_cache_size, 3, self.patch_size[1], self.patch_size[0]))
        self.load_patches()

        print("Total Samples", (self.patch_x**2)*self.length)

        print("Loading Complete")

    def load_patches(self):
        size = min(self.length-self.progress, FRAME_CACHE_SIZE)
        i = self.progress
        j = 0
        
        while(i-(FRAME_CACHE_SIZE*self.cache_refreshes) < size and i-(FRAME_CACHE_SIZE*self.cache_refreshes) < self.lengths[self.current_movie]):
            frame = self.toTensor(self.content[self.current_movie].get_data(i))

            for l in range(self.patch_x):
                for k in range(self.patch_y):
                    # print("j", j, "l", l, "k", k, "i", i)
                    self.cache[j] = frame[:, k*self.patch_size[1]:k*self.patch_size[1]+self.patch_size[1], l*self.patch_size[0]:l*self.patch_size[0]+self.patch_size[0]]
                    j += 1

            i += 1

    def __len__(self):
        return (self.patch_x**2)*self.length

    def __getitem__(self, index):
        if(index-(self.cache_refreshes*self.patch_cache_size) > self.patch_cache_size):
            self.cache_refreshes += 1
            self.load_patches()

        return self.cache[index % self.patch_cache_size]