from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

from pathlib import Path
import imageio
import cv2

BLOCK_SIZE = 100

class FrameSet(Dataset):
    def __init__(self, movie_path):
        self.shufflemode = False

        self.toTensor = transforms.ToTensor()

        print("Loading Started")

        clip_path = Path(movie_path)

        self.movie = imageio.get_reader(clip_path)

        cap = cv2.VideoCapture(str(clip_path))
                    
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.length = int(self.length)

        print("Total Samples", self.length)

        print("Loading Complete")

        self.ids = self.generate_shuffle()

    def generate_shuffle(self):
        chunck_amount = self.length // BLOCK_SIZE
        relative_length = chunck_amount * BLOCK_SIZE
        trimming = self.length - relative_length
        l_trim = trimming // 2
        r_trim = trimming - l_trim

        ids = list(range(l_trim, self.length-r_trim))
        ids = np.asarray(ids)
        ids = ids.reshape((BLOCK_SIZE, relative_length // BLOCK_SIZE))
        np.random.shuffle(ids)

        return ids.flatten()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        patch = self.toTensor(self.movie.get_data(self.ids[index]))

        return patch