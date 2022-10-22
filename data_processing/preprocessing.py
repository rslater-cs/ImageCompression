from pathlib import Path
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
from PIL import Image
from typing import List
import os
import os.path
from time import time
import cv2

PatchSequence = List[Image.Image]

IMAGE_RES = (512, 288)

VIDEO_PATH = Path("C:\\Users\\ryans\\OneDrive - University of Surrey\\Documents\\Computer Science\\Modules\\Year3\\FYP\\MoviesDataset\\DVU_Challenge\\Movies\\1024_576")
IMAGE_PATH = Path("E:\\Programming\\Datasets\\MoviesDataset\\DVU_Challenge\\Movies\\{}_{}_IMS".format(IMAGE_RES[0], IMAGE_RES[1]))

def make_directory(video_path: Path, name) -> Path:
    full_path = video_path / name
    
    return full_path
    

def save_image(target_path: Path, frame, progress):
    print
    if(not os.path.exists(target_path)):
        os.makedirs(target_path)

    TensorToPIL = transforms.ToPILImage()
    name = "{}.jpg".format(progress)
    id_path = target_path / name

    Image.Image.save(TensorToPIL(frame), id_path)


def video_to_images(name) -> None:
    i = 0

    frame_path = VIDEO_PATH / "{}.mp4".format(name)
    image_path = IMAGE_PATH / name
    
    print("Building from path:", frame_path)
    print("Building in path:", image_path)
    
    cur_time = time()

    for frame in imageio.imiter(frame_path):
        if(i % 100 == 0):
            print("FPS:", 100/(time()-cur_time))
            cur_time = time()
            print("Frame:", i)

        frame = cv2.resize(frame, dsize=(512, 288), interpolation=cv2.INTER_AREA)

        save_image(image_path, frame, i)
        i += 1

if __name__ == '__main__':
    video_to_images("nuclearFamily")