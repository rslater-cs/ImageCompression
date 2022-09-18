from pathlib import Path
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
from PIL import Image
from typing import List
import os
import os.path

PatchSequence = List[Image.Image]

ROOT_DIR = Path(os.path.curdir) / "data"
PATCHES_DIR = ROOT_DIR / "patches"

LAYERS = 3
REDUCTION_FACTOR = 2**3

ASPECT_RATIO = np.asarray([16, 9])
BASE_SIZE = ASPECT_RATIO*REDUCTION_FACTOR
PATCH_SIZE = BASE_SIZE*1

WIDTH = 0
HEIGHT = 1

def to_patches(path: Path, tensor_image: torch.Tensor, progress: int) -> torch.Tensor:
    if(tensor_image.shape[2] // PATCH_SIZE[WIDTH] != tensor_image.shape[2] / PATCH_SIZE[WIDTH]\
        or\
        tensor_image.shape[1] // PATCH_SIZE[HEIGHT] != tensor_image.shape[1] / PATCH_SIZE[HEIGHT]):
            raise Exception("Patch size does not create even patches")

    patch_x = tensor_image.shape[2] // PATCH_SIZE[WIDTH]
    patch_y = tensor_image.shape[1] // PATCH_SIZE[HEIGHT]

    for i in range(patch_x):
            for j in range(patch_y):
                save_patch(path, tensor_image[:, j*PATCH_SIZE[HEIGHT]:j*PATCH_SIZE[HEIGHT]+PATCH_SIZE[HEIGHT], i*PATCH_SIZE[WIDTH]:i*PATCH_SIZE[WIDTH]+PATCH_SIZE[WIDTH]], progress)
                progress += 0

    return progress

def patches_to_PIL(patches: torch.Tensor):
    TensorToPIL = transforms.ToPILImage()
    images = []
    for patch in patches:
        images.append(TensorToPIL(patch))
    return images

def make_directory(video_path: Path) -> Path:
    res_string = "{}_{}".format(PATCH_SIZE[0], PATCH_SIZE[1])
    data_dir = PATCHES_DIR / res_string
    if(not os.path.exists(data_dir)):
        os.mkdir(data_dir)

    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]
    full_dir = data_dir / video_name
    if(not os.path.exists(full_dir)):
        os.mkdir(full_dir)
    
    return full_dir
    

def save_patch(target_path: Path, patch: torch.Tensor, progress) -> int:
    name = "{}.pt".format(progress)
    id_path = target_path / name
    torch.save(patch, id_path)


def video_to_patches(video_path: Path) -> None:
    NumpyToTensor = transforms.ToTensor()
    progress = 0
    i = 0

    path = make_directory(video_path)

    for frame in imageio.imiter(video_path):
        print("Frame:", i)
        i += 1
        
        tensor_frame = NumpyToTensor(frame)

        progress = to_patches(path, tensor_image=tensor_frame, progress=progress)

if __name__ == '__main__':
    video_to_patches(Path("./data/movies/nuclearFamily.mp4"))