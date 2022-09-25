import torch
from torchvision import transforms
from models.ConvCompression import ConvCompression
import imageio
import cv2
import random
from PIL import Image
import numpy as np
from data.postprocessing import fix_bounds

PATCH_SIZE = (80, 80)
FRAME_SIZE = (1280, 720)
ToImage = transforms.ToPILImage()
ToTensor = transforms.ToTensor()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_random_patch(frame: torch.Tensor) -> torch.Tensor:
    limit_x = frame.shape[2] // PATCH_SIZE[0]
    limit_y = frame.shape[1] // PATCH_SIZE[1]
    patch_x = random.randint(0, limit_x-1)
    patch_y = random.randint(0, limit_y-1)
    print(patch_y*PATCH_SIZE[1], "-", (patch_y+1)*PATCH_SIZE[1])
    print(patch_x*PATCH_SIZE[0], "-", (patch_x+1)*PATCH_SIZE[0])

    return frame[:, patch_y*PATCH_SIZE[1]:(patch_y+1)*PATCH_SIZE[1], patch_x*PATCH_SIZE[0]:(patch_x+1)*PATCH_SIZE[0]]

def get_random_frame(path):
    movie = imageio.get_reader(path)
    cap = cv2.VideoCapture(str(path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    index = random.randint(0, length-1)

    frame = ToTensor(movie.get_data(index))

    return frame

def to_patches(frame: torch.Tensor) -> torch.Tensor:
    width = frame.shape[2]
    height = frame.shape[1]

    x_patches = width // PATCH_SIZE[0]
    y_patches = height // PATCH_SIZE[1]

    patches = torch.empty((x_patches*y_patches, 3, PATCH_SIZE[1], PATCH_SIZE[0]))

    print(patches.shape)

    x = 0

    for i in range(x_patches):
        for j in range(y_patches):
            patches[x] = frame[:, j*PATCH_SIZE[1]:j*PATCH_SIZE[1]+PATCH_SIZE[1], i*PATCH_SIZE[0]:i*PATCH_SIZE[0]+PATCH_SIZE[0]]
            x += 1

    return patches

def to_frame(patches: torch.Tensor) -> torch.Tensor:
    width = FRAME_SIZE[0]
    height = FRAME_SIZE[1]

    x_patches = width // PATCH_SIZE[0]
    y_patches = height // PATCH_SIZE[1]

    frame = torch.empty((3, height, width))

    x = 0

    for i in range(x_patches):
        for j in range(y_patches):
            frame[:, j*PATCH_SIZE[1]:j*PATCH_SIZE[1]+PATCH_SIZE[1], i*PATCH_SIZE[0]:i*PATCH_SIZE[0]+PATCH_SIZE[0]] = patches[x]
            x += 1

    return frame

model = torch.load(".\\saved_models\\compressionnet_1\\compressionnet.pth").to(device)
print(model)
model.eval()

frame = get_random_frame(".\\data\\movies\\nuclearFamily.mp4")
print(frame.shape)
patches = to_patches(frame)

print(patches.shape)
patches = patches.to(device)
print(patches.shape)
output = model(patches)
print(output.shape)

output = fix_bounds(output)
output_frame = to_frame(output)
original_frame = ToImage(frame)
image_output = ToImage(output_frame)

Image.fromarray(np.hstack((np.array(original_frame),np.array(image_output)))).show()


