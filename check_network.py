import torch
from torchvision import transforms
from models.ConvCompression import ConvCompression
import imageio
import cv2
import random
from PIL import Image
import numpy as np

PATCH_SIZE = (80, 80)
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

model = torch.load(".\\saved_models\\compressionnet.pth").to(device)
print(model)
model.eval()

frame = get_random_frame(".\\data\\movies\\nuclearFamily_Trim.mp4")
print(frame.shape)
patch = get_random_patch(frame)

print(patch.shape)
patch = patch.reshape((1, patch.shape[0], patch.shape[1], patch.shape[2]))
patch = patch.to(device)
print(patch.shape)
output = model(patch)
print(output.shape)

image_patch = ToImage(patch[0])
image_output = ToImage(output[0])

Image.fromarray(np.hstack((np.array(image_patch),np.array(image_output)))).show()


