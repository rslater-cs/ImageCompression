from distutils.sysconfig import customize_compiler
import torch
from torchvision import transforms
import imageio
import cv2
import random
from PIL import Image
import numpy as np
from data_processing import postprocessing
from data_loading import imagenet
from torch.utils.data import DataLoader
from saved_models.SwinCompression_4 import SwinCompression
from pathlib import Path

# MOVIE_PATH = "C:\\Users\\ryans\\OneDrive - University of Surrey\\Documents\\Computer Science\\Modules\\Year3\\FYP\\MoviesDataset\\DVU_Challenge\\Movies\\1024_576\\nuclearFamily.mp4"

ROOT_PATH = Path(".\\saved_models\\SwinCompression_4")

PATCH_SIZE = (1280, 720)
FRAME_SIZE = (1280, 720)
BATCH_SIZE = 3

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

def get_random_frames(path):
    movie = imageio.get_reader(path)
    cap = cv2.VideoCapture(str(path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    index = random.randint(0, length-1)

    frame = ToTensor(movie.get_data(index))

    frames = torch.empty((3, frame.shape[0], frame.shape[1], frame.shape[2]))

    frames[0] = frame
    
    for i in range(BATCH_SIZE-1):
        index += 1
        frames[i+1] = ToTensor(movie.get_data(index))

    return frames

compressor = SwinCompression.FullSwinCompressor(embed_dim=16, transfer_dim=16, patch_size=[2,2], depths=[2,2,2,6,2], num_heads=[2,2,2,2,2], window_size=[2,2])
compressor.encoder.load_state_dict(torch.load(ROOT_PATH / "encoder.pt"))
compressor.decoder.load_state_dict(torch.load(ROOT_PATH / "decoder.pt"))
compressor = compressor.to(device)
compressor.eval()

loader = DataLoader(imagenet.IN(portion=1).trainset, shuffle=True)
frame, _ = next(iter(loader))

print(frame.shape)

frame = frame.to(device)

print(frame.shape)

compressed = compressor.encoder(frame)

print(compressed.shape)

output = compressor.decoder(compressed)

print(output.shape)

output_frame = postprocessing.fix_bounds(output)
original_frame = ToImage(frame[0])
image_output = ToImage(output_frame[0])

Image.fromarray(np.hstack((np.array(original_frame),np.array(image_output)))).show()


