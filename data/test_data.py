from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch

# PATCH_WIDTH = 384
# PATCH_HEIGHT = 216

class TestSet(Dataset):
    def __init__(self, patch_width, patch_height):
        test_image = Image.open("data/test_data/monkey.jpg")
        PILtoTensor = transforms.ToTensor()
        test_image = PILtoTensor(test_image)

        if(test_image.shape[2] // patch_width != test_image.shape[2] / patch_width or test_image.shape[1] // patch_height != test_image.shape[1] / patch_height):
            raise Exception("Patch size does not create even patches")

        patch_x = test_image.shape[2] // patch_width
        patch_y = test_image.shape[1] // patch_height
        self.patches = torch.zeros([patch_x*patch_y, test_image.shape[0], patch_height, patch_width], dtype=torch.float32)

        for i in range(patch_x):
            for j in range(patch_y):
                self.patches[i+j*patch_x] = test_image[:, j*patch_height:j*patch_height+patch_height, i*patch_width:i*patch_width+patch_width]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        return self.patches[index]