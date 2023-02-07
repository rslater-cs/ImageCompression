from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

class IN():

    def __init__(self, root):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((112, 112)),
            ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.trainset = ImageFolder(root=root, transform=transform)
        # self.validset = ImageNet(root='E:\Programming\Datasets\train_blurred', train=False, download=True, transform=transform)

        self.trainset = Subset(self.trainset, list(range(100_000)))
        self.shufflemode = True