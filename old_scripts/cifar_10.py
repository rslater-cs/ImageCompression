from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

class CIFAR():

    def __init__(self):
        transform = ToTensor()
        self.trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.validset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.shufflemode = True