from turtle import forward
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Module, Conv2d, BatchNorm2d, BatchNorm1d, Dropout, ConvTranspose2d, MaxPool2d

class ConvCompression(Module):

    def __init__(self):
        super(ConvCompression, self).__init__()

        self.reduction_layer1 = self.make_reduction_layer(3, 32)
        self.reduction_layer2 = self.make_reduction_layer(32, 64)
        self.reduction_layer3 = self.make_reduction_layer(64, 128)

        self.compression_head = Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            ReLU(),
        )

        self.gen_layer1 = self.make_generative_layer(16, 32)
        self.gen_layer2 = self.make_generative_layer(32, 64)
        self.gen_layer3 = self.make_generative_layer(64, 3)

    def make_reduction_layer(self, in_channels, out_channels=32):
        return Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
            # MaxPool2d(kernel_size=2, stride=2),
            ReLU()
        )

    def make_generative_layer(self, in_channels, out_channels=32):
        return Sequential(
            ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ReLU(),
            ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ReLU(),
            ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
            ReLU()
        )

    def forward(self, x):
        x = self.reduction_layer1(x)
        x = self.reduction_layer2(x)
        x = self.reduction_layer3(x)
        x = self.compression_head(x)

        x = self.gen_layer1(x)
        x = self.gen_layer2(x)
        x = self.gen_layer3(x)

        return x
