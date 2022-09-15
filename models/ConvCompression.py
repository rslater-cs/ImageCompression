from turtle import forward
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Module, Conv2d, BatchNorm2d, BatchNorm1d, Dropout, ConvTranspose2d

class ConvCompression(Module):

    def __init__(self):
        super(ConvCompression, self).__init__()

        self.reduction_layer1 = self.make_reduction_layer(3, 128)
        self.reduction_layer2 = self.make_reduction_layer(128, 64)
        self.reduction_layer3 = self.make_reduction_layer(64, 32)

        self.gen_layer1 = self.make_generative_layer(32, 128)
        self.gen_layer2 = self.make_generative_layer(128, 64)
        self.gen_layer3 = self.make_generative_layer(64, 3)

    def make_reduction_layer(self, in_channels, out_channels=32):
        return Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(out_channels),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Dropout(0.5),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
            ReLU()
        )

    def make_generative_layer(self, in_channels, out_channels=32):
        return Sequential(
            ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(out_channels),
            ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Dropout(0.5),
            ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
            ReLU()
        )

    def forward(self, x):
        x = self.reduction_layer1(x)
        x = self.reduction_layer2(x)
        x = self.reduction_layer3(x)

        x = self.gen_layer1(x)
        x = self.gen_layer2(x)
        x = self.gen_layer3(x)

        return x
