from turtle import forward
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Module, Conv2d, BatchNorm2d, BatchNorm1d, Dropout, ConvTranspose2d, MaxPool2d, Tanh, LeakyReLU

class ConvCompression(Module):

    def __init__(self):
        super(ConvCompression, self).__init__()

        self.reduction_layer1 = self.make_reduction_layer(3, 32)
        self.reduction_layer2 = self.make_reduction_layer(32, 128)
        self.reduction_layer3 = self.make_reduction_layer(128, 16)

        self.compression_head = Sequential(
            Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Dropout(0.5),
            Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(32),
            Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            ReLU(),
        )

        self.gen_layer1 = self.make_generative_layer(16, 128)
        self.gen_layer2 = self.make_generative_layer(128, 64)
        self.head = self.make_head(64)

    def make_reduction_layer(self, in_channels, out_channels=32):
        return Sequential(
            # Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            # ReLU(),
            # Dropout(0.5),
            # BatchNorm2d(out_channels),
            # Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            # ReLU(),
            # Dropout(0.5),
            # BatchNorm2d(out_channels),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Dropout(0.5),
            BatchNorm2d(out_channels),
        )

    def make_generative_layer(self, in_channels, out_channels=32):
        return Sequential(
            ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            LeakyReLU(),
            Dropout(0.5),
            BatchNorm2d(out_channels),
        )

    def make_head(self, in_channels):
        return Sequential(
            ConvTranspose2d(in_channels=in_channels, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            Tanh()
        )

    def forward(self, x):
        x = self.reduction_layer1(x)
        x = self.reduction_layer2(x)
        x = self.reduction_layer3(x)
        # x = self.compression_head(x)

        x = self.gen_layer1(x)
        x = self.gen_layer2(x)
        x = self.head(x)

        return x
