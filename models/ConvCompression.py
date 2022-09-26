from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Module, Conv2d, BatchNorm2d, BatchNorm1d, Dropout, ConvTranspose2d, MaxPool2d, Tanh, LeakyReLU, Flatten

class ConvCompression(Module):

    def __init__(self):
        super(ConvCompression, self).__init__()

        self.encoder = Sequential(
            self.make_reduction_layer(3, 32),
            self.make_reduction_layer(32, 64),
            self.make_reduction_layer(64, 256),
            self.make_reduction_layer(256, 32),
        )

        self.midpoint = Sequential(
            Flatten(),
            Linear(5*5*16, 100),
            ReLU()
        )
        
        self.decoder = Sequential(
            self.make_generative_layer(32, 256),
            self.make_generative_layer(256, 64),
            self.make_generative_layer(64, 32),
            self.make_head(32),
        )

    def make_reduction_layer(self, in_channels, out_channels=32):
        return Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            ReLU(),
            BatchNorm2d(out_channels),
        )

    def make_generative_layer(self, in_channels, out_channels=32):
        return Sequential(
            ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            ReLU(),
            BatchNorm2d(out_channels),
        )

    def make_head(self, in_channels):
        return Sequential(
            ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            ReLU(),
            BatchNorm2d(in_channels),
            Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            Tanh()
        )

    def forward(self, x):

        x = self.encoder(x)

        # x = self.midpoint(x)
        # x = x.view((-1,1,10,10))
        x = self.decoder(x)

        return x
