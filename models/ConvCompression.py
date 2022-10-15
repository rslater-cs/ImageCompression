from typing import List
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embed_dim: int, transfer_dim: int, reduction_layers: int):
        super().__init__()
        compress = []

        prev_dim = 3
        for i in range(reduction_layers):
            dim = embed_dim*2**i
            compress.append(self.make_layer(prev_dim, dim))
            prev_dim = dim

        self.compress = nn.Sequential(*compress)

        self.head = nn.Sequential(
            nn.Conv2d(prev_dim, transfer_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(transfer_dim)
        )

    def make_layer(self, in_channels, out_channels=32):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.compress(x)
        x = self.head(x)

        return x

class Decoder(nn.Module):
    def __init__(self, transfer_dim: int, embed_dim: int, reduction_layers: int):
        super().__init__()

        self.transfer = nn.Sequential(
            nn.Conv2d(transfer_dim, embed_dim*2**(reduction_layers-1), kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(embed_dim*2**(reduction_layers-1))
        )

        decompress: List[nn.Module] = []

        prev_dim = embed_dim*2**(reduction_layers-1)
        for i in range(reduction_layers-1):
            dim = embed_dim*2**(reduction_layers-i-2)
            print(dim)
            decompress.append(self.make_layer(prev_dim, dim))
            prev_dim = dim

        self.decompress = nn.Sequential(*decompress)

        self.head = self.make_head(prev_dim)

    def make_layer(self, in_channels, out_channels=32):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def make_head(self, in_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels//2),
            nn.Conv2d(in_channels=in_channels//2, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.transfer(x)
        x = self.decompress(x)
        x = self.head(x)

        return x

class FullConvConvCompressor(nn.Module):
    def __init__(self, embed_dim: int, transfer_dim: int, reduction_layers: int):
        super().__init__()

        self.encoder = Encoder(
            embed_dim=embed_dim, 
            transfer_dim=transfer_dim, 
            reduction_layers=reduction_layers
            )

        self.decoder = Decoder(
            embed_dim=embed_dim,
            transfer_dim=transfer_dim,
            reduction_layers=reduction_layers
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


