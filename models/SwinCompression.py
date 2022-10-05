import torch.nn as nn
import torch
from torchvision.models.swin_transformer import SwinTransformer, ShiftedWindowAttention, SwinTransformerBlock

# For editing the original Swin Transformer the classification layers
# need to be removed, this class implements the equivilent of 
# deleting a layer
class NoneLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# The encoder utilises a normal Swin Transformer with the classification
# layers removed
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = SwinTransformer(patch_size=[2,2], embed_dim=48, depths=[2,2,2,2], num_heads=[2,2,2,2], window_size=[2,2])
        self.encoder.avgpool = NoneLayer()
        self.encoder.norm = NoneLayer()
        self.encoder.head = NoneLayer()

    def forward(self, x):
        x = self.encoder(x)

        return x

# While the normal Swin Tranformer merges patches, the decompression requires 
# the patches be split to reach the original resolution of the input image
class PatchSplitter(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.enlargement = nn.Linear(dim, 2*dim)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor):

        B, L, C = x.shape
        H, W = self.input_resolution

        assert L == H * W, "input feature has wrong size"
        assert C % 2 == 0, "channels not divisible by 2"

        x = self.norm(x)
        x = self.enlargement(x) # B H*W 2C

        x = x.view(B, H, W, 2*C) # B H W 2C

        diff = (2*C)//4

        x0 = x[:,:,:,0:diff]        # B H W C/2
        x1 = x[:,:,:,diff:2*diff]   # B H W C/2
        x2 = x[:,:,:,2*diff:3*diff] # B H W C/2
        x3 = x[:,:,:,3*diff:4*diff] # B H W C/2

        x = torch.empty(B, 2*H, 2*W, C//2) # B 2*W 2*H C/2

        x[:, 0::2, 0::2, :] = x0
        x[:, 1::2, 0::2, :] = x1
        x[:, 0::2, 1::2, :] = x2
        x[:, 1::2, 1::2, :] = x3

        return x

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(x):
        return x