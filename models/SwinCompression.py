from typing import Optional, Callable, List, Any

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
class Encoder(SwinTransformer):

    def __init__(self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None
        ):
        super().__init__(patch_size, embed_dim, depths, num_heads, 
            window_size, mlp_ratio, dropout, attention_dropout, 
            stochastic_depth_prob, num_classes, norm_layer, block)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
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