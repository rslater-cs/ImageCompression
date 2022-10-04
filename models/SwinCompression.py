import torch.nn as nn
import torch
from torchvision.models.swin_transformer import SwinTransformer, ShiftedWindowAttention

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class NoneLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = SwinTransformer(patch_size=[4,4], embed_dim=48, depths=[2,2,2,2], num_heads=[7,7,7,7], window_size=[4,4])
        self.encoder.avgpool = NoneLayer()
        self.encoder.norm = NoneLayer()
        self.encoder.head = NoneLayer()

    def forward(self, x):
        x = self.encoder(x)

        return x

class PatchSplitter(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.enlargement = nn.Linear(dim, 2*dim)
        self.norm = norm_layer(dim)

    # def forward(self, x):
    #     """
    #     x: B, H*W, C
    #     """
    #     H, W = self.input_resolution
    #     B, L, C = x.shape
    #     assert L == H * W, "input feature has wrong size"
    #     assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

    #     x = x.view(B, H, W, C)

    #     x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    #     x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    #     x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    #     x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    #     x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    #     x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

    #     x = self.norm(x)
    #     x = self.reduction(x)

    #     return x

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