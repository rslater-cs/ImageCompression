from torch.nn import Module, Linear
from torch import Tensor
from torchvision.models.swin_transformer import SwinTransformer

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

transformer = SwinTransformer(patch_size=[5,5], embed_dim=48, depths=[2,2,2,2], num_heads=[7,7,7,7], window_size=[5,5])
print(transformer)

print(get_n_params(transformer))