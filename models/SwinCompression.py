from typing import Optional, Callable, List, Any
from functools import partial
from pathlib import Path

import torch.nn as nn
import torch
from torchvision.models.swin_transformer import SwinTransformer, ShiftedWindowAttention, SwinTransformerBlock, Permute
from torchvision.models import vision_transformer
from time import time

class ViTBlock(nn.Module):
    def __init__(self,
        num_heads: int,
        num_features: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
        ):
        super().__init__()

        self.block = vision_transformer.EncoderBlock(num_heads=num_heads,
            hidden_dim=num_features, 
            mlp_dim=int(mlp_ratio*num_features), 
            dropout=dropout, 
            attention_dropout=dropout
            )

    def forward(self, x):
        # input: B C H W

        # x: B H W C
        x = torch.permute(x, (0, 2, 3, 1))
        B, Hx, Wx, C = x.shape

        # B L C
        x = x.reshape(B, Hx*Wx, C)
        x = self.block(x)

        # B H W C
        x = x.reshape(B, Hx, Wx, C)

        # B C H W
        x = torch.permute(x, (0, 3, 1, 2))

        return x


# The encoder utilises a normal Swin Transformer with the classification
# layers removed
class Encoder(SwinTransformer):

    def __init__(self,
        patch_size: List[int],
        embed_dim: int,
        output_dim: int,
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
        num_features = embed_dim * 2 ** (len(depths) - 1)

        self.out_conv = nn.Conv2d(num_features, output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        #input size: B, C, H, W

        # x: B, H, W, C
        x = self.features(x)

        # x: B, C, H, W
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.out_conv(x)
        return x

# While the normal Swin Tranformer merges patches, the decompression requires 
# the patches be split to reach the original resolution of the input image
class PatchSplitting(nn.Module):

    def __init__(self, dim, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.enlargement = nn.Linear(dim, 2*dim)
        self.norm = norm_layer(dim)

    # Check page 17 of notes for diagram of plans
    def split_first(self, x: torch.Tensor):
        return x

    def enlargement_first(self, x: torch.Tensor):
        return x

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        assert C % 2 == 0, "channels not divisible by 2"

        x = self.norm(x) # B H W C

        diff = C//2

        x = self.enlargement(x) # B H W 2C

        x_s = torch.split(x, split_size_or_sections=diff, dim=3)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        x = torch.empty(B, 2*H, 2*W, C//2).to(device)

        x[:, 0::2, 0::2, :] = x_s[0]
        x[:, 1::2, 0::2, :] = x_s[1]
        x[:, 0::2, 1::2, :] = x_s[2]
        x[:, 1::2, 1::2, :] = x_s[3]

        return x

# Used to decompress the image from (H, W, C) to (H*d**2, W*d**2, C//d**2)
# where d = depth
class Decoder(nn.Module):

    def __init__(self,
        input_embed_dim: int,
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        if block is None:
            block = SwinTransformerBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    input_embed_dim, embed_dim, kernel_size=1, stride=1
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim)
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim // (2 ** i_stage)
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            # if i_stage < (len(depths) - 1):
            layers.append(PatchSplitting(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim // (2 ** len(depths))
        self.norm = norm_layer(num_features)

        self.head = nn.Sequential(
            nn.Conv2d(num_features, 3, kernel_size=1, stride=1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        

    def forward(self, x):
        x = self.features(x)
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.head(x)
        return x

class FullSwinCompressor(nn.Module):
    def __init__(self,
        transfer_dim: int,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        ):
        super().__init__()

        self.network_type = "SwinCompression"

        self.encoder = Encoder(
            embed_dim=embed_dim, 
            output_dim=transfer_dim, 
            patch_size=patch_size, 
            depths=depths, 
            num_heads=num_heads, 
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            block=block
        )

        self.vit_block = ViTBlock(num_heads=num_heads[-1], num_features=transfer_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        
        output_dim = embed_dim * 2 ** (len(depths)-1)

        self.decoder = Decoder(
            embed_dim=output_dim, 
            input_embed_dim=transfer_dim,
            depths=depths, 
            num_heads=num_heads, 
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            block=block
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.vit_block(x)
        x = self.decoder(x)

        return x

# NEXT MOVE: Make movie dataset into resolution of (1024, 576) which will allow 6 reduction layers of 
# output size (B, 1, 16, 9) resulting in a compressed image size of 576 bytes and a compressed movie of
# 576*30*60*24 24,883,200 bytes === 24.9MB (movie is currently 422MB)