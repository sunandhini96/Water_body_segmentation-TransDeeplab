# installing all the required packages
!pip install timm
!pip install einops
!pip install tensorboardX
!pip install medpy
!pip install yacs
!pip install torchinfo
!pip install einops

# importing all required packages
import math
import copy
import torch.optim as optim
from torch.nn.parallel import DataParallel
from collections import OrderedDict
from torch import nn
from torchinfo import summary
import copy
import torch 
from torch import nn
from torch.nn import init
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os
from sklearn.metrics import jaccard_score, precision_score, recall_score
import numpy as np
from torch import nn
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


## Transdeeplab architecture : Deeplabv3+ architecture is based on transformers (here we are using swin transformer in encoder, ASPP and decoder) part.


# defining class for patch embedding

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(224)
        patch_size = to_2tuple(4)
        #print("image size",img_size)
        img_size=tuple(map(int, img_size))
        patch_size =tuple(map(int,patch_size))
        #print(type(int(img_size[0]))
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        #print(patches_resolution)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * \
            (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

# defining the class for patch merging

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


# defining the class for window attention

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

# defining the class for MLP

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# defining the class for Swin transformer block

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        #print("input resolution",self.input_resolution)
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        #print("shift size",self.shift_size)
        #print("window size",self.window_size)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

# defining the class for Basic layer 

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        down = None
        if self.downsample is not None:
            down = self.downsample(x)
        return x, down

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

# defining the class for Swin encoder

class SwinEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 high_level_idx=2, low_level_idx=0, low_level_after_block=False, high_level_after_block=True,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, high_level_norm=False, low_level_norm=False, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):

        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.img_size = 224
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio


        self.high_level_idx = high_level_idx
        #print(high_level_idx)
        self.low_level_idx = low_level_idx
        self.low_level_after_block = low_level_after_block
        self.high_level_after_block = high_level_after_block

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        #print(patches_resolution)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            
        # storing sizes and dimensions of the outputs
        #print(high_level_idx)
        
        #print(type(self.img_size))
        #print(type(high_level_idx))
        self.high_level_size = int(self.img_size) // (4 * 2**int(high_level_idx))
        self.high_level_dim = int(embed_dim * 2 ** high_level_idx)
        self.low_level_dim = int(embed_dim * 2 ** low_level_idx)

        self.high_level_norm = norm_layer(self.high_level_dim) if high_level_norm else None
        self.low_level_norm = norm_layer(self.low_level_dim) if low_level_norm else None
        
    def forward(self, x):
        """
        x: input batch with shape (batch_size, in_chans, img_size, img_size)
        returns 
            1. low_level_features with shape (batch_size, low_size, low_size, low_chans)
            2. high_level_features with shape (batch_size, high_size, high_size, high_chans)
        """
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
            
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        low_level = high_level = None
        if self.low_level_idx == 0 and not self.low_level_after_block:
            low_level = x
    
        down = None
        depth = 0
        
        for layer in self.layers:
            x, down = layer(x if down is None else down)
            
            if depth == self.low_level_idx and self.low_level_after_block:
                low_level = x
            if depth == self.high_level_idx and self.high_level_after_block:
                high_level = x
            
            depth += 1
            
            if depth == self.low_level_idx and not self.low_level_after_block:
                low_level = down
            if depth == self.high_level_idx and not self.high_level_after_block:
                high_level = down            

        if self.high_level_norm is not None:
            high_level = self.high_level_norm(high_level)
        if self.low_level_norm is not None:
            low_level = self.low_level_norm(low_level)
            
        low_size = int(math.sqrt(low_level.size(1)))
        high_size = int(math.sqrt(high_level.size(1)))

        low_level = low_level.view(-1, low_size, low_size, low_level.shape[-1])
        high_level = high_level.view(-1, high_size, high_size, high_level.shape[-1])

        return low_level, high_level

    def load_from(self, pretrained_path):
        pretrained_path = pretrained_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
                    
            found = 0
            for k in list(full_dict.keys()):
                if k in model_dict:
                    # print("here")
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]
                    else:
                        found += 1

            msg = self.load_state_dict(full_dict, strict=False)
            # print(msg)
            
            print(f"Encoder Found Weights: {found}")
        else:
            print("none pretrain")




# defining the encoder , ASPP and decoder configurations classes

class EncoderConfig:    
    encoder_name = 'swin'
    load_pretrained = False
    
    img_size = 224
    #print(type(img_size))
    window_size = 7

    patch_size = 4
    in_chans = 3
    embed_dim = 96
    depths = [2, 2, 6 ]
    num_heads = [3, 6, 12]

    low_level_idx = 0
    high_level_idx = int(2)
    high_level_after_block = True
    low_level_after_block = True

    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.1
    attn_drop_rate = 0.1
    drop_path_rate = 0.1
    
    norm_layer = 'layer'
    high_level_norm = False
    low_level_norm = True
    
    ape = False
    patch_norm = True
    use_checkpoint = False


class ASPPConfig:
    aspp_name = 'swin'
    load_pretrained = False
    cross_attn = 'CBAM' # set to None to disable
    
    depth = 2
    num_heads = 3
    start_window_size = 2 ## This means we have 2, 7, 14 as window sizes so 3 level
    
    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.1
    attn_drop_rate = 0.1
    drop_path_rate = 0.1
    
    norm_layer = 'layer'
    aspp_norm = False
    aspp_activation = 'relu' # set to None in order to deactivate
    aspp_dropout = 0.2
    
    downsample = None
    use_checkpoint = False
    

class DecoderConfig:
    decoder_name = 'swin'
    load_pretrained = True
    extended_load = False

    window_size = EncoderConfig.window_size
    
    num_classes = 2
    
    low_level_idx = EncoderConfig.low_level_idx
    high_level_idx = EncoderConfig.high_level_idx
    
    depth = 2
    last_layer_depth = 6
    num_heads = 3
    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.1
    attn_drop_rate = 0.1
    drop_path_rate = 0.1
    norm_layer = 'layer'
    decoder_norm = True
    
    use_checkpoint = False


# defining the building the Encoder 

import os
import urllib.request
import timm
import torch
from torch import nn
#import model.backbones.resnets as resnets
#from model.backbones.swin import SwinEncoder
#from model.backbones.xception import AlignedXception
#from model.backbones.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d



def build_encoder(config):
    if config.encoder_name == 'swin':
        if config.norm_layer == 'layer':
            norm_layer = nn.LayerNorm
            
        return SwinEncoder(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            high_level_idx=config.high_level_idx,
            low_level_idx=config.low_level_idx,
            high_level_after_block=config.high_level_after_block,
            low_level_after_block=config.low_level_after_block,
            embed_dim=config.embed_dim, 
            depths=config.depths, 
            num_heads=config.num_heads,
            window_size=config.window_size, 
            mlp_ratio=config.mlp_ratio, 
            qkv_bias=config.qkv_bias, 
            qk_scale=config.qk_scale,
            drop_rate=config.drop_rate, 
            attn_drop_rate=config.attn_drop_rate, 
            drop_path_rate=config.drop_path_rate,
            norm_layer=norm_layer,
            high_level_norm=config.high_level_norm,
            low_level_norm=config.low_level_norm,
            ape=config.ape, 
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint
        )
        
    if config.encoder_name == 'xception':
        if config.sync_bn:
            bn = SynchronizedBatchNorm2d
        else:
            bn = nn.BatchNorm2d
        return AlignedXception(output_stride=config.output_stride,
                               input_size=config.img_size,
                               BatchNorm=bn, pretrained=config.pretrained,
                               high_level_dim=config.high_level_dim)
    
    if config.encoder_name == 'resnet':
        model = timm.create_model('resnet50_encoder', 
                                  pretrained=False,
                                  high_level=None,
                                  num_classes=0)
        if config.load_pretrained:
            path = os.path.expanduser("~") + '/.cache/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth'
            if not os.path.isfile(path):
                print("downloading ResNet50 pretrained weights...")
                urllib.request.urlretrieve('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth',
                                           path)
                
            weight = torch.load(path)
            msg = model.load_state_dict(weight, strict=False)
            print(msg)
        
        model.layer4 = nn.Identity()
        model.high_level_size = 14
        model.high_level_dim = 384
        model.low_level_dim = 128
        
        return model
        



# Cross Attention


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x) :
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x) :
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result],1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, input_dim, reduction, input_size, out_dim):
        super().__init__()
        self.input_size = input_size
        self.ca = ChannelAttention(channel=input_dim, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=1)

        self.proj = nn.Linear(input_dim, out_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.input_size ** 2
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, self.input_size, self.input_size)
        
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        out = out + residual
        out = out.view(B, C, L).permute(0, 2, 1).contiguous()
        return self.proj(out)
        


# if __name__ == '__main__':
#     input = torch.randn(4, 196, 1152)
#     channels = input.shape[-1]
#     cbam = CBAMBlock(input_dim=channels,reduction=24,input_size=14, out_dim=96)
#     param = sum([p.numel() for p in cbam.parameters()]) / 10**6
    
#     output = cbam(input)
#     print(output.shape)
#     print(param)






# Swin ASPP


#from model.backbones.swin import BasicLayer
#from model.cross_attn import CBAMBlock

class SwinASPP(nn.Module):
    def __init__(self, input_size, input_dim, out_dim, cross_attn,
                 depth, num_heads, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate, 
                 norm_layer, aspp_norm, aspp_activation, start_window_size,
                 aspp_dropout, downsample, use_checkpoint):
        
        super().__init__()
        
        self.out_dim = out_dim
        if input_size == 24:
            self.possible_window_sizes = [4, 6, 8, 12, 24]
        else:
            self.possible_window_sizes = [i for i in range(start_window_size, input_size+1) if input_size%i==0]

        self.layers = nn.ModuleList()
        for ws in self.possible_window_sizes:
            layer = BasicLayer(dim=int(input_dim),
                               input_resolution=(input_size, input_size),
                               depth=1 if ws==input_size else depth,
                               num_heads=num_heads,
                               window_size=ws,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=drop_path_rate,
                               norm_layer=norm_layer,
                               downsample=downsample,
                               use_checkpoint=use_checkpoint)
            
            self.layers.append(layer)
        
        if cross_attn == 'CBAM':
            self.proj = CBAMBlock(input_dim=len(self.layers)*input_dim, 
                                  reduction=12, 
                                  input_size=input_size,
                                  out_dim=out_dim)
        else:
            self.proj = nn.Linear(len(self.layers)*input_dim, out_dim)
        
        # Check if needed
        self.norm = norm_layer(out_dim) if aspp_norm else None
        if aspp_activation == 'relu':
            self.activation = nn.ReLU()
        elif aspp_activation == 'gelu':
            self.activation = nn.GELU()
        elif aspp_activation is None:
            self.activation = None
        
        self.dropout = nn.Dropout(aspp_dropout)

    def forward(self, x):
        """
        x: input tensor (high level features) with shape (batch_size, input_size, input_size, input_dim)
        returns ...
        """
        B, H, W, C = x.shape
        x = x.view(B, H*W, C)

        features = []
        for layer in self.layers:
            out, _ = layer(x)
            features.append(out)

        features = torch.cat(features, dim=-1)
        features = self.proj(features)

        # Check if needed 
        if self.norm is not None:
            features = self.norm(features)
        if self.activation is not None:
            features = self.activation(features)
        features = self.dropout(features)

        return features.view(B, H, W, self.out_dim)
    
    def load_from(self, pretrained_path):
        pretrained_path = pretrained_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.state_dict()
            num_layers = len(self.layers)
            num_pretrained_layers = set([int(k[7]) for k, v in pretrained_dict.items() if 'layers' in k])

            full_dict = copy.deepcopy(pretrained_dict)
            
            layer_dict = OrderedDict()
            
            for i in range(num_layers):
                keys = [item for item in pretrained_dict.keys() if f'layers.{i}' in item]
                for key in keys:
                    for j in num_pretrained_layers:
                        if key in layer_dict: continue
                        # new_k = "layers." + str(i) + k[8:]
                        pre_k = "layers." + str(j) + key[8:]
                        pre_v = pretrained_dict.get(pre_k, None)
                        if pre_v is not None:
                            layer_dict[key] = copy.deepcopy(pre_v)
                    
                        
                        for k in list(layer_dict.keys()):
                            if k in model_dict:
                                if layer_dict[k].shape != model_dict[k].shape:
                                    # print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                                    del layer_dict[k]
                            elif k not in model_dict:
                                del layer_dict[k]
            msg = self.load_state_dict(layer_dict, strict=False)
            # print(msg)
            
            print(f"ASPP Found Weights: {len(layer_dict)}")
        else:
            print("none pretrain")


def build_aspp(input_size, input_dim, out_dim,config):
    if config.norm_layer == 'layer':
        norm_layer = nn.LayerNorm
    
    if config.aspp_name == 'swin':
        return SwinASPP(
            input_size=input_size,
            input_dim=input_dim,
            out_dim=out_dim,
            depth=config.depth,
            cross_attn=config.cross_attn,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qk_scale=config.qk_scale,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            norm_layer=norm_layer,
            aspp_norm=config.aspp_norm,
            aspp_activation=config.aspp_activation,
            start_window_size=config.start_window_size,
            aspp_dropout=config.aspp_dropout,
            downsample=config.downsample,
            use_checkpoint=config.use_checkpoint
        )
    



# if __name__ == '__main__':
#     # from config import ASPPConfig
    
#     batch = torch.randn(2, 24, 24, 384)
#     model = build_aspp(24, 384, 96, ASPPConfig)

#     print("Num of parameters: ", sum([p.numel() for p in model.parameters()])/10**6)
#     print(model.possible_window_sizes)

#     out = model(batch)
#     print(out.shape)




# Decoder class


import math
import copy
from collections import OrderedDict

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange

#from model.backbones.swin import SwinTransformerBlock

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 4*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
import torch.nn.init as init
class SwinDecoder(nn.Module):
    def __init__(self, low_level_idx, high_level_idx, 
                 input_size, input_dim, num_classes,
                 depth, last_layer_depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate, norm_layer, decoder_norm, use_checkpoint):
        super().__init__()
        self.low_level_idx = low_level_idx
        self.high_level_idx = high_level_idx

        self.layers_up = nn.ModuleList()
        for i in range(high_level_idx - low_level_idx):
            layer_up = BasicLayer_up(dim=int(input_dim),
                                    input_resolution=(input_size*2**i, input_size*2**i),
                                    depth=depth,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=drop_path_rate,
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=use_checkpoint)
            
            self.layers_up.append(layer_up)

        self.last_layers_up = nn.ModuleList()
        for _ in range(low_level_idx+1):
            i+=1
            last_layer_up = BasicLayer_up(dim=int(input_dim)*2,
                                            input_resolution=(input_size*2**i, input_size*2**i),
                                            depth=last_layer_depth,
                                            num_heads=num_heads,
                                            window_size=window_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate,
                                            drop_path=0.0,
                                            norm_layer=norm_layer,
                                            upsample=PatchExpand,
                                            use_checkpoint=use_checkpoint)
            self.last_layers_up.append(last_layer_up)
        
        i += 1
        self.final_up = PatchExpand(input_resolution=(input_size*2**i, input_size*2**i),
                                    dim=int(input_dim)*2,
                                    dim_scale=2,
                                    norm_layer=norm_layer)
        
        if decoder_norm:
            self.norm_up = norm_layer(int(input_dim)*2)
        else:
            self.norm_up = None
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.output_1 = nn.Conv2d(int(input_dim)*2, num_classes, kernel_size=1, bias=False).to(self.device)
        self.output = nn.Linear(num_classes,1)
        #init.xavier_uniform_(self.conv.weight)
        #x = self.output(x)
        #self.output = self.output.to(self.device)
    



      

    def forward(self, low_level, aspp):
        """
        low_level: B, Hl, Wl, C
        aspp: B, Ha, Wa, C
        """
        B, Hl, Wl, C = low_level.shape
        _, Ha, Wa, _ = aspp.shape

        low_level = low_level.view(B, Hl*Wl, C)
        aspp = aspp.view(B, Ha*Wa, C)

        for layer in self.layers_up:
            aspp = layer(aspp)
        
        x = torch.cat([low_level, aspp], dim=-1)

        for layer in self.last_layers_up:
            x = layer(x)

        if self.norm_up is not None:
            x = self.norm_up(x)
            
        x = self.final_up(x)
    
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.to(self.device)
        x = self.output_1(x)
        x = x.permute(0,3,2,1)
        x = self.output(x)
        x = x.permute(0,3,2,1)
        #x = torch.unsqueeze(x, dim=1) 
        #print("device",x.device)
        
        
        return x

    def load_from(self, pretrained_path):
        pretrained_path = pretrained_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin decoder---")

            model_dict = self.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 1 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    current_k_2 = 'last_layers_up.' + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
                    full_dict.update({current_k_2:v})
                    
            found = 0
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        # print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]
                    else:
                        found += 1

            msg = self.load_state_dict(full_dict, strict=False)
            # print(msg)
            
            print(f"Decoder Found Weights: {found}")
        else:
            print("none pretrain")
    
    def load_from_extended(self, pretrained_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        pretrained_dict = pretrained_dict['model']
        model_dict = self.state_dict()
        
        selected_weights = OrderedDict()
        for k, v in model_dict.items():
            # if 'relative_position_index' in k: continue
            if 'blocks' in k:
                name = ".".join(k.split(".")[2:])
                shape = v.shape
                
                for pre_k, pre_v in pretrained_dict.items():
                    if name in pre_k and shape == pre_v.shape:
                        selected_weights[k] = pre_v
                        
        msg = self.load_state_dict(selected_weights, strict=False)
        found = len(model_dict.keys()) - len(msg.missing_keys)
        
        print(f"Decoder Found Weights: {found}")



def build_decoder(input_size, input_dim, config):
    if config.norm_layer == 'layer':
        norm_layer = nn.LayerNorm
    
    if config.decoder_name == 'swin':
        return SwinDecoder(
            input_dim=input_dim,
            input_size=input_size,
            low_level_idx=config.low_level_idx,
            high_level_idx=config.high_level_idx,
            num_classes=config.num_classes,
            depth=config.depth,
            last_layer_depth=config.last_layer_depth,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qk_scale=config.qk_scale,
            qkv_bias=config.qkv_bias,
            drop_path_rate=config.drop_path_rate,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            norm_layer=norm_layer,
            decoder_norm=config.decoder_norm,
            use_checkpoint=config.use_checkpoint
        )



# if __name__ == '__main__':
#     #from config import DecoderConfig
    
#     low_level = torch.randn(2, 96, 96, 96)
#     aspp = torch.randn(2, 24, 24, 96)

#     decoder = build_decoder(24, 96, DecoderConfig)
#     print(sum([p.numel() for p in decoder.parameters()])/10**6)

#     features = decoder(low_level, aspp)
#     print(features.shape)




# Swin deeplab

import torch
from torch import nn

# from model.encoder import build_encoder
# from model.decoder import build_decoder
# from model.aspp import build_aspp

class SwinDeepLab(nn.Module):
    def __init__(self, encoder_config, aspp_config, decoder_config):
        super().__init__()
        self.encoder = build_encoder(encoder_config)
        self.aspp = build_aspp(input_size=self.encoder.high_level_size,
                               input_dim=self.encoder.high_level_dim,
                               out_dim=self.encoder.low_level_dim, config=aspp_config)
        self.decoder = build_decoder(input_size=self.encoder.high_level_size,
                                     input_dim=self.encoder.low_level_dim,
                                     config=decoder_config)

    def run_encoder(self, x):
        low_level, high_level = self.encoder(x)
        return low_level, high_level
    
    def run_aspp(self, x):
        return self.aspp(x)

    def run_decoder(self, low_level, pyramid):
        return self.decoder(low_level, pyramid)

    def run_upsample(self, x):
        return self.upsample(x)

    def forward(self, x):
        low_level, high_level = self.run_encoder(x)
        x = self.run_aspp(high_level)
        x = self.run_decoder(low_level, x)
        
        return x
    

    
