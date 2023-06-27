import math
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


def exist(object):
    return not object is None


class TimeEmbeddings(nn.Module):
    """ Build sinusoidal embeddings """
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

class Residual(nn.Module):
    def __init__(self, fn) :
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if exist(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8) -> None:
        super().__init__()

        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
            if exist(time_emb_dim)
            else None
        )

        self.block1 = Block(in_channels, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb=None):
        scale_shift = None
        if exist(self.mlp) and exist(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb[:, :, None, None]
            scale_shift = time_emb.chunk(2, dim=1)

        x_conv = self.block1(x, scale_shift=scale_shift)
        x_conv = self.block2(x_conv)
        return x_conv + self.res_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim=None) -> None:
        super().__init__()

        self.conv = ResnetBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, time_emb=None):
        x_conv = self.conv(x, time_emb)
        x_pool = self.pool(x_conv)
        return x_conv, x_pool


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim=None) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ResnetBlock(2 * out_channels, out_channels, time_emb_dim)

    def forward(self, x, x_corresponding, time_emb=None):
        x = self.up(x)
        x = torch.cat((x_corresponding, x), dim=1)
        x = self.conv(x, time_emb)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32) -> None:
        super().__init__()

        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
    

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class UNet(nn.Module):
    def __init__(self, img_dim, img_channels, in_channels=16, channel_mults=(1, 2, 4, 8, 16)) -> None:
        super().__init__()
        
        time_dim = img_dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbeddings(img_dim),
            nn.Linear(img_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        conv_channels = [img_channels, *map(lambda m: in_channels * m, channel_mults[:-1])]
        down_in_out = list(zip(conv_channels[:-1], conv_channels[1:]))
        mid_in_out = tuple([conv_channels[-1], 2 * conv_channels[-1]])
        conv_channels = list(reversed([in_channels, *map(lambda m: in_channels * m, channel_mults[1:])]))
        up_in_out = list(zip(conv_channels[:-1], conv_channels[1:]))

        self.downs = nn.ModuleList([])
        for in_ch, out_ch in down_in_out:
            self.downs.append(nn.ModuleList([
                ResnetBlock(in_ch, out_ch, time_emb_dim=time_dim),
                Residual(PreNorm(out_ch, LinearAttention(out_ch))),
                nn.MaxPool2d(kernel_size=2)
            ]))

        # for in_ch, out_ch in down_in_out:
        #     self.downs.append(DownBlock(in_ch, out_ch, time_emb_dim=time_dim))

        self.middle = ResnetBlock(*mid_in_out, time_emb_dim=time_dim)
        self.middle_attn = Residual(PreNorm(mid_in_out[-1], Attention(mid_in_out[-1])))

        self.ups = nn.ModuleList([])
        for in_ch, out_ch in up_in_out:
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                Residual(PreNorm(2 * out_ch, LinearAttention(2 * out_ch))),
                ResnetBlock(2 * out_ch, out_ch, time_emb_dim=time_dim)
            ]))

        # for in_ch, out_ch in up_in_out:
        #     self.ups.append(UpBlock(in_ch, out_ch, time_emb_dim=time_dim))
    
        self.output = nn.Conv2d(in_channels, img_channels, kernel_size=1)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x_corresponding = []

        for resnet, attn, down, in self.downs:
            x_conv = resnet(x, t)
            x_corresponding.append(x_conv)

            x = attn(x_conv)
            x = down(x)

        # for down in self.downs:
        #     x_conv, x = down(x, t)
        #     x_corresponding.append(x_conv)

        x = self.middle(x, t)
        x = self.middle_attn(x)

        for up, attn, resnet in self.ups:
            x = up(x)
            x = torch.cat((x_corresponding.pop(), x), dim=1)
            x = attn(x)
            x = resnet(x, t)

        # for up in self.ups:
        #     x = up(x, x_corresponding.pop(), t)
        
        return self.output(x)