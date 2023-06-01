import math
import torch
import torch.nn as nn


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
    

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.norm = nn.GroupNorm(groups, out_channels)
        # self.act = nn.ReLU()
        self.act = nn.SiLU()
    
    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim) -> None:
        super().__init__()

        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
            if time_emb_dim
            else None
        )

        self.block1 = Block(in_channels, out_channels)
        self.block2 = Block(out_channels, out_channels)
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


class UNet(nn.Module):
    def __init__(self, img_dim, img_channels, in_channels=16, channel_mults=(1, 2, 4, 8, 16)) -> None:
        super().__init__()
        self.resolution = img_dim
        
        time_dim = img_dim * 4
        self.time_mlp = nn.Sequential(
            TimeEmbeddings(img_dim),
            nn.Linear(img_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        conv_channels = [img_channels, *map(lambda m: 16 * m, channel_mults[:-1])]
        down_in_out = list(zip(conv_channels[:-1], conv_channels[1:]))
        mid_in_out = tuple([conv_channels[-1], 2 * conv_channels[-1]])
        conv_channels = list(reversed([16, *map(lambda m: 16 * m, channel_mults[1:])]))
        up_in_out = list(zip(conv_channels[:-1], conv_channels[1:]))

        self.downs = nn.ModuleList([])
        for in_ch, out_ch in down_in_out:
            self.downs.append(DownBlock(in_ch, out_ch, time_emb_dim=time_dim))

        self.middle = ResnetBlock(*mid_in_out, time_emb_dim=time_dim)

        self.ups = nn.ModuleList([])
        for in_ch, out_ch in up_in_out:
            self.ups.append(UpBlock(in_ch, out_ch, time_emb_dim=time_dim))
    
        self.output = nn.Conv2d(in_channels, img_channels, kernel_size=1)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x_corresponding = []

        for down in self.downs:
            x_conv, x = down(x, t)
            x_corresponding.append(x_conv)

        x = self.middle(x, t)

        for up in self.ups:
            x = up(x, x_corresponding.pop(), t)
        
        return self.output(x)