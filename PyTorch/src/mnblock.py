# Main and efficient version of the bottleneck block (the other is mnblock_sep.py)

import torch
import torch.nn as nn
from torchstat import stat
# from utils import ChannelShuffle

def bn_layer(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False, act=nn.SiLU()):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
        nn.BatchNorm2d(out_channels),
        # ChannelShuffle(out_channels//2),
        act,
    )

class SEBlock(nn.Module):
    def __init__(self, channels, groups=1, se_ratio=0.25, act=nn.SiLU()):
        super(SEBlock, self).__init__()

        reduced_channels = max(1, int(channels * se_ratio))
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1, groups=groups),
            act,
            nn.Conv2d(reduced_channels, channels, kernel_size=1, groups=groups),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))

class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=2, split_input=True, split_factor=32, fused=False, kernel=3, stride=1, padding=1, use_se=True, se_ratio=0.25):
        super(InvertedBottleneck, self).__init__()

        mid_channels = in_channels * expansion_factor
        
        split_size = min(in_channels, split_factor) if split_input else in_channels
        assert in_channels % split_size == 0

        self.num_branches = in_channels // split_size # can be 1

        self.bottleneck = nn.Sequential()
        
        if fused:
            # Regular convolution [GROUPED]
            self.bottleneck.append(bn_layer(in_channels, mid_channels, kernel_size=kernel, stride=stride, padding=padding, groups=self.num_branches))
        else:
            # Pointwise (1x1) expansion [GROUPED]
            self.bottleneck.append(bn_layer(in_channels, mid_channels, groups=self.num_branches))
            # Depthwise convolution
            self.bottleneck.append(bn_layer(mid_channels, mid_channels, kernel_size=kernel, stride=stride, padding=padding, groups=mid_channels))
        
        if self.num_branches > 1 and use_se:
            # Squeeze and Excitation mechanism [GROUPED]
            self.bottleneck.append(SEBlock(mid_channels, groups=self.num_branches, se_ratio=se_ratio))
            # Pointwise (1x1) linear projection [GROUPED]
            self.bottleneck.append(bn_layer(mid_channels, mid_channels, groups=self.num_branches, act=nn.Identity()))

        if use_se:
            # Squeeze and Excitation mechanism
            self.bottleneck.append(SEBlock(mid_channels, se_ratio=se_ratio))

        # Pointwise (1x1) linear projection
        self.bottleneck.append(bn_layer(mid_channels, out_channels, act=nn.Identity()))

    def forward(self, x):
        y = self.bottleneck(x)
        return y + x if y.size() == x.size() else y


if __name__ == "__main__":
    bottleneck = InvertedBottleneck(1, 32)
    stat(bottleneck, (1, 128, 1000))