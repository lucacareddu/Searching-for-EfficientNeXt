# Another version of mnblock.py that does not exploit "groups" arg in Conv2d and instead uses for-cycles, lists, and concatenations
# They have the same number of parameters and behave the same given the same input
# but this version performs worse most likely because kernels of each branch are initialized independently

# We can say that this version is wrong because the input and its weight matrix should be treated always in a standalone fashion
# but for the moment in which they may have to be split over the channels to perform independent (grouped) convolutions

import torch
import torch.nn as nn
from torchstat import stat

def bn_layer(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False, act=nn.SiLU()):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
        nn.BatchNorm2d(out_channels),
        act,
    )

class SEBlock(nn.Module):
    def __init__(self, channels, se_ratio=0.25, act=nn.SiLU()):
        super(SEBlock, self).__init__()

        reduced_channels = max(1, int(channels * se_ratio))
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            act,
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
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
        branch_in_channels = split_size
        branch_out_channels = mid_channels // self.num_branches

        self.branches = nn.ModuleList()

        for _ in range(self.num_branches):
            modules = []
        
            if fused:
                # Regular convolution
                modules.append(bn_layer(branch_in_channels, branch_out_channels, kernel_size=kernel, stride=stride, padding=padding))
            else:
                # Pointwise (1x1) expansion
                modules.append(bn_layer(branch_in_channels, branch_out_channels))
                # Depthwise convolution
                modules.append(bn_layer(branch_out_channels, branch_out_channels, kernel_size=kernel, stride=stride, padding=padding, groups=branch_out_channels))
            
            if self.num_branches > 1 and use_se:
                # Squeeze and Excitation mechanism
                modules.append(SEBlock(branch_out_channels, se_ratio=se_ratio))
                # Pointwise (1x1) linear projection
                modules.append(bn_layer(branch_out_channels, branch_out_channels, act=nn.Identity()))

            self.branches.append(nn.Sequential(*modules))

        self.final = nn.Sequential()

        if use_se:
            # Squeeze and Excitation mechanism
            self.final.append(SEBlock(mid_channels, se_ratio=se_ratio))

        # Pointwise (1x1) linear projection
        self.final.append(bn_layer(mid_channels, out_channels, act=nn.Identity()))

    def forward(self, x):
        chunks = torch.chunk(x, self.num_branches, dim=1)
        y = torch.cat([branch(chunk) for chunk, branch in zip(chunks, self.branches)], dim=1)
        y = self.final(y)
        return y + x if y.size() == x.size() else y


if __name__ == "__main__":
    bottleneck = InvertedBottleneck(1, 32)
    stat(bottleneck, (1, 128, 1000))