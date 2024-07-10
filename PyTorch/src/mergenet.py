import torch.nn as nn
from mnblock import InvertedBottleneck as mnblock
# from mnblock_sep import InvertedBottleneck as mnblock
from torchstat import stat

class MergeNet(nn.Module):
    def __init__(self, num_classes=200, split=True):
        super(MergeNet, self).__init__()

        self.split = split

        self.conv_stem = nn.Sequential(
            mnblock(1, 32, expansion_factor=32, fused=True, stride=2, padding=1, split_input=split),
            mnblock(32, 128, expansion_factor=4, fused=True, stride=2, padding=1, split_input=split),
        )
        self.bottlenecks = nn.Sequential(
            mnblock(128, 128, expansion_factor=2, split_input=split),
            mnblock(128, 256, expansion_factor=2, stride=2, padding=1, split_input=split),
            mnblock(256, 256, expansion_factor=2, split_input=split),
            mnblock(256, 512, expansion_factor=2, stride=2, padding=1, split_input=split),
            mnblock(512, 512, expansion_factor=2, split_input=split),
            mnblock(512, 1024, expansion_factor=2, stride=2, padding=1, split_input=split),
            mnblock(1024, 1024, expansion_factor=2, split_input=split),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bottlenecks(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = MergeNet()
    stat(model, (1, 128, 1000))