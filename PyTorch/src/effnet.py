import torch.nn as nn
import timm
from torchstat import stat

class EfficientNet(nn.Module):
    def __init__(self, num_classes=200, pretrained=False):
        super(EfficientNet, self).__init__()

        self.pretrained = pretrained

        self.model = timm.create_model('efficientnet_b2', pretrained=pretrained)

        original_weights = self.model.conv_stem.weight.data
        averaged_weights = original_weights.mean(dim=1, keepdim=True)
        self.model.conv_stem.weight.data = averaged_weights

        self.fc = nn.Linear(self.model.classifier.in_features, num_classes)
        self.model.classifier = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = EfficientNet()
    stat(model, (1, 128, 1000))