import torch
import torch.nn as nn  # Neural network layers
import numpy as np  # Handling arrays and matrices


CONFIG = {
    "A": [[64], [128], [256, 256], [512, 512], [512, 512]],
    "B": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    "D": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    "E": [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
}


class VGG(nn.Module):
    def __init__(self, arch, num_classes=1000):
        super(VGG, self).__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.conv_layers = self._make_conv()
        self.fc_layers = self._make_fc()
        # self._initialize_weights()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.AdaptiveAvgPool2d((7, 7))(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def _make_conv(self):
        cfg = CONFIG[self.arch]
        conv_layers = nn.Sequential()
        in_channel = 3

        for conv_block in cfg:
            for oc in conv_block:
                conv_layers.append(nn.Conv2d(in_channels=in_channel, out_channels=oc, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
                conv_layers.append(nn.ReLU())
                in_channel = oc
            conv_layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=2))
            
        return conv_layers

    def _make_fc(self):
        return nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, self.num_classes),
            )

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=1e-4)
                nn.init.constant_(layer.bias, 0)
        