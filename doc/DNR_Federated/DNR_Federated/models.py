import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet_300_100(nn.Module):
    """Simple LeNet model having only NN with hidden layers [300, 100]
    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """
    def __init__(self, save_features=None, bench_model=False):
        super(LeNet_300_100, self).__init__()
        self.fc1 = nn.Linear(28*28, 300, bias=True)
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)
        self.mask = None

    def forward(self, x):
        x0 = x.view(-1, 28*28)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return F.log_softmax(x3, dim=1)

'''
Model LeNet_5 used in caffe.
'''

class LeNet_5_Caffe(nn.Module):
    """LeNet-5 without padding in the first layer.
    This is based on Caffe's implementation of Lenet-5 and is slightly different
    from the vanilla LeNet-5. Note that the first layer does NOT have padding
    and therefore intermediate shapes do not match the official LeNet-5.
    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, save_features=None, bench_model=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0, bias=True)
        self.conv2 = nn.Conv2d(20, 50, 5, bias=True)
        self.fc3 = nn.Linear(50 * 4 * 4, 500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

'''
The following VGG model is imported from the standard VGG16 code that we have used for periodic 
sparsity code and ADMM code. 
The VGG16 is a VGG16-like model having nearest similarity to VGG16-D, with only one FC layer
at the end.
'''
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, init_weights=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(512, 10)
        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512,4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096,4096),
                nn.ReLU(True),
                nn.Linear(4096,10),
                )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain = 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        input.data=input.data.sign()
        out = nn.functional.linear(input, self.weight)

        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = input.data.sign()
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

'''
Following is the VGG9 binary activation based code having 6 CONV layes
and 3 FC layers. The binary activations are at CONV layers only.
'''
class HardtanhSign(nn.Hardtanh):
    def __init__(self, low=-1, high=1):
        super(HardtanhSign, self).__init__(low, high)
        self.low = low
        self.high = high

    def forward(self, input):
        hardtanh = F.hardtanh(input, self.low, self.high)
        return hardtanh + (torch.sign(hardtanh) - hardtanh).detach()

class VGG9(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG9, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            HardtanhSign(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            HardtanhSign(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            HardtanhSign(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            HardtanhSign(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            HardtanhSign(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            HardtanhSign(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


