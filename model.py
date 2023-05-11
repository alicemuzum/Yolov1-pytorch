import torch.nn as nn
import torch

config = [
    (7, 64, 2, 3),                      # (kernel_size, number of filters as output, stride, padding)
    "M",
    (3, 192, 1, 1),
    "M",                                # M for Max Pooling
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],# Repeat first two 4 times
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(inp_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self,x):
        return self.leakyrelu(self.bn(self.conv(x)))

class YoloV1(nn.Module):
    def __init__(self,inp_channels=3, **kwargs):
        super(YoloV1,self).__init__()
        self.config = config
        self.inp_channels = inp_channels
        self.darknet = self._create_conv_layers(self.config)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self,x):
        x =self.darknet(x)
        return self.fcs(torch.flatten(x,start_dim=1))
    
    def _create_conv_layers(self,config):
        layers = []
        inp_channels = self.inp_channels

        for block in config:
            if type(block) == tuple:
                layers += [
                    CNNBlock(
                    inp_channels, block[1],  kernel_size=block[0], stride=block[2], padding=block[3]
                    )
                ]
                inp_channels = block[1]

            elif type(block) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                

            elif type(block) == list:
                conv1 = block[0]# tuple
                conv2 = block[1]# tuple
                num_repeats = block[2]#integer

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                        inp_channels,
                        conv1[1],
                        kernel_size=conv1[0],
                        stride=conv1[2],
                        padding=conv1[3]
                        )
                    ]
                    layers += [
                        CNNBlock(
                        conv1[1],
                        conv2[1],
                        kernel_size=conv2[0],
                        stride=conv2[2],
                        padding=conv2[3]
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496, ), # Original should be 4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496,S*S * (C + B * 5)) # (S, S, 30) olmalı en son, herhalde loss fonks.da yapıcaz. C+B*5 = 30
        )
    
def test(S=7, B=2, C=20):
    model = YoloV1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2,3,448,448))
    print(model(x).shape)

test()