import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn

class EEGNet_Conv(nn.Module):
    def __init__(self, args):
        super(EEGNet_Conv, self).__init__()

        self.height = len(args.electrodes) # numbers of electrodes be used
        self.c_feats = args.c_feats
        self.c_feats1 = self.c_feats // 2
        self.out_feats = args.out_feats
        self.pool_wight = args.pool_wight
        self.dropout = args.dropout

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.c_feats1 , (1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(self.c_feats1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.c_feats1, self.c_feats, (self.height, 1), groups=self.c_feats1, bias=False),
            nn.BatchNorm2d(self.c_feats),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_wight)),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.c_feats, self.c_feats, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(self.c_feats),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_wight * 2 )),
            nn.Dropout(self.dropout)
        )
        self.remain = 58
        self.classifier = nn.Linear(self.c_feats * self.remain, self.out_feats)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    model = nn.Conv2d(1,16,(3,3),padding=0)

    model2 = nn.Conv2d(16, 32, (3, 3), padding=0,groups=2)
    data = torch.rand((16, 1, 12, 12))
    da = model(data)
    da2 = model2(da)
    print(';')