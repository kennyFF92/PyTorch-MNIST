import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def weight_initializer(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)


def network_init(net_class, classes, epochs, learning_rate=1e-3):
    net = net_class(classes)
    net.apply(weight_initializer)

    cost = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.5),
                               int(epochs * 0.8)])

    return net, cost, optimizer, lr_scheduler


class Swish(nn.Module):
    def forward(self, x):
        swish = x * torch.sigmoid(x)
        return swish


class ConvLeNet(nn.Module):
    def __init__(self, classes):
        super(ConvLeNet, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=0), nn.BatchNorm2d(32), Swish(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=0), nn.BatchNorm2d(32), Swish(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=2,
                      padding=2), nn.BatchNorm2d(32), Swish(),
            nn.Dropout(p=0.4))
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0), nn.BatchNorm2d(64), Swish(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0), nn.BatchNorm2d(64), Swish(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=5,
                      stride=2,
                      padding=2), nn.BatchNorm2d(64), Swish(),
            nn.Dropout(p=0.4))
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=1,
                      padding=0), nn.BatchNorm2d(128), Swish(), nn.Flatten(),
            nn.Dropout(p=0.4))
        self.fc_preds = nn.Linear(in_features=128, out_features=classes)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        logits = self.fc_preds(x)
        probs = F.softmax(logits, 1)

        return logits, probs
