from config import vgg_path
from torch import nn
from torchvision import models
import torch


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16()
        vgg.load_state_dict(torch.load(vgg_path))
        self.vgg = nn.Sequential(*(list(vgg.features.children())[:36])).eval()

        self.mse = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        return self.mse(self.vgg(input), self.vgg(target).detach())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
