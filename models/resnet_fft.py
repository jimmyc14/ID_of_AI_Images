import torch.nn as nn
from torchvision import models
import torch

def build_resnet50_fft(zero_init=False):

    # MODEL SETUP (ResNet50)

    model = models.resnet50(weights=True)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),   # hidden layer
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2)           # final output
    )

    # updating model to handle FFT, swiching CNN to 6
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # initialize the extra channels for CNN
    with torch.no_grad():
        model.conv1.weight[:, :3] = old_conv.weight  # keep pretrained RGB filters
        if zero_init is True:
            model.conv1.weight[:, 3:] = torch.zeros_like(old_conv.weight[:, :3]) # zero init
        else:
            model.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)  # duplicate as “gray”

    return model