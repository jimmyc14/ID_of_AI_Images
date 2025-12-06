import torch.nn as nn
from torchvision import models

def build_resnet50(zero_init=False):

    model = models.resnet50(weights=True)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),   # hidden layer
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2)           # final output
    )

    return model