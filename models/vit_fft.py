import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch

def build_vit_fft(zero_init=False):
    # MODEL SETUP ViT
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    old_conv = model.conv_proj
    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    with torch.no_grad():

        if zero_init is True:
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = 0.0
        else:
            new_conv.weight[:, :3, :, :] = old_conv.weight # RGB weights
            new_conv.weight[:, 3:, :, :] = old_conv.weight

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.conv_proj = new_conv

    model.heads.head = nn.Sequential(
        nn.LayerNorm(model.heads.head.in_features),
        nn.Linear(model.heads.head.in_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    return model