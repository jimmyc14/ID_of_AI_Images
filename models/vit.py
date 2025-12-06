import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def build_vit(zero_init=False):
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    model.heads.head = nn.Sequential(
        nn.LayerNorm(model.heads.head.in_features),
        nn.Linear(model.heads.head.in_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    return model