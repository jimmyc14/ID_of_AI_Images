from .resnet import build_resnet50
from .vit import build_vit
from .resnet_fft import build_resnet50_fft
from .vit_fft import build_vit_fft

MODEL_REGISTRY = {
    "resnet50": build_resnet50,
    "vit": build_vit,
    "resnet50_fft": build_resnet50_fft,
    "vit_fft": build_vit_fft
}

def get_model(model_name: str, zero_init=False):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](zero_init=zero_init)