from torchvision import transforms
from torchvision.transforms import functional as F
import io
from PIL import Image
import random

class RandomJPEGCompression:
    def __init__(self, quality_range=(30, 95)):
        """
        Simulates JPEG compression by encoding and decoding the image.
        quality_range: (min_quality, max_quality)
        Lower quality → stronger compression & artifacts.
        """
        self.quality_range = quality_range

    def __call__(self, img):
        # Pick random JPEG quality between the quality range
        quality = random.randint(*self.quality_range)

        # Encode the image to JPEG bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        # Decode back into a PIL image
        compressed = Image.open(buffer).convert("RGB")
        return compressed