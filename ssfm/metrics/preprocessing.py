from collections.abc import Iterator

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "torch is required for FID computation. "
        "Install with: uv add --optional metrics torch torchvision"
    ) from e

from PIL import Image


def preprocess_images(
    images: np.ndarray,
    target_size: int = 299,
    batch_size: int = 64,
) -> Iterator[torch.Tensor]:
    n = images.shape[0]
    for start in range(0, n, batch_size):
        batch = images[start : start + batch_size]
        processed = []
        for img in batch:
            img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)

            img = img.transpose(1, 2, 0)
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(
                (target_size, target_size),
                Image.BICUBIC,  # pyright: ignore
            )
            arr = np.array(pil_img).transpose(2, 0, 1)
            processed.append(arr)

        tensor = torch.from_numpy(np.stack(processed)).float() / 255.0
        yield tensor
