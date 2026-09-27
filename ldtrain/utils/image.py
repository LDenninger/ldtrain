
import numpy as np
import torch

def to_bgr_uint8(image: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert RGB(A) data to the 8-bit BGR(A) layout OpenCV writes.

    Args:
        image: Image or frame stack, shape (..., H, W), (..., H, W, 1), (..., H, W, 3)
            or (..., H, W, 4). `uint8`, or floating point in [0, 1].

    Returns:
        Same data as `uint8`, shape (..., H, W) or (..., H, W, C), channels reversed to
        BGR(A). A trailing singleton channel is dropped.

    Raises:
        TypeError: If the dtype is neither `uint8` nor floating point.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    elif image.dtype != np.uint8:
        raise TypeError(f'Expected uint8 or floating point image, got {image.dtype}')

    if image.shape[-1] == 1:
        image = image[..., 0]  # (..., H, W)
    elif image.shape[-1] in (3, 4):
        # RGB -> BGR and RGBA -> BGRA: swap the first three channels, keep alpha
        image = image[..., [2, 1, 0, *range(3, image.shape[-1])]]
    return np.ascontiguousarray(image)