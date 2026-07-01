import numpy as np


def apply_body_part_occlusion(image: np.ndarray, x: int, y: int, w: int, h: int, region: str):
    """Apply mean-color occlusion to a body region inside a GT box."""
    h2 = h // 2
    h4 = h // 4
    h8 = h // 8
    y2 = y + h2

    if region == 'head':
        image[y:y + h8, x:x + w, :] = np.mean(image[y:y + h8, x:x + w, :], axis=(0, 1))
    elif region == 'top':
        image[y:y2, x:x + w, :] = np.mean(image[y:y2, x:x + w, :], axis=(0, 1))
    elif region == 'bottom':
        image[y2:y2 + h2, x:x + w, :] = np.mean(image[y2:y2 + h2, x:x + w, :], axis=(0, 1))
    elif region == 'legs':
        image[y2 + h4:y2 + h2, x:x + w, :] = np.mean(
            image[y2 + h4:y2 + h2, x:x + w, :], axis=(0, 1)
        )
    else:
        raise ValueError(f'Unknown occlusion region: {region}')

    return image
