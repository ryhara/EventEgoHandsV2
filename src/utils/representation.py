import torch
import numpy as np
from torchvision.transforms import functional as TF
from torchvision.transforms import transforms
from torchvision.transforms import v2

ORIGINAL_WIDTH = 346
ORIGINAL_HEIGHT = 260

COLOR_MAP = {
    0: (0, 0, 0),  # background: BGR(black)
    1: (0, 0, 255),  # hand: BGR(red)
    2: (255, 0, 0),  # object: BGR(blue)
}


# =============================================================================
# V1 Transform Functions (従来のtransforms)
# =============================================================================


def get_v1_transform_with_normalize(output_size, use_center_crop=True, num_channels=2):
    """
    V1 style transform with normalization [0,1]
    ToPILImage -> Resize/Crop -> ToTensor (0-1 normalization)
    """
    if use_center_crop:
        resize_or_crop = transforms.Compose(
            [
                transforms.Resize((ORIGINAL_WIDTH, ORIGINAL_WIDTH)),
                transforms.CenterCrop(output_size),
            ]
        )
    else:
        resize_or_crop = transforms.Resize(output_size)

    if num_channels == 2:
        normalize = transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
    else:  # 3 channels
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            resize_or_crop,
            transforms.ToTensor(),  # [0,255] -> [0,1]
            normalize,  # [0,1] -> [-1,1]
        ]
    )


def get_v1_transform_no_normalize(output_size, use_center_crop=True):
    """
    V1 style transform without normalization [0,255]
    ToPILImage -> Resize/Crop -> pil_to_tensor (no normalization)
    """
    if use_center_crop:
        resize_or_crop = transforms.Compose(
            [
                transforms.Resize((ORIGINAL_WIDTH, ORIGINAL_WIDTH)),
                transforms.CenterCrop(output_size),
            ]
        )
    else:
        resize_or_crop = transforms.Resize(output_size)

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            resize_or_crop,
            TF.pil_to_tensor,  # PIL -> Tensor without normalization [0,255]
        ]
    )


# =============================================================================
# V2 Transform Functions (新しいv2 transforms)
# =============================================================================


def get_v2_transform(output_size, use_center_crop=True, normalize=True):
    """
    V2 style transform with normalization [0,1]
    ToImage -> Resize/Crop -> ToDtype (with scale=True)
    """
    if use_center_crop:
        resize_or_crop = v2.Compose(
            [
                v2.Resize((ORIGINAL_WIDTH, ORIGINAL_WIDTH)),
                v2.CenterCrop(output_size),
            ]
        )
    else:
        resize_or_crop = v2.Resize(output_size)

    return v2.Compose(
        [
            v2.ToImage(),  # NumPy/PIL -> Tensor
            resize_or_crop,
            v2.ToDtype(torch.float32, scale=normalize),  # if normalize is True, [0,255] -> [0,1]
            v2.Lambda(lambda x: x / (x.max() + 1e-6)),
        ]
    )

class CustomTransform:
    def __init__(self, output_size, use_center_crop=True):
        self.output_size = output_size
        self.use_center_crop = use_center_crop

    def __call__(self, numpy_array):
        tensor = torch.as_tensor(numpy_array, dtype=torch.float32)

        tensor = tensor.permute(2, 0, 1)

        if self.use_center_crop:
            tensor = TF.resize(tensor, (ORIGINAL_WIDTH, ORIGINAL_WIDTH))
            tensor = TF.center_crop(tensor, self.output_size)
        else:
            tensor = TF.resize(tensor, self.output_size)

        return tensor


def visualize_lnes_frame(lnes: torch.Tensor, is_normalize: bool = False) -> np.ndarray:
    """
    Visualize the LNES frame by converting it to a BGR image format.

    Args:
        lnes (torch.Tensor): The LNES tensor with shape (2, H, W).
        normalize_to_255 (bool): If True, normalize to range 0~1 and then scale to 0~255.

    Returns:
        np.ndarray: The BGR image representation of the LNES.
    """
    lnes = lnes.permute(1, 2, 0).cpu().numpy().astype(np.float32)

    if is_normalize:
        lnes = (lnes / (np.max(lnes) + 1e-6)) * 255
    else:
        lnes = lnes * 255
    h, w, c = lnes.shape
    b = np.zeros((h, w), dtype=np.uint8)  # Blue channel is zero
    g = lnes[..., 0]  # Green channel
    r = lnes[..., 1]  # Red channel

    lnes_bgr = np.stack([b, g, r], axis=2).astype(np.uint8)
    return lnes_bgr


def visualize_event_frame(event_frame: torch.Tensor) -> np.ndarray:
    """
    Visualize the event frame by converting it to a BGR image format.
    """
    event_frame = event_frame.permute(1, 2, 0).cpu().numpy() * 255
    event_frame = np.clip(event_frame, 0, 255).astype(np.uint8)
    return event_frame

def visualize_predicted_mask(predicted_mask: torch.Tensor) -> np.ndarray:
    """
    Visualize the predicted mask by converting it to a BGR image format.
    """
    predicted_mask = torch.argmax(predicted_mask, dim=0).cpu().numpy().astype(np.uint8)
    color_mask = np.zeros(
        (predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8
    )
    for class_id, color in COLOR_MAP.items():
        color_mask[predicted_mask == class_id] = color
    return color_mask

def visualize_gt_mask(gt_mask: torch.Tensor) -> np.ndarray:
    gt_mask = gt_mask.cpu().numpy().astype(np.uint8)
    color_mask = np.zeros(
        (gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8
    )
    for class_id, color in COLOR_MAP.items():
        color_mask[gt_mask == class_id] = color
    return color_mask