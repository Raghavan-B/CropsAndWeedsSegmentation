from ensure import ensure_annotations
import numpy as np
import albumentations as A


@ensure_annotations
def convert_color_mask_to_label(mask:np.ndarray, COLOR_TO_LABEL:dict)->np.ndarray:
    """
    Convert an RGB mask (H, W, 3) into a label mask (H, W) based on a predefined mapping.

    Args:
        mask (numpy.ndarray): RGB image mask with shape (H, W, 3).

    Returns:
        numpy.ndarray: 2D array of shape (H, W) with label values.
    """
    # Create an empty label mask.
    label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    # Iterate over the mapping and update label_mask
    for color, label in COLOR_TO_LABEL.items():
        # Create a boolean mask for pixels that match the color.
        match = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
        label_mask[match] = label
    return label_mask

@ensure_annotations
def colorize_label_mask(label_mask:np.ndarray ,LABEL_TO_COLOR:dict)->np.ndarray:
    """
    Convert a 2D label mask (numpy array) to an RGB color image.

    Args:
        label_mask (numpy.ndarray): Array of shape (H, W) with label values.

    Returns:
        numpy.ndarray: Color image of shape (H, W, 3).
    """
    # Create an empty RGB image.
    h, w = label_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in LABEL_TO_COLOR.items():
        # Apply color where the label matches.
        color_mask[label_mask == label] = color

    return color_mask

def get_train_augs():
    return A.Compose([
        A.HorizontalFlip(0.5),
        A.VerticalFlip(0.43),
        A.RandomBrightnessContrast(0.2),
        A.GaussianBlur(p=0.1),
        A.GridDistortion(p=0.2),
        A.RandomRotate90(p=0.5),
    ])
