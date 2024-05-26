import numpy as np
from scipy.ndimage import rotate, shift


def apply_rigid_transformation(
    img: np.ndarray, parameters: tuple[float, ...]
) -> np.ndarray:
    """Apply a rigid transformation to an image."""

    angle_0, angle_1, angle_2, translation_0, translation_1, translation_2 = parameters

    # Rotate the image
    rotated_img = rotate(img, angle_0, axes=(1, 2), reshape=False)
    rotated_img = rotate(rotated_img, angle_1, axes=(0, 2), reshape=False)
    rotated_img = rotate(rotated_img, angle_2, axes=(0, 1), reshape=False)

    # Translate the image
    translated_img = shift(rotated_img, (translation_0, translation_1, translation_2))

    return translated_img


def apply_inverse_rigid_transformation(
    img: np.ndarray, parameters: tuple[float, ...]
) -> np.ndarray:
    """Apply the inverse of a rigid transformation to an image."""
    angle_0, angle_1, angle_2, translation_0, translation_1, translation_2 = parameters

    # Translate the image
    translated_img = shift(img, (-translation_0, -translation_1, -translation_2))

    # Rotate the image
    rotated_img = rotate(translated_img, -angle_2, axes=(0, 1), reshape=False)
    rotated_img = rotate(rotated_img, -angle_1, axes=(0, 2), reshape=False)
    rotated_img = rotate(rotated_img, -angle_0, axes=(1, 2), reshape=False)

    return rotated_img


def fix_parameters_scale(
    parameters: tuple[float, ...],
    coregistration_size: np.ndarray,
    image_size: np.ndarray,
) -> tuple[float, ...]:
    """Fix the scale of the translation parameters, depending on the size the coregistration was performed and
    the size of the image to be transformed."""

    # Compute the scale factor for each dimension
    scale_factor = image_size / coregistration_size

    # Fix the translation parameters
    angle_0, angle_1, angle_2, translation_0, translation_1, translation_2 = parameters
    translation_0 *= scale_factor[0]
    translation_1 *= scale_factor[1]
    translation_2 *= scale_factor[2]

    return (angle_0, angle_1, angle_2, translation_0, translation_1, translation_2)


def print_parameters(parameters: tuple[float, ...]):
    """Print the parameters of the rigid transformation."""
    angle_0, angle_1, angle_2, translation_0, translation_1, translation_2 = parameters

    print(f"Angle 0: {angle_0:0.2f}")
    print(f"Angle 1: {angle_1:0.2f}")
    print(f"Angle 2: {angle_2:0.2f}")
    print(f"Translation 0: {translation_0:0.2f}")
    print(f"Translation 1: {translation_1:0.2f}")
    print(f"Translation 2: {translation_2:0.2f}")
