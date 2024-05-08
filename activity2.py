from pathlib import Path

import numpy as np
from scipy.ndimage import rotate, shift, zoom
from scipy.optimize import minimize
from skimage.transform import resize

from dicom import get_atlas_mask, read_dicom_files
from visualize import plot_interactive_dicom

REF_FOLDER = Path("data/REF")
INPUT_FOLDER = Path("data/RM_Brain_3D-SPGR")
ATLAS_FOLDER = Path("data/atlas/dcm/")


def mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute the mean squared error between two images."""
    # Your code here:
    #   ...
    return np.mean((image1 - image2) ** 2)


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


def found_best_coregistration(
    ref_img: np.ndarray, input_img: np.ndarray
) -> tuple[float, ...]:
    """Find the best registration parameters."""
    initial_parameters = [
        0,  # Angle of rotation in degrees around axial axis (0)
        0,  # Angle of rotation in degrees around coronal axis (1)
        0,  # Angle of rotation in degrees around sagittal axis (2)
        0,  # Translation axis 0
        0,  # Translation axis 1
        0,  # Translation axis 2
    ]

    def function_to_minimize(parameters):
        """Transform the input image using a screw transformation and compute mean squared error."""

        transformed_img = apply_rigid_transformation(input_img, parameters)

        return mean_squared_error(ref_img, transformed_img)

    # Apply least squares optimization
    result = minimize(function_to_minimize, initial_parameters, method="Nelder-Mead")
    return result


def main():

    # Load dicom file to be registered
    dicom_input = read_dicom_files(INPUT_FOLDER)
    input_pixel_array = dicom_input[1]["pixel_array"]
    # input_metadata = dicom_input[1]["metadata"]
    print(f"{input_pixel_array.shape=}")

    # Load dicom reference files
    dicom_ref = read_dicom_files(REF_FOLDER)
    ref_pixel_array = dicom_ref[1]["pixel_array"]
    ref_pixel_array = resize(
        ref_pixel_array, input_pixel_array.shape, anti_aliasing=True
    )
    # ref_metadata = dicom_ref[1]["metadata"]
    print(f"{ref_pixel_array.shape=}")

    # # Find the best coregistration parameters
    # best_parameters = found_best_coregistration(ref_pixel_array, input_pixel_array)
    # print(f"{best_parameters=}")

    # Load atlas dicom files
    atlas_dicom = read_dicom_files(ATLAS_FOLDER)
    atlas_pixel_array = atlas_dicom[1]["pixel_array"]
    # atlas_metadata = atlas_dicom[1]["metadata"]
    print(f"{atlas_pixel_array.shape=}")

    # Get atlas thalamus mask
    thalamus_mask = get_atlas_mask(atlas_pixel_array, "Thal")
    print(f"{thalamus_mask.shape=}")
    print(np.unique(thalamus_mask))

    plot_interactive_dicom(thalamus_mask.astype(np.float32), axis=0)


if __name__ == "__main__":
    main()
