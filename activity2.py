from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from skimage.transform import resize

from dicom import get_atlas_mask, read_dicom_files
from transform import apply_inverse_rigid_transformation, apply_rigid_transformation
from visualize import plot_interactive_dicom

REF_FOLDER = Path("data/REF")
INPUT_FOLDER = Path("data/RM_Brain_3D-SPGR")
ATLAS_FOLDER = Path("data/atlas/dcm/")

COREGISTRAION_SIZE = (64, 64, 64)


def mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute the mean squared error between two images."""
    # Your code here:
    #   ...
    return np.mean((image1 - image2) ** 2)


def layer_mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute the mean squared error between two images."""
    # Your code here:
    #   ...
    return np.mean((image1 - image2) ** 2, axis=(1, 2))


def residual_vector(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Compute the residual vector between two images."""
    # Your code here:
    #   ...
    return (image1 - image2).flatten()


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

        # return layer_mean_squared_error(ref_img, transformed_img)
        return residual_vector(ref_img, transformed_img)

    # Apply least squares optimization
    result = least_squares(function_to_minimize, initial_parameters, verbose=2)
    return result


def main():

    # Load dicom file to be registered
    dicom_input = read_dicom_files(INPUT_FOLDER)
    input_pixel_array = dicom_input[1]["pixel_array"]
    input_metadata = dicom_input[1]["metadata"]

    ordered_indices = np.argsort([m.ImagePositionPatient[2] for m in input_metadata])[
        ::-1
    ]
    input_pixel_array = input_pixel_array[ordered_indices]
    input_metadata = [input_metadata[i] for i in ordered_indices]

    print(f"{input_pixel_array.shape=}")

    # plot_interactive_dicom(input_pixel_array, axis=0, normalize=True, apply_log=True)

    # Load dicom reference files
    dicom_ref = read_dicom_files(REF_FOLDER)
    ref_pixel_array = dicom_ref[1]["pixel_array"]
    ref_metadata = dicom_ref[1]["metadata"]

    ref_pixel_array = resize(
        ref_pixel_array, input_pixel_array.shape, anti_aliasing=True
    )

    # Sort the pixel array
    ordered_indices = np.argsort(
        [
            ref_metadata[0]
            .PerFrameFunctionalGroupsSequence[i]
            .PlanePositionSequence[0]
            .ImagePositionPatient[2]
            for i in range(len(ref_metadata[0].PerFrameFunctionalGroupsSequence))
        ]
    )[::-1]
    ref_pixel_array = ref_pixel_array[ordered_indices]
    print(f"{ref_pixel_array.shape=}")

    plot_interactive_dicom(ref_pixel_array, axis=0, normalize=True, apply_log=True)

    # # Find the best coregistration parameters
    # resized_ref_pixel_array = resize(
    #     ref_pixel_array, COREGISTRAION_SIZE, anti_aliasing=True
    # )
    # resized_input_pixel_array = resize(
    #     input_pixel_array, COREGISTRAION_SIZE, anti_aliasing=True
    # )
    # best_parameters = found_best_coregistration(
    #     resized_ref_pixel_array, resized_input_pixel_array
    # )
    # print(f"{best_parameters.x=}")

    # # Load atlas dicom files
    # atlas_dicom = read_dicom_files(ATLAS_FOLDER)
    # atlas_pixel_array = atlas_dicom[1]["pixel_array"]
    # # atlas_metadata = atlas_dicom[1]["metadata"]
    # print(f"{atlas_pixel_array.shape=}")
    #
    # # Get atlas thalamus mask
    # thalamus_mask = get_atlas_mask(atlas_pixel_array, "Thal")
    # print(f"{thalamus_mask.shape=}")
    # print(np.unique(thalamus_mask))
    #
    # plot_interactive_dicom(thalamus_mask.astype(np.float32), axis=0)


if __name__ == "__main__":
    main()
