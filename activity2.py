from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from dicom import get_atlas_mask, read_dicom_files
from transform import translation_then_axialrotation
from visualize import plot_interactive_dicom

REF_FOLDER = Path("data/REF")
INPUT_FOLDER = Path("data/RM_Brain_3D-SPGR")
ATLAS_FOLDER = Path("data/atlas/dcm/")


def mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute the mean squared error between two images."""
    # Your code here:
    #   ...
    return np.mean((image1 - image2) ** 2)


def found_best_coregistration(
    ref_img: np.ndarray, input_img: np.ndarray
) -> tuple[float, ...]:
    """Find the best registration parameters."""
    initial_parameters = [
        0,
        0,
        0,  # Translation vector
        0,  # Angle in rads
        1,
        0,
        0,  # Axis of rotation
    ]

    def function_to_minimize(parameters):
        """Transform input landmarks, then compare with reference landmarks."""
        # Your code here:
        #   ...
        inp_landmarks_transf = np.asarray(
            [
                translation_then_axialrotation(point, parameters)
                for point in inp_landmarks
            ]
        )
        return vector_of_residuals(ref_landmarks, inp_landmarks_transf)

    # Apply least squares optimization
    result = least_squares(function_to_minimize, x0=initial_parameters, verbose=1)
    return result


def main():

    # Load dicom reference files
    dicom_ref = read_dicom_files(REF_FOLDER)
    ref_pixel_array = dicom_ref[1]["pixel_array"]
    ref_metadata = dicom_ref[1]["metadata"]
    print(f"{ref_pixel_array.shape=}")

    # Load dicom file to be registered
    dicom_input = read_dicom_files(INPUT_FOLDER)
    input_pixel_array = dicom_input[1]["pixel_array"]
    input_metadata = dicom_input[1]["metadata"]
    print(f"{input_pixel_array.shape=}")

    # Load atlas dicom files
    atlas_dicom = read_dicom_files(ATLAS_FOLDER)
    atlas_pixel_array = atlas_dicom[1]["pixel_array"]
    atlas_metadata = atlas_dicom[1]["metadata"]
    print(f"{atlas_pixel_array.shape=}")
    print(np.unique(atlas_pixel_array))

    # Get atlas thalamus mask
    thalamus_mask = get_atlas_mask(atlas_pixel_array, "Amygdala")
    print(f"{thalamus_mask.shape=}")
    print(np.unique(thalamus_mask))

    plot_interactive_dicom(thalamus_mask, axis=0)


if __name__ == "__main__":
    main()
