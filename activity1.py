from pathlib import Path

import numpy as np
from scipy.ndimage import rotate

from dicom import read_dicom_files
from visualize import plot_interactive_dicom

# Define constants for the paths to the DICOM files
DICOM_FOLDER = "data/HCC-TACE-Seg/HCC_003"
SEGMENTATION_PATH = (
    f"{DICOM_FOLDER}/09-12-1997-NA-AP LIVER-64595/300.000000-Segmentation-45632/1-1.dcm"
)


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate the image on the axial plane."""
    return rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)


def main():

    # Load all DICOM files in the folder
    dicom_folder = Path(f"{DICOM_FOLDER}/09-12-1997-NA-AP LIVER-64595")
    dicom_files = read_dicom_files(dicom_folder)

    # Get first DICOM data
    dicom_data = dicom_files["2.000000-PRE LIVER-87624"][1]
    pixel_array = dicom_data["pixel_array"]
    metadata = dicom_data["metadata"]

    # Get segmentation data
    segmentation_data = dicom_files["300.000000-Segmentation-45632"][1]
    segmentation_pixel_array = segmentation_data["pixel_array"]
    segmentation_metadata = segmentation_data["metadata"]

    print(f"Segmentation metadata: {segmentation_metadata}")

    # # Get slice thickness
    # slice_thickness = metadata[0].SliceThickness
    # print(f"Slice thickness: {slice_thickness}")
    #
    # # Get pixel spacing
    # pixel_spacing = metadata[0].PixelSpacing
    # print(f"Pixel spacing: {pixel_spacing}")
    #
    # # Plot the DICOM files
    # aspects = [
    #     1.0,
    #     slice_thickness / pixel_spacing[0],
    #     slice_thickness / pixel_spacing[1],
    # ]
    # for axis in range(3):
    #     plot_interactive_dicom(pixel_array, axis=axis, aspect=aspects[axis])


if __name__ == "__main__":
    main()
