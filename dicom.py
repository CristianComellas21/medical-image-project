from pathlib import Path
from typing import Union

import numpy as np
import pydicom as dicom

ATLAS_TEXT_FILE = Path("data/atlas/AAL3_1mm.txt")
ATLAS_INFO = None


def read_dicom_files(dicom_folder: Path) -> dict[Union[str, int], dicom.FileDataset]:
    """Read all DICOM files in a folder and return them as a dictionary."""

    # Create an empty dictionary to store the DICOM files
    dicom_files = {}

    # Detect all DICOM files in the folder and save as a list
    dicom_files_list = dicom_folder.glob("**/*.dcm")

    # Get paths without names and make them unique
    dicom_folders = set([file_path.parent for file_path in dicom_files_list])

    # Loop through all DICOM files in the folder to create a dictionary
    # with the folder structure and populate it with the DICOM files
    for folder_path in dicom_folders:

        relative_path = folder_path.relative_to(dicom_folder)

        # Loop through each parent folder
        current_dict = dicom_files
        for parent in relative_path.parts:

            # Create a new dictionary if the parent folder does not exist
            if parent not in current_dict:
                current_dict[parent] = {}

            # Move to the next dictionary
            current_dict = current_dict[parent]

        # Read all DICOM files in the folder
        dicom_files_list = folder_path.glob("*.dcm")

        # Read dicoms
        dicom_data = []
        prev_dicom_idx = None
        for dicom_file in dicom_files_list:
            # Get current dicom index
            current_dicom_idx = (
                int(dicom_file.stem.split("-")[0]) if "-" in dicom_file.stem else 1
            )

            if prev_dicom_idx is None:
                prev_dicom_idx = current_dicom_idx

            # If the current dicom index is not the same as the previous one
            # then we have a new dicom series
            if current_dicom_idx != prev_dicom_idx:
                current_dict[prev_dicom_idx] = __concat_dicom_files(dicom_data)
                dicom_data = []
                prev_dicom_idx = current_dicom_idx

            # Read the DICOM file
            dicom_data.append(dicom.read_file(dicom_file))

        # Concatenate the last dicom series
        current_dict[prev_dicom_idx] = __concat_dicom_files(dicom_data)

    return dicom_files


def get_segmentation_layers(
    segmentation_pixel_array: np.ndarray, segmentation_metadata: dicom.FileDataset
) -> dict[str, np.ndarray]:
    """Get the segmentation layers from a DICOM segmentation file."""

    # Create an empty dictionary to store the segmentation layers
    layers = {}

    # Get the number of sequences and elements per sequence
    n_sequences = len(segmentation_metadata.SegmentSequence)
    n_elements_per_sequence = segmentation_pixel_array.shape[0] // n_sequences

    # Loop through all sequences in the segmentation metadata
    for sequence in segmentation_metadata.SegmentSequence:

        # Get the name of the sequence and the segment number
        name = sequence.SegmentDescription.lower()
        segment_number = sequence.SegmentNumber

        # Get the pixel array for the sequence
        pixel_array = segmentation_pixel_array[
            (segment_number - 1)
            * n_elements_per_sequence : segment_number
            * n_elements_per_sequence
        ]
        pixel_array = pixel_array.astype(np.bool_)

        # Get the image position patient of the indices
        ordered_indices = np.argsort(
            [
                segmentation_metadata.PerFrameFunctionalGroupsSequence[i]
                .PlanePositionSequence[0]
                .ImagePositionPatient[2]
                for i in range(
                    (segment_number - 1) * n_elements_per_sequence,
                    segment_number * n_elements_per_sequence,
                )
            ]
        )[::-1]

        # Sort the pixel array
        pixel_array = pixel_array[ordered_indices]

        # Rotate the pixel array 180 degrees due to
        # the orientation of the patient
        pixel_array = np.rot90(pixel_array, k=2, axes=(1, 2))

        # Store the pixel array in the dictionary along with the name and number
        layers[name] = {
            "pixel_array": pixel_array,
            "name": sequence.SegmentLabel,
            "num": sequence.SegmentNumber,
        }

    return layers


def get_atlas_mask(img_atlas: np.ndarray, region_name: str) -> np.ndarray:

    # Load the atlas info if it is not already loaded
    global ATLAS_INFO
    if ATLAS_INFO is None:
        ATLAS_INFO = __parse_atlas_file(ATLAS_TEXT_FILE)

    # Get the region number from the atlas text file
    left, right = ATLAS_INFO[region_name]
    return (img_atlas >= left) & (img_atlas <= right)


# ---------- Helper functions ----------


def __concat_dicom_files(dicom_files: [dicom.FileDataset]) -> dict:
    """Concatenate a list of DICOM files into a single DICOM file."""
    # print(dicom_files)
    # print(len(dicom_files))

    if len(dicom_files) == 1:
        pixel_array = dicom_files[0].pixel_array
        del dicom_files[0].PixelData
        return {
            "pixel_array": pixel_array,
            "metadata": [dicom_files[0]],
        }

    # Get the first DICOM file in the list
    dicom_file = dicom_files[0]

    # Get the pixel array from the first DICOM file
    pixel_array = dicom_file.pixel_array

    # Get the shape of the pixel array
    shape = pixel_array.shape

    # Create an empty array to store the concatenated pixel array
    concatenated_pixel_array = np.empty((len(dicom_files), shape[0], shape[1]))

    # Create empty list to store metadata
    metadata = []

    # Loop through all DICOM files in the list
    for idx, dicom_file in enumerate(dicom_files):

        # Get the pixel array from the DICOM file
        pixel_array = dicom_file.pixel_array

        # Store the pixel array in the concatenated pixel array
        concatenated_pixel_array[idx] = pixel_array

        # Remove the pixel array from the DICOM file
        del dicom_file.PixelData

        # Store metadata
        metadata.append(dicom_file)

    return {"pixel_array": concatenated_pixel_array, "metadata": metadata}


def __parse_atlas_file(atlas_file: Path) -> dict[str, tuple[int, int]]:
    """Parse the atlas text file and return it as a dictionary."""
    with open(atlas_file, "r") as f:
        lines = f.readlines()

    # Create an empty dictionary to store the atlas regions
    atlas_regions = {}

    # Loop through the lines 2 by 2
    prev_name = None
    left_number = -1
    for i in range(0, len(lines), 2):

        # Get the region base name and the region number
        number, region_base_name = lines[i].strip().split(" ")[:2]
        region_base_name = region_base_name.split("_")[0]

        # If we are at the first region, store the name and number
        if prev_name is None:
            left_number = number
            prev_name = region_base_name
            continue

        if region_base_name != prev_name:

            atlas_regions[prev_name] = (int(left_number), int(number) - 1)
            prev_name = region_base_name
            left_number = number

    return atlas_regions
