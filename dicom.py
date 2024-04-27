from pathlib import Path

import numpy as np
import pydicom as dicom


def concat_dicom_files(dicom_files: [dicom.FileDataset]) -> dict:
    """Concatenate a list of DICOM files into a single DICOM file."""
    # print(dicom_files)
    # print(len(dicom_files))

    if len(dicom_files) == 1:
        return {
            "pixel_array": dicom_files[0].pixel_array,
            "metadata": [
                {
                    key: dicom_files[0][key].value
                    for key in dicom_files[0].keys()
                    if key != "PixelData"
                }
            ],
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

        # Store metadata
        metadata.append(
            {
                key: dicom_file[key].value
                for key in dicom_file.keys()
                if key != "PixelData"
            }
        )

    return {"pixel_array": concatenated_pixel_array, "metadata": metadata}


def read_dicom_files(dicom_folder: Path) -> dict[str, dicom.FileDataset]:
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
            current_dicom_idx = int(dicom_file.stem.split("-")[0])
            if prev_dicom_idx is None:
                prev_dicom_idx = current_dicom_idx

            # If the current dicom index is not the same as the previous one
            # then we have a new dicom series
            if current_dicom_idx != prev_dicom_idx:
                current_dict[prev_dicom_idx] = concat_dicom_files(dicom_data)
                dicom_data = []
                prev_dicom_idx = current_dicom_idx

            # Read the DICOM file
            dicom_data.append(dicom.read_file(dicom_file))

        # Concatenate the last dicom series
        current_dict[prev_dicom_idx] = concat_dicom_files(dicom_data)

    return dicom_files
