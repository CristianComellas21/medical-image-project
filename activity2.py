from pathlib import Path

import numpy as np

from dicom import read_dicom_files

REF_FOLDER = Path("data/REF")
INPUT_FOLDER = Path("data/RM_Brain_3D-SPGR")


def main():

    # Load dicom reference files
    dicom_ref = read_dicom_files(REF_FOLDER)
    ref_pixel_array = dicom_ref[1]["pixel_array"]
    ref_metadata = dicom_ref[1]["metadata"]
    print(ref_pixel_array.shape)

    # Load dicom file to be registered
    dicom_input = read_dicom_files(INPUT_FOLDER)
    input_pixel_array = dicom_input[1]["pixel_array"]
    input_metadata = dicom_input[1]["metadata"]
    print(input_pixel_array.shape)


if __name__ == "__main__":
    main()
