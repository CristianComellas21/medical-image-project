from pathlib import Path

import numpy as np

from dicom import read_dicom_files

REF_FOLDER = Path("data/REF")
INPUT_FOLDER = Path("data/RM_Brain_3D-SPGR")


def main():

    # Load dicom reference files
    dicom_ref = read_dicom_files(REF_FOLDER)

    # Load dicom file to be registered
    dicom_input = read_dicom_files(INPUT_FOLDER)

    print("Dicom reference files:")
    for key, value in dicom_ref.items():
        print(key, value)


if __name__ == "__main__":
    main()
