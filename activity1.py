import numpy as np
import pydicom as dicom

DICOM_FILES_PATH = "data/HCC-TACE-Seg/HCC_003"


def main():
    file = dicom.read_file(
        f"{DICOM_FILES_PATH}/10-31-1997-NA-CT ABDOMEN LIVER-88989/4.000000-Recon 2 3 PHASE LIVER ABD-94194/2-103.dcm"
    )
    print(file)


if __name__ == "__main__":
    main()
