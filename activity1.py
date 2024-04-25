import numpy as np
import pydicom as dicom

DICOM_FILES_PATH = "data/HCC-TACE-Seg/HCC_003"
SEGMENTATION_PATH = f"{DICOM_FILES_PATH}/09-12-1997-NA-AP LIVER-64595/300.000000-Segmentation-45632/1-1.dcm"


def main():

    # Load the segmentation DICOM file
    segmentation_dicom = dicom.read_file(SEGMENTATION_PATH)
    print(segmentation_dicom)
    print(segmentation_dicom.pixel_array.shape)


if __name__ == "__main__":
    main()
