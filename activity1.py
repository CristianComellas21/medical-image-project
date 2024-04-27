import numpy as np
import pydicom as dicom

from visualize import plot_interactive_dicom

# Define constants for the paths to the DICOM files
DICOM_FOLDER = "data/HCC-TACE-Seg/HCC_003"
SEGMENTATION_PATH = (
    f"{DICOM_FOLDER}/09-12-1997-NA-AP LIVER-64595/300.000000-Segmentation-45632/1-1.dcm"
)
DICOM_PATH = f"{DICOM_FOLDER}/09-12-1997-NA-AP LIVER-64595/4.000000-Recon 2 LIVER 3 PHASE AP-18688/1-001.dcm"


def main():

    # Load the segmentation DICOM file
    segmentation_dicom = dicom.read_file(SEGMENTATION_PATH)
    # print(segmentation_dicom)
    # print(segmentation_dicom.pixel_array.shape)
    plot_interactive_dicom(segmentation_dicom.pixel_array)

    # Load the first DICOM file
    dicom_files = dicom.dcmread(DICOM_PATH)
    # print(dicom_files)
    # print(dicom_files.pixel_array.shape)

    # Create a mask from the segmentation DICOM file


if __name__ == "__main__":
    main()
