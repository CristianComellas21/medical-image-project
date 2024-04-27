from pathlib import Path

from dicom import read_dicom_files
from visualize import plot_interactive_dicom

# Define constants for the paths to the DICOM files
DICOM_FOLDER = "data/HCC-TACE-Seg/HCC_003"
SEGMENTATION_PATH = (
    f"{DICOM_FOLDER}/09-12-1997-NA-AP LIVER-64595/300.000000-Segmentation-45632/1-1.dcm"
)
DICOM_PATH = f"{DICOM_FOLDER}/09-12-1997-NA-AP LIVER-64595/4.000000-Recon 2 LIVER 3 PHASE AP-18688/1-001.dcm"


def main():

    # Load all DICOM files in the folder
    dicom_folder = Path(f"{DICOM_FOLDER}/09-12-1997-NA-AP LIVER-64595")

    dicom_files = read_dicom_files(dicom_folder)

    # Get 1 DICOM data
    dicom_data = dicom_files["4.000000-Recon 2 LIVER 3 PHASE AP-18688"][1]
    pixel_array = dicom_data["pixel_array"]

    # Plot the DICOM files
    plot_interactive_dicom(pixel_array)


if __name__ == "__main__":
    main()
