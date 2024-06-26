from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.ndimage import rotate
from tqdm import tqdm

from dicom import get_segmentation_layers, read_dicom_files

# Define constants for the paths to the DICOM files
DICOM_FOLDER = "data/HCC-TACE-Seg/HCC_003"
SEGMENTATION_PATH = (
    f"{DICOM_FOLDER}/09-12-1997-NA-AP LIVER-64595/300.000000-Segmentation-45632/1-1.dcm"
)
RESULTS_FOLDER = "results/activity1"
FULL_CYCLE_SECONDS = 3


def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the maximum intensity projection on the coronal orientation."""
    return np.max(img_dcm, axis=1)


def closest_index_different_from_zero_coronal_plane(img):
    """Return the closest index different from zero on the coronal plane."""

    # Get the indices of the maximum value along the y-axis
    indices = np.argmax(img != 0, axis=1)

    result = img[np.arange(img.shape[0])[:, None], indices, np.arange(img.shape[2])]    
    return result


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate the image on the axial plane."""
    return rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)


def main():

    # ====================================================
    # ================= ARGUMENT PARSING =================
    # ====================================================

    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        help="Mode of projection",
        type=str,
        choices=["MIP", "CIP"],
        default="MIP",
    )
    parser.add_argument(
        "-n",
        "--n_projections",
        help="Number of projections",
        type=int,
        default=260,
    )
    args = parser.parse_args()

    # Get the mode of projection
    mode = args.mode

    # ====================================================
    # ================== LOAD DICOM FILES ================
    # ====================================================

    # --------- INPUT DICOM FILES ---------

    # Load all DICOM files in the folder
    dicom_folder = Path(f"{DICOM_FOLDER}/09-12-1997-NA-AP LIVER-64595")
    dicom_files = read_dicom_files(dicom_folder)

    # Get first DICOM data
    dicom_data = dicom_files["4.000000-Recon 2 LIVER 3 PHASE AP-18688"][1]
    pixel_array = dicom_data["pixel_array"]
    metadata = dicom_data["metadata"]

    # I have checked that the pixel spacing is the same for the images and the segmentation
    # Get slice thickness
    slice_thickness = metadata[0].SliceThickness

    # Get pixel spacing
    pixel_spacing = metadata[0].PixelSpacing

    # --------- REFERENCE DICOM FILES ---------

    # Get segmentation data
    segmentation_data = dicom_files["300.000000-Segmentation-45632"][1]
    segmentation_pixel_array = segmentation_data["pixel_array"]
    segmentation_metadata = segmentation_data["metadata"][0]

    # Get segmentation layers
    segmentation_layers = get_segmentation_layers(
        segmentation_pixel_array, segmentation_metadata
    )

    # Sort the slices of the pixel array
    # The slices of the segmentation are sorted on the load function
    ordered_indices = np.argsort([m.ImagePositionPatient[2] for m in metadata])[::-1]
    pixel_array = pixel_array[ordered_indices]
    metadata = [metadata[i] for i in ordered_indices]

    # ====================================================
    # =============== CREATE PROJECTIONS =================
    # ====================================================

    # Create folder for the results
    Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)

    # Variables for the visualization of the projections
    fig, _ = plt.subplots()
    cm_image = "bone"
    cm_segmentation = "tab10"
    img_min = -200
    img_max = 1000
    aspect = slice_thickness / pixel_spacing[0]
    alpha = 0.5

    #  Create projections
    n = args.n_projections
    projections = []

    print(f"Creating {n} projections with mode {mode}.")
    for idx, angle in tqdm(
        enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)),
        total=n,
        desc="Creating projections",
    ):

        # Rotate the image on the axial plane
        rotated_img = rotate_on_axial_plane(pixel_array, angle)

        # Rotate the segmentation layers pixel arrays on the axial plane
        rotated_segmentation_layers = [
            rotate_on_axial_plane(layer["pixel_array"], angle).astype(np.int8)
            * layer["num"]
            for layer in segmentation_layers.values()
        ]

        # Compute the maximum intensity projection on the coronal orientation
        projection = MIP_coronal_plane(rotated_img)

        if mode == "CIP":
            # Combine the segmentation layers
            projection_segmentation = np.sum(rotated_segmentation_layers, axis=0)
            projection_segmentation = closest_index_different_from_zero_coronal_plane(
                projection_segmentation
            )
        elif mode == "MIP":
            # Combine the segmentation layers
            projection_segmentation_layers = np.array(
                [MIP_coronal_plane(layer) for layer in rotated_segmentation_layers]
            )
            projection_segmentation = projection_segmentation_layers.max(axis=0)

        # Normalize the projection
        normalized_projection = (projection - img_min) / (img_max - img_min)

        # Apply colormap to the projections
        projection_cmp = plt.colormaps[cm_image](normalized_projection)
        projection_segmentation_cmp = plt.colormaps[cm_segmentation](
            projection_segmentation
        )

        # Alpha fuse the image and the segmentation
        final_projection = (
            alpha * projection_cmp + (1 - alpha) * projection_segmentation_cmp
        )

        indices = np.isclose(projection_segmentation, 0, atol=1e-5, rtol=1e-5)
        final_projection[indices] = projection_cmp[indices]

        # Visualize the projection without axis neither white borders
        # and put a black border around the image
        plt.imshow(
            final_projection,
            aspect=aspect,
        )
        plt.axis("off")

        plt.savefig(f"{RESULTS_FOLDER}/Projection_{mode}_{idx}.png")  # Save animation
        projections.append(final_projection)  # Save for later animation

    # ====================================================
    # ================== SAVE ANIMATION ==================
    # ====================================================

    # Save and visualize animation
    animation_data = [
        [
            plt.imshow(
                img,
                animated=True,
                aspect=aspect,
            )
        ]
        for img in projections
    ]

    # Add legend with the segmentation layers with boxes to indicate the color
    plt.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab10(i+1))
            for i in range(len(segmentation_layers))
        ],
        labels=[layer["name"] for layer in segmentation_layers.values()],
        loc="upper right",
    )

    interval = FULL_CYCLE_SECONDS * 1000 // n
    anim = animation.ArtistAnimation(fig, animation_data, interval=interval, blit=True)
    anim.save(f"{RESULTS_FOLDER}/Animation_{mode}.gif")  # Save animation
    plt.show()  # Show animation


if __name__ == "__main__":
    main()
