from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.ndimage import rotate
from tqdm import tqdm

from dicom import get_segmentation_layers, read_dicom_files
from visualize import plot_interactive_dicom

# Define constants for the paths to the DICOM files
DICOM_FOLDER = "data/HCC-TACE-Seg/HCC_003"
SEGMENTATION_PATH = (
    f"{DICOM_FOLDER}/09-12-1997-NA-AP LIVER-64595/300.000000-Segmentation-45632/1-1.dcm"
)
RESULTS_FOLDER = "results"


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the maximum intensity projection on the sagittal orientation."""
    # Your code here:
    #   See `np.max(...)`
    # ...
    return np.max(img_dcm, axis=2)


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

    # for value in metadata:
    #     print(value.SliceLocation)

    # Get slice thickness
    slice_thickness = metadata[0].SliceThickness

    # Get pixel spacing
    pixel_spacing = metadata[0].PixelSpacing

    # Get segmentation data
    segmentation_data = dicom_files["300.000000-Segmentation-45632"][1]
    segmentation_pixel_array = segmentation_data["pixel_array"]
    segmentation_metadata = segmentation_data["metadata"][0]

    # Get segmentation layers
    segmentation_layers = get_segmentation_layers(
        segmentation_pixel_array, segmentation_metadata
    )

    # Align number of slices with the reference image
    n_slices_ref = pixel_array.shape[0]
    n_slices_seg = list(segmentation_layers.values())[0]["pixel_array"].shape[0]

    n_slices = min(n_slices_ref, n_slices_seg)

    if n_slices < n_slices_ref:
        indices = np.linspace(0, n_slices_ref - 1, num=n_slices, dtype=int)
        pixel_array = pixel_array[indices]
    else:
        indices = np.linspace(0, n_slices_seg - 1, num=n_slices, dtype=int)
        segmentation_layers = {
            key: {"pixel_array": value["pixel_array"][indices]}
            for key, value in segmentation_layers.items()
        }

    # Create folder for the results
    Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)

    # Variables for the visualization of the projections
    fig, _ = plt.subplots()
    cm_image = "bone"
    cm_segmentation = "tab10"
    img_min = pixel_array.min()
    img_max = pixel_array.max()
    aspect = slice_thickness / pixel_spacing[0]
    alpha = 0.3

    #  Create projections
    n = 16
    projections = []

    # for idx, angle in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
    for idx, angle in tqdm(
        enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)),
        total=n,
        desc="Creating projections",
    ):

        # Rotate the image on the axial plane
        rotated_img = rotate_on_axial_plane(pixel_array, angle)

        # Rotate the segmentation layers pixel arrays on the axial plane
        rotated_segmentation_layers = [
            rotate_on_axial_plane(layer["pixel_array"], angle)
            for layer in segmentation_layers.values()
        ]

        # Compute the maximum intensity projection on the sagittal orientation
        projection = MIP_sagittal_plane(rotated_img)
        projection_segmentation_layers = np.array(
            [MIP_sagittal_plane(layer) for layer in rotated_segmentation_layers]
        )
        projection_segmentation = projection_segmentation_layers.max(axis=0)

        # Normalize the projections
        normalized_projection = (projection - img_min) / (img_max - img_min)
        normalized_projection_segmentation = (
            projection_segmentation - projection_segmentation.min()
        ) / (projection_segmentation.max() - projection_segmentation.min())

        # Apply colormap to the projections
        projection = plt.colormaps[cm_image](normalized_projection)
        projection_segmentation = plt.colormaps[cm_segmentation](
            normalized_projection_segmentation
        )

        # Alpha fuse the image and the segmentation
        final_projection = alpha * projection + (1 - alpha) * projection_segmentation

        final_projection[normalized_projection_segmentation == 0] = projection[
            normalized_projection_segmentation == 0
        ]

        plt.imshow(
            final_projection,
            vmin=img_min,
            vmax=img_max,
            aspect=aspect,
        )
        plt.savefig(f"{RESULTS_FOLDER}/Projection_{idx}.png")  # Save animation
        projections.append(final_projection)  # Save for later animation

    # Save and visualize animation
    animation_data = [
        [
            plt.imshow(
                img,
                animated=True,
                vmin=img_min,
                vmax=img_max,
                aspect=aspect,
            )
        ]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data, interval=250, blit=True)
    anim.save("results/Animation.gif")  # Save animation
    plt.show()  # Show animation


if __name__ == "__main__":
    main()
