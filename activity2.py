from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from skimage.transform import resize

from dicom import get_atlas_mask, read_dicom_files
from transform import (
    apply_inverse_rigid_transformation,
    apply_rigid_transformation,
    print_parameters,
)
from visualize import plot_interactive_dicom

REF_FOLDER = Path("data/REF")
INPUT_FOLDER = Path("data/RM_Brain_3D-SPGR")
ATLAS_FOLDER = Path("data/atlas/dcm/")

COREGISTRATION_SIZE = (64, 64, 64)

INPUT_INTEREST_REGION = (slice(0, 150), slice(30, 470), slice(70, 450))

RESULTS_FOLDER = "results/activity2"


def mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute the mean squared error between two images."""
    # Your code here:
    #   ...
    return [np.mean((image1 - image2) ** 2)]


def found_best_coregistration(
    ref_img: np.ndarray, input_img: np.ndarray
) -> tuple[float, ...]:
    """Find the best registration parameters."""
    initial_parameters = [
        120,  # Angle of rotation in degrees around axial axis (0)
        0,  # Angle of rotation in degrees around coronal axis (1)
        0,  # Angle of rotation in degrees around sagittal axis (2)
        0,  # Translation axis 0
        0,  # Translation axis 1
        0,  # Translation axis 2
    ]

    def function_to_minimize(parameters):
        """Transform the input image using a screw transformation and compute mean squared error."""

        transformed_img = apply_rigid_transformation(input_img, parameters)

        return mean_squared_error(ref_img, transformed_img)

    result = minimize(
        function_to_minimize,
        initial_parameters,
        method="Powell",
        options={"maxiter": 1000, "disp": True},
    )
    return result


def main():

    # ====================================================
    # ================= ARGUMENT PARSING =================
    # ====================================================

    # Parse the arguments
    parser = ArgumentParser()

    # Override the transformation parameters
    parser.add_argument(
        "-o",
        "--override",
        action="store_true",
        default=False,
        help="Override the transformation parameters, recalculating them.",
    )
    parser.add_argument(
        "-g",
        "--generate_gif",
        action="store_true",
        default=False,
        help="Generate gifs with the results",
    )

    args = parser.parse_args()

    # Get the override flag
    override = args.override

    # Get the generate gif flag
    generate_gif = args.generate_gif

    # ====================================================
    # ================== LOAD DICOM FILES ================
    # ====================================================

    # --------- INPUT DICOM FILES ---------

    # Load dicom file to be registered
    dicom_input = read_dicom_files(INPUT_FOLDER)
    input_pixel_array = dicom_input[1]["pixel_array"]
    input_metadata = dicom_input[1]["metadata"]

    # Sort the pixel array
    ordered_indices = np.argsort([m.ImagePositionPatient[2] for m in input_metadata])[
        ::-1
    ]
    input_pixel_array = input_pixel_array[ordered_indices]
    input_metadata = [input_metadata[i] for i in ordered_indices]

    # Select the region of interest
    input_pixel_array = input_pixel_array[INPUT_INTEREST_REGION]
    input_metadata = input_metadata[INPUT_INTEREST_REGION[0]]

    # Normalize the pixel array
    input_pixel_array = (input_pixel_array - np.min(input_pixel_array)) / (
        np.max(input_pixel_array) - np.min(input_pixel_array)
    )

    # --------- REFERENCE DICOM FILES ---------

    # Load dicom reference files
    dicom_ref = read_dicom_files(REF_FOLDER)
    ref_pixel_array = dicom_ref[1]["pixel_array"][::-1, :, :]

    # Normalize the pixel array
    ref_pixel_array = (ref_pixel_array - np.min(ref_pixel_array)) / (
        np.max(ref_pixel_array) - np.min(ref_pixel_array)
    )

    # --------- ATLAS DICOM FILES ---------

    # Load atlas dicom files
    atlas_dicom = read_dicom_files(ATLAS_FOLDER)
    atlas_pixel_array = atlas_dicom[1]["pixel_array"][::-1, :, :]

    # Pad atlas pixel array with 6 zeros at each side on each axis
    # to match the reference pixel array size
    atlas_pixel_array = np.pad(atlas_pixel_array, 6)

    # ====================================================
    # =============== COREGISTRATION PROCESS =============
    # ====================================================

    # Override the transformation parameters if needed
    # executing the optimization process
    if override:
        print("Calculating coregistration parameters...")
        # Resize the images to the same size, which is smaller to speed up the process
        resized_ref_pixel_array = resize(
            ref_pixel_array, COREGISTRATION_SIZE, anti_aliasing=True
        )
        resized_input_pixel_array = resize(
            input_pixel_array, COREGISTRATION_SIZE, anti_aliasing=True
        )

        # Find the best coregistration parameters
        best_parameters = found_best_coregistration(
            resized_ref_pixel_array, resized_input_pixel_array
        )

        # Save the best parameters
        best_parameters = best_parameters.x
        np.save("best_parameters.npy", best_parameters)

        print_parameters(best_parameters)

    else:
        print("Loading coregistration parameters...")
        best_parameters = np.load("best_parameters.npy")
        print_parameters(best_parameters)

    # ====================================================
    # ============= VISUALIZE COREGISTRATION =============
    # ====================================================

    # Resize the input image to the reference image size
    resized_input_pixel_array = resize(
        input_pixel_array, ref_pixel_array.shape, anti_aliasing=False
    )

    # Apply the best coregistration parameters to the input image
    transformed_input_pixel_array = apply_rigid_transformation(
        resized_input_pixel_array, best_parameters
    )

    # Apply colormap to both, the transformed input image and the reference image
    colormapped_transformed_input_pixel_array = plt.cm.bone(
        transformed_input_pixel_array
    )
    colormapped_ref_pixel_array = plt.cm.afmhot(ref_pixel_array)

    # Alpha blend the images
    alpha = 0.3

    blended_image = (
        alpha * colormapped_ref_pixel_array
        + (1 - alpha) * colormapped_transformed_input_pixel_array
    )

    # Plot the blended image
    plot_interactive_dicom(
        blended_image,
        axis=0,
        normalize=False,
        apply_log=False,
        apply_colormap=False,
        title="Coregistration",
    )

    if generate_gif:

        # Create folder to save the results
        Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)

        # Create gif with the coregistration result
        fig = plt.figure()
        fig.patch.set_visible(False)
        plt.axis("off")

        gif_data = [
            [plt.imshow(blended_image[i], animated=True)]
            for i in range(blended_image.shape[0])
        ]
        gif_data += gif_data[::-1]

        interval = 4 * 1000 / blended_image.shape[0]
        anim = animation.ArtistAnimation(
            fig,
            gif_data,
            interval=interval,
            blit=True,
        )

        anim.save(f"{RESULTS_FOLDER}/coregistration.gif")
        plt.close()

    # ====================================================
    # ====== CHECK ATLAS IS ALIGNED WITH REFERENCE =======
    # ====================================================

    # Convert atlas to binary mask
    atlas_binary_pixel_array = atlas_pixel_array > 0

    # Apply colormap to the reference image and the atlas mask
    colormapped_ref_pixel_array = plt.cm.bone(ref_pixel_array)
    colormapped_atlas_binary_pixel_array = plt.cm.tab10(atlas_binary_pixel_array)

    # Alpha blend the images
    alpha = 0.3
    blended_image = (
        alpha * colormapped_ref_pixel_array
        + (1 - alpha) * colormapped_atlas_binary_pixel_array
    )

    indices = atlas_binary_pixel_array == 0
    blended_image[indices] = colormapped_ref_pixel_array[indices]

    # Plot the blended image
    plot_interactive_dicom(
        blended_image,
        axis=0,
        normalize=False,
        apply_log=False,
        apply_colormap=False,
        title="Atlas in reference space",
    )

    if generate_gif:

        # Create gif with the atlas alignment result
        fig, _ = plt.subplots()
        fig.patch.set_visible(False)
        plt.axis("off")

        gif_data = [
            [plt.imshow(blended_image[i], animated=True)]
            for i in range(blended_image.shape[0])
        ]
        gif_data += gif_data[::-1]

        interval = 4 * 1000 / blended_image.shape[0]
        anim = animation.ArtistAnimation(
            fig,
            gif_data,
            interval=interval,
            blit=True,
        )

        anim.save(f"{RESULTS_FOLDER}/atlas_alignment.gif")
        plt.close()

    # ====================================================
    # ======= VISUALIZE THALAMUS IN INPUT SPACE ==========
    # ====================================================

    # Get atlas thalamus mask
    thalamus_mask = get_atlas_mask(atlas_pixel_array, "Thal")

    # Resize the thalamus mask to the input size
    resized_thalamus_mask = resize(
        thalamus_mask, input_pixel_array.shape, anti_aliasing=False
    )

    # Apply the inverse of the best coregistration parameters to the thalamus mask
    transformed_thalamus_mask = apply_inverse_rigid_transformation(
        resized_thalamus_mask.astype(np.float32), best_parameters
    )

    # Convert again to binary mask
    transformed_thalamus_mask = (np.abs(transformed_thalamus_mask) > 0.5).astype(
        np.int8
    )

    # Apply colormap to both, the input image and the thalamus mask
    colormapped_input_pixel_array = plt.cm.bone(input_pixel_array)
    colormapped_transformed_thalamus_mask = plt.cm.tab10(transformed_thalamus_mask)

    # Alpha blend the images
    alpha = 0.3
    blended_image = (
        alpha * colormapped_input_pixel_array
        + (1 - alpha) * colormapped_transformed_thalamus_mask
    )

    indices = transformed_thalamus_mask == 0
    blended_image[indices] = colormapped_input_pixel_array[indices]

    # Plot the blended image
    plot_interactive_dicom(
        blended_image,
        axis=0,
        normalize=False,
        apply_log=False,
        apply_colormap=False,
        title="Thalamus in input space",
    )

    if generate_gif:

        # Create gif with the thalamus alignment result
        fig, _ = plt.subplots()
        fig.patch.set_visible(False)
        plt.axis("off")

        gif_data = [
            [plt.imshow(blended_image[i], animated=True)]
            for i in range(blended_image.shape[0])
        ]
        gif_data += gif_data[::-1]

        interval = 8 * 1000 / blended_image.shape[0]
        anim = animation.ArtistAnimation(
            fig,
            gif_data,
            interval=interval,
            blit=True,
        )

        anim.save(f"{RESULTS_FOLDER}/thalamus_alignment.gif")
        plt.close()


if __name__ == "__main__":
    main()
