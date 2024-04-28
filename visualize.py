from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from dicom import read_dicom_files

INITIAL_SLICE = 0


def plot_interactive_dicom(dicom_pixel_array: np.ndarray, axis: int = 0) -> None:

def plot_interactive_dicom(
    dicom_pixel_array: np.ndarray, axis: int = 0, aspect: float = 1.0
) -> None:

    # Convert all the slices to a color mapped version using the "bone" colormap
    color_mapped_slices = __apply_colormap_to_dicom(
        dicom_pixel_array, axis=axis, normalize=True, apply_log=True
    )

    # Create a figure and axis and adjust the bottom of the plot to make space for the slider
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    color_mapped_slice = color_mapped_slices[INITIAL_SLICE]
    ax.imshow(color_mapped_slice, aspect=aspect)

    slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
    slider = Slider(
        slider_ax,
        "Slice",
        0,
        dicom_pixel_array.shape[axis] - 1,
        valinit=INITIAL_SLICE,
    )

    def update(val):
        slice_idx = int(val)
        color_mapped_slice = color_mapped_slices[slice_idx]
        ax.imshow(color_mapped_slice, aspect=aspect)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def __apply_colormap_to_dicom(
    dicom_pixel_array: np.ndarray,
    axis: int = 0,
    colormap: str = "bone",
    normalize: bool = False,
    apply_log: bool = False,
) -> np.ndarray:
    """Apply a colormap to a DICOM pixel array."""
    max_value = dicom_pixel_array.max()
    min_value = dicom_pixel_array.min()

    if normalize:
        dicom_pixel_array = (dicom_pixel_array - min_value) / (max_value - min_value)

    if apply_log:
        dicom_pixel_array = np.log2(dicom_pixel_array + 1)  # Logarithmic scaling

    color_mapped_slices = []
    for i in range(dicom_pixel_array.shape[axis]):
        color_mapped_slices.append(
            plt.colormaps[colormap](dicom_pixel_array.take(i, axis=axis))
        )
    return np.array(color_mapped_slices)


