import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

INITIAL_SLICE = 0


def plot_interactive_dicom(dicom_pixel_array):

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    color_mapped_slice = plt.colormaps["bone"](dicom_pixel_array[INITIAL_SLICE])
    print(color_mapped_slice.shape)
    ax.imshow(color_mapped_slice)

    slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
    slider = Slider(
        slider_ax,
        "Slice",
        0,
        dicom_pixel_array.shape[0] - 1,
        valinit=INITIAL_SLICE,
    )

    def update(val):
        slice_idx = int(val)
        color_mapped_slice = plt.colormaps["bone"](dicom_pixel_array[slice_idx])
        ax.imshow(color_mapped_slice)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()
