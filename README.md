# DICOM Image Processing Project

## Overview

This is the final project of the subject Medical Image Processing of the Master in Intelligent Systems of the Balearic Islands University. It focuses on processing and visualizing DICOM (Digital Imaging and Communications in Medicine) files. It includes tools for applying rigid transformations, creating projections, and coregistering images. The project is structured into multiple Python scripts, each serving a specific purpose.

## Project Structure

The project is divided into the following main scripts:

1. `transform.py`
2. `visualize.py`
3. `dicom.py`
4. `activity1.py`
5. `activity2.py`

#### Scripts

#### 1. `transform.py`

This script includes functions to apply rigid transformations to DICOM images, which involve rotating and translating the images.

- **Functions:**
  - `apply_rigid_transformation(img: np.ndarray, parameters: tuple[float, ...]) -> np.ndarray`
  - `apply_inverse_rigid_transformation(img: np.ndarray, parameters: tuple[float, ...]) -> np.ndarray`
  - `print_parameters(parameters: tuple[float, ...])`

#### 2. `visualize.py`

This script provides functionalities for visualizing DICOM images with interactive plots using Matplotlib.

- **Functions:**
  - `plot_interactive_dicom(dicom_pixel_array: np.ndarray, axis: int = 0, aspect: float = 1.0, colormap: str = "bone", apply_colormap: bool = True, normalize: bool = False, apply_log: bool = False, title: str = "") -> None`
  - `main()`

#### 3. `dicom.py`

This script includes functions to read DICOM files and extract segmentation layers from DICOM segmentation files.

- **Functions:**
  - `read_dicom_files(dicom_folder: Path) -> dict[Union[str, int], dicom.FileDataset]`
  - `get_segmentation_layers(segmentation_pixel_array: np.ndarray, segmentation_metadata: dicom.FileDataset) -> dict[str, np.ndarray]`
  - `get_atlas_mask(img_atlas: np.ndarray, region_name: str) -> np.ndarray`

#### 4. `activity1.py`

This script performs maximum intensity projection (MIP) and closest index different from zero projection (CIP) on DICOM images.

- **Functions:**
  - `MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray`
  - `closest_index_different_from_zero_coronal_plane(img) -> np.ndarray`
  - `rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray`
  - `main()`

#### 5. `activity2.py`

This script handles the coregistration of input DICOM images with reference images using optimization techniques to find the best transformation parameters.

- **Functions:**
  - `mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> List`
  - `found_best_coregistration(ref_img: np.ndarray, input_img: np.ndarray) -> tuple[float, ...]`
  - `main()`

## Installation

To run the scripts, you need to follow these steps:

0. Install Miniconda following the instructions in the [official website](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/CristianComellas21/medical-image-project.git
cd medical-image-project
```

2. Create a new Conda environment with Python 3.11.0 and activate it:

```bash
conda env create -n {env_name} python=3.11.0
conda activate {env_name}
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

Now you can run the scripts using the provided command line arguments.

## Usage

### Activity 1 Script Usage

The `activity1.py` script performs maximum intensity projection (MIP) and closest index different from zero projection (CIP) on DICOM images. You can choose between these two modes and specify the number of projections to create.

#### Command Line Arguments

- `-m, --mode`: Mode of projection, either "MIP" or "CIP". Default is "MIP".
- `-n, --n_projections`: Number of projections to create. Default is 260.

#### Running the Script

To run the script with the default settings (MIP mode with 260 projections):

```bash
python activity1.py
```

To run the script with CIP mode and 100 projections:

```bash
python activity1.py -m CIP -n 100
```

#### Example

Here is an example command to create 360 projections in MIP mode:

```bash
python activity1.py -m MIP -n 360
```

This will generate the specified number of projections and save them in the `results/activity1` directory.

### Activity 2 Script Usage

The `activity2.py` script handles the coregistration of input DICOM images with reference images using optimization techniques to find the best transformation parameters.

#### Command Line Arguments

- `-o, --override`: Override the transformation parameters and recalculate them.
- `-g, --generate_gif`: Generate GIFs with the results.
- `-p, --plot`: Plot the results.
- `-3, --three_d`: Generate 3D GIFs.

#### Running the Script

To run the script with default settings:

```bash
python activity2.py
```

To override the transformation parameters and generate GIFs with plots:

```bash
python activity2.py -o -g -p -3
```

### Visualize Script Usage

The `visualize.py` script provides functionalities for visualizing DICOM images with interactive plots using Matplotlib.

#### Running the Script

To run the script:

```bash
python visualize.py
```

This will load the DICOM files from the specified folder and create interactive plots for visualization.

## Data

Ensure the DICOM files are organized in the expected directory structure as mentioned in the scripts, e.g., `data/HCC-TACE-Seg/HCC_003`.

## Results

Results such as projections and animations are saved in the `results` folder. Make sure this folder exists or is created during the script execution.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
