# Medical Image Synthesis with MONAI

This repository contains a Jupyter notebook implementation for medical image synthesis using MONAI transforms. The main focus is on generating realistic medical images from segmentation maps using random intensity sampling techniques.

## Overview

The notebook (`Untitled16.ipynb`) implements a custom MONAI transform called `RandomIntensityFromLabels` that can generate realistic-looking medical images from segmentation maps by sampling intensity values for each label according to specified distributions.

## Features

- Custom MONAI transform for medical image synthesis
- Configurable intensity sampling for different tissue types
- Support for various intensity distributions (uniform, normal, etc.)
- Integration with MONAI's transform pipeline
- Visualization tools for generated images
- Test-Driven Development (TDD) approach with comprehensive test suite

## Dependencies

- MONAI
- PyTorch
- NumPy
- Matplotlib
- Nibabel (for NIfTI file handling)

## Full Pipeline Description

The image synthesis pipeline consists of the following components:

1. **Input Preparation**:
   - Load segmentation maps (label maps) in various formats (NIfTI, NumPy arrays, PyTorch tensors)
   - Preprocess label maps if needed (resizing, normalization, etc.)

2. **Configuration**:
   - Define intensity distributions for each label/tissue type using the `SamplingConfig` class
   - Configure distribution types (uniform, normal) and parameters for each label
   - Set global parameters like output data type and clamping values

3. **Image Synthesis**:
   - Apply the `RandomIntensityFromLabels` transform to generate synthetic images
   - For each label in the segmentation map:
     - Sample intensities from the configured distribution
     - Assign sampled intensities to the corresponding voxels in the output image
   - Apply optional post-processing (clamping, normalization)

4. **Integration with MONAI Pipeline**:
   - Combine with other MONAI transforms in a transform pipeline
   - Use as part of data augmentation or synthetic data generation workflows

5. **Visualization and Evaluation**:
   - Display generated images alongside original segmentation maps
   - Analyze intensity distributions and image statistics
   - Compare with real medical images if available

## Testing

The repository includes a comprehensive test suite (`test_image_synthesis.py`) that follows a Test-Driven Development (TDD) approach. The test file:

- Tests each component individually using try-except blocks instead of unittest
- Verifies the functionality of the `SamplingConfig` class
- Tests the `RandomIntensityFromLabels` transform with various inputs:
  - NumPy arrays
  - PyTorch tensors
  - With and without batch dimensions
- Includes edge case testing (empty label maps, missing keys, etc.)
- Validates intensity distributions in the generated images

To run the tests:

```bash
python test_image_synthesis.py
```

## Visualization

The repository includes two scripts for visualization:

### Basic Visualization (`run_clean_notebook.py`)

This script generates basic visualizations of the synthetic images and saves them to the `figures/` directory:

- Creates simple 3D shapes (sphere, cube) and generates synthetic images
- Creates a more complex example with multiple anatomical structures (brain, tumor, blood vessels, lesions)
- Demonstrates the use of different intensity distributions (uniform, normal)
- Generates multiple random samples from the same label map
- Saves visualizations of different slices to show the 3D nature of the synthesis

To run the basic visualization script:

```bash
python run_clean_notebook.py
```

### Full Pipeline Visualization (`full_pipeline.py`)

This script implements a comprehensive end-to-end pipeline for medical image synthesis and saves the results to the `pipeline_figures/` directory:

- Creates a realistic synthetic segmentation with multiple anatomical structures
- Generates multiple modalities (T1-weighted, T2-weighted) from the same segmentation
- Adds realistic noise and artifacts (Gaussian noise, bias field) to simulate real medical images
- Saves all data as NIfTI files (`.nii.gz`) for compatibility with medical imaging software
- Creates detailed visualizations of all modalities and processing steps

To run the full pipeline script:

```bash
python full_pipeline.py
```

### Example Visualizations

#### Basic Visualizations (`figures/` directory)

1. **Simple Shapes**: Visualizations of basic 3D shapes with different intensity distributions
   - `simple_shapes_slice_*.png`: Different slices through the 3D volume

2. **Advanced Example**: More complex anatomical structures with different intensity distributions
   - `advanced_example_slice_*.png`: Different slices through the 3D volume
   - `advanced_example_multi_slice.png`: Multiple slices in a single figure

3. **Multiple Samples**: Different random samples generated from the same label map
   - `multiple_samples.png`: Shows how different random seeds produce different intensity patterns

#### Full Pipeline Visualizations (`pipeline_figures/` directory)

1. **Multi-Modal Comparison**: Visualizations of different modalities and processing steps
   - `full_pipeline_slice_*.png`: Comparison of segmentation, T1, T2, and processed images
   - `modality_comparison.png`: Side-by-side comparison of all modalities for a single slice

2. **Segmentation Overlay**: Visualization of the segmentation overlaid on the T1 image
   - `segmentation_overlay.png`: Shows how the segmentation corresponds to the generated image

3. **3D Visualization**: Multiple slices to show the 3D nature of the data
   - `t1_3d_visualization.png`: Grid of slices through the T1 volume

4. **NIfTI Files**: Medical imaging format files for use with specialized software
   - `*.nii.gz`: Compressed NIfTI files for each modality and processing step

These visualizations demonstrate the flexibility of the `RandomIntensityFromLabels` transform for generating synthetic medical images with controlled intensity distributions and realistic artifacts.

## Usage

### Basic Usage

The main class `RandomIntensityFromLabels` can be integrated into any MONAI transform pipeline:

```python
from monai.transforms import Compose
from your_module import RandomIntensityFromLabels, SamplingConfig

# Create a sampling configuration
config = SamplingConfig(
    label_intensities={
        0: (0, 50),      # Background
        1: (100, 150),   # Structure 1
        2: (200, 250)    # Structure 2
    },
    label_distributions={
        0: "uniform",
        1: "normal",
        2: "uniform"
    },
    label_std={
        1: 10.0
    }
)

# Create a transform pipeline
transform = Compose([
    RandomIntensityFromLabels(
        label_key="label",
        image_key="image",
        config=config,
        clamp_output_min=0.0,
        clamp_output_max=255.0
    ),
    # Add other transforms as needed
])

# Apply the transform
data = {"label": your_segmentation}
result = transform(data)
```

### Full Pipeline Usage

For a complete end-to-end pipeline, you can use the `full_pipeline.py` script which demonstrates:

1. Creating synthetic segmentations
2. Generating multiple modalities (T1, T2)
3. Adding realistic noise and artifacts
4. Saving as NIfTI files
5. Visualizing results

The full pipeline can be run as:

```bash
python full_pipeline.py
```

Or imported and used in your own code:

```python
from full_pipeline import create_synthetic_segmentation, create_intensity_config, generate_synthetic_images

# Create a segmentation
segmentation = create_synthetic_segmentation()

# Create intensity configurations for different modalities
configs = create_intensity_config()

# Generate synthetic images
data = generate_synthetic_images(segmentation, configs)

# Use the generated images in your application
t1_image = data["t1"]
t2_image = data["t2"]
```

## Implementation Details

The implementation includes:

1. **SamplingConfig Class**:
   - Configures intensity sampling parameters for each label
   - Supports different distribution types per label
   - Provides default values for unspecified parameters

2. **RandomIntensityFromLabels Transform**:
   - Generates synthetic images from label maps
   - Handles both NumPy arrays and PyTorch tensors
   - Supports batch processing
   - Maintains input data types and dimensions

3. **Distribution Sampling**:
   - Uniform distribution: Samples values uniformly between min and max
   - Normal distribution: Samples from normal distribution with specified mean and standard deviation
   - Extensible to other distribution types

4. **Visualization Utilities**:
   - Functions to display generated images
   - Tools to analyze intensity distributions

## Advanced Usage

### Handling Batch Dimensions

The transform can handle inputs with batch dimensions:

```python
# Input with batch dimension [B, H, W, D]
batch_data = {"label": batch_label_map}
result = transform(batch_data)
# Output will maintain batch dimension: result["image"].shape == batch_label_map.shape
```

### Custom Distribution Types

You can extend the transform to support additional distribution types:

```python
# In the __call__ method of RandomIntensityFromLabels
if distribution == "uniform":
    # Uniform distribution between min and max
    min_val, max_val = intensity_range
    intensities = np.random.uniform(min_val, max_val, size=np.sum(mask))
elif distribution == "normal":
    # Normal distribution with mean at center of range and specified std
    min_val, max_val = intensity_range
    mean = (min_val + max_val) / 2
    intensities = np.random.normal(mean, std, size=np.sum(mask))
elif distribution == "your_custom_distribution":
    # Implement your custom distribution sampling here
    pass
```

## License

This project is open-source and available under standard open-source licenses.

## Acknowledgments

- [MONAI Project](https://github.com/Project-MONAI/MONAI)
- [lab2im](https://github.com/BBillot/lab2im) for inspiration