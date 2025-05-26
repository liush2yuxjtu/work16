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

## Usage

The notebook contains a standalone implementation that can be run directly. The main class `RandomIntensityFromLabels` can be integrated into any MONAI transform pipeline.

Example usage:

```python
import monai.transforms as mt
from monai.transforms import Compose

# Configure the intensity sampling
config = SamplingConfig(
    label_intensities={
        0: (0, 50),      # Background
        1: (100, 150),   # Label 1
        2: (200, 250)    # Label 2
    },
    label_distributions={
        0: "uniform",    # Uniform distribution for background
        1: "normal",     # Normal distribution for Label 1
        2: "uniform"     # Uniform distribution for Label 2
    },
    label_std={
        1: 10.0          # Standard deviation for normal distribution (Label 1)
    }
)

# Create a transform pipeline
transforms = Compose([
    # Other transforms...
    RandomIntensityFromLabels(
        label_key="label",
        image_key="image",
        config=config,
        clamp_output_min=0.0,
        clamp_output_max=255.0
    )
])

# Apply to your data
result = transforms(data)
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