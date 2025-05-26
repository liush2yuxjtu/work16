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

## Dependencies

- MONAI
- PyTorch
- NumPy
- Matplotlib
- Nibabel (for NIfTI file handling)

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
    }
)

# Create a transform pipeline
transforms = Compose([
    # Other transforms...
    RandomIntensityFromLabels(
        label_key="label",
        image_key="image",
        config=config
    )
])

# Apply to your data
result = transforms(data)
```

## Implementation Details

The implementation includes:

1. A `SamplingConfig` class to configure intensity sampling parameters
2. A `RandomIntensityFromLabels` transform that generates images from label maps
3. Support for various intensity distributions and noise models
4. Visualization utilities for the generated images

## License

This project is open-source and available under standard open-source licenses.

## Acknowledgments

- [MONAI Project](https://github.com/Project-MONAI/MONAI)
- [lab2im](https://github.com/BBillot/lab2im) for inspiration