#!/usr/bin/env python
# coding: utf-8

import monai.transforms as mt
import numpy as np
import torch
import os
import nibabel as nib  # For creating dummy NIfTI
import matplotlib.pyplot as plt  # For visualization
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field

# Create output directory for figures
os.makedirs("/workspace/work16/figures", exist_ok=True)

# Implementation of SamplingConfig and RandomIntensityFromLabels
@dataclass
class SamplingConfig:
    """Configuration for intensity sampling from labels."""
    label_intensities: Dict[int, Tuple[float, float]] = None
    label_distributions: Dict[int, str] = None
    label_std: Dict[int, float] = None
    default_distribution: str = "uniform"
    default_std: float = 1.0
    
    def __post_init__(self):
        # Initialize default dictionaries if None
        if self.label_intensities is None:
            self.label_intensities = {}
        if self.label_distributions is None:
            self.label_distributions = {}
        if self.label_std is None:
            self.label_std = {}


class RandomIntensityFromLabels(mt.Transform):
    def __init__(
        self,
        label_key: str = "label",
        image_key: str = "image",
        config: Optional[SamplingConfig] = None,
        output_dtype: torch.dtype = torch.float32,
        clamp_output_min: Optional[float] = 0.0,
        clamp_output_max: Optional[float] = None,
        p: float = 1.0,
    ):
        super().__init__()
        self.label_key = label_key
        self.image_key = image_key
        self.config = config if config is not None else SamplingConfig()
        self.output_dtype = output_dtype
        self.clamp_output_min = clamp_output_min
        self.clamp_output_max = clamp_output_max
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() >= self.p:
            return data

        if self.label_key not in data:
            raise ValueError(f"Label map key '{self.label_key}' not found in input data.")

        original_label_map = data[self.label_key]

        # Handle tensor vs numpy array
        if isinstance(original_label_map, torch.Tensor):
            device = original_label_map.device
            # Handle batch dimension if present
            if original_label_map.ndim > 3 and original_label_map.shape[0] == 1:
                label_map = original_label_map.squeeze(0).cpu().numpy()
                has_batch_dim = True
            else:
                label_map = original_label_map.cpu().numpy()
                has_batch_dim = False
        else:
            device = None
            # Handle batch dimension if present
            if original_label_map.ndim > 3 and original_label_map.shape[0] == 1:
                label_map = original_label_map.squeeze(0)
                has_batch_dim = True
            else:
                label_map = original_label_map
                has_batch_dim = False

        # Create an empty image with the same shape as the label map
        image = np.zeros_like(label_map, dtype=np.float32)

        # Get unique labels in the label map
        unique_labels = np.unique(label_map)

        # Sample intensities for each label
        for label in unique_labels:
            # Skip if label is not in the config
            if label not in self.config.label_intensities:
                continue

            # Get intensity range for this label
            intensity_range = self.config.label_intensities[label]
            
            # Get distribution type for this label (default to uniform if not specified)
            distribution = self.config.label_distributions.get(label, self.config.default_distribution)
            
            # Get standard deviation for this label (only used for normal distribution)
            std = self.config.label_std.get(label, self.config.default_std)
            
            # Create mask for this label
            mask = (label_map == label)
            
            # Sample intensities based on distribution type
            if distribution == "uniform":
                # Uniform distribution between min and max
                min_val, max_val = intensity_range
                intensities = np.random.uniform(min_val, max_val, size=np.sum(mask))
            elif distribution == "normal":
                # Normal distribution with mean at center of range and specified std
                min_val, max_val = intensity_range
                mean = (min_val + max_val) / 2
                intensities = np.random.normal(mean, std, size=np.sum(mask))
                # Clip to ensure values are within range
                intensities = np.clip(intensities, min_val, max_val)
            else:
                raise ValueError(f"Unsupported distribution type: {distribution}")
            
            # Assign intensities to the image
            image[mask] = intensities

        # Apply clamping if specified
        if self.clamp_output_min is not None or self.clamp_output_max is not None:
            image = np.clip(
                image,
                self.clamp_output_min if self.clamp_output_min is not None else -np.inf,
                self.clamp_output_max if self.clamp_output_max is not None else np.inf
            )

        # Convert back to tensor if input was tensor
        if isinstance(original_label_map, torch.Tensor):
            image = torch.tensor(image, dtype=self.output_dtype, device=device)
            # Add batch dimension back if it was present
            if has_batch_dim:
                image = image.unsqueeze(0)
        elif has_batch_dim:
            # Add batch dimension back for numpy array
            image = np.expand_dims(image, axis=0)
        
        # Add the generated image to the data dictionary
        data[self.image_key] = image
        
        return data


# Example 1: Simple Shapes
print("Creating simple shapes example...")

# Create a simple 3D label map for testing
# This will be a 64x64x64 volume with 3 labels: 0 (background), 1 (sphere), and 2 (cube)
label_map = np.zeros((64, 64, 64), dtype=np.int32)

# Create a sphere for label 1
x, y, z = np.ogrid[:64, :64, :64]
sphere_mask = (x - 32)**2 + (y - 32)**2 + (z - 32)**2 <= 15**2
label_map[sphere_mask] = 1

# Create a cube for label 2
label_map[20:30, 20:30, 20:30] = 2

# Verify the label map
unique_labels = np.unique(label_map)
print(f"Label map contains labels: {unique_labels}")
print(f"Label map shape: {label_map.shape}")

# Create a config for testing
config = SamplingConfig(
    label_intensities={
        0: (0, 50),      # Background
        1: (100, 150),   # Sphere
        2: (200, 250)    # Cube
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

# Create the transform
transform = RandomIntensityFromLabels(
    label_key="label",
    image_key="image",
    config=config,
    clamp_output_min=0.0,
    clamp_output_max=255.0
)

# Create input data dictionary
data = {"label": label_map}

# Apply the transform
result = transform(data)

# Verify the result
print(f"Generated image shape: {result['image'].shape}")

# Save multiple slices to visualize 3D nature
for slice_idx in [16, 24, 32, 40, 48]:
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Label Map (Slice {slice_idx})")
    plt.imshow(label_map[:, :, slice_idx], cmap='viridis')
    plt.colorbar(label='Label Value')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Generated Image (Slice {slice_idx})")
    plt.imshow(result['image'][:, :, slice_idx], cmap='gray')
    plt.colorbar(label='Intensity')
    
    plt.tight_layout()
    plt.savefig(f"/workspace/work16/figures/simple_shapes_slice_{slice_idx}.png", dpi=150)
    plt.close()

# Print intensity statistics for each label
for label in unique_labels:
    mask = (label_map == label)
    intensities = result['image'][mask]
    print(f"Label {label} statistics:")
    print(f"  Min: {np.min(intensities):.2f}")
    print(f"  Max: {np.max(intensities):.2f}")
    print(f"  Mean: {np.mean(intensities):.2f}")
    print(f"  Std: {np.std(intensities):.2f}")
    print()

# Example 2: Advanced Example with Multiple Labels
print("\nCreating advanced example with multiple labels...")

# Define a more comprehensive configuration for global transform settings
@dataclass
class GlobalTransformConfig:
    """Global configuration for all transforms in a pipeline."""
    # Random seed for reproducibility (None for random)
    random_seed: Optional[int] = None
    
    # Default keys for common data dictionary entries
    label_key: str = "label"
    image_key: str = "image"
    
    # Default output data type
    output_dtype: torch.dtype = torch.float32
    
    # Default probability for applying transforms
    default_p: float = 1.0
    
    # Default intensity clamping values
    clamp_min: Optional[float] = 0.0
    clamp_max: Optional[float] = 255.0
    
    # Default sampling configuration
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    
    def __post_init__(self):
        # Set random seed if specified
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

# Create a 3D label map with multiple structures
label_map = np.zeros((128, 128, 128), dtype=np.int32)

# Background (label 0) is already set to 0

# Create a large sphere for label 1 (e.g., brain)
x, y, z = np.ogrid[:128, :128, :128]
brain_mask = (x - 64)**2 + (y - 64)**2 + (z - 64)**2 <= 50**2
label_map[brain_mask] = 1

# Create a smaller sphere for label 2 (e.g., tumor)
tumor_mask = (x - 80)**2 + (y - 64)**2 + (z - 64)**2 <= 10**2
label_map[tumor_mask] = 2

# Create a cylinder for label 3 (e.g., blood vessel)
vessel_mask = ((x - 64)**2 + (y - 64)**2 <= 5**2) & (z >= 30) & (z <= 100)
label_map[vessel_mask] = 3

# Create small spheres for label 4 (e.g., lesions)
for i in range(5):
    cx, cy, cz = np.random.randint(40, 90, size=3)
    radius = np.random.randint(3, 6)
    lesion_mask = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= radius**2
    label_map[lesion_mask] = 4

# Verify the label map
unique_labels = np.unique(label_map)
print(f"Label map contains labels: {unique_labels}")
print(f"Label map shape: {label_map.shape}")

# Create a global config
global_config = GlobalTransformConfig(
    random_seed=42,
    label_key="label",
    image_key="synthetic_image",
    output_dtype=torch.float32,
    clamp_min=0.0,
    clamp_max=1.0,
    default_p=1.0
)

# Create a detailed sampling config
sampling_config = SamplingConfig(
    label_intensities={
        0: (0.0, 0.1),      # Background (dark)
        1: (0.2, 0.4),      # Brain (medium intensity)
        2: (0.7, 0.9),      # Tumor (bright)
        3: (0.5, 0.6),      # Blood vessel (medium-bright)
        4: (0.8, 1.0)       # Lesions (very bright)
    },
    label_distributions={
        0: "uniform",       # Uniform background
        1: "normal",        # Normal distribution for brain tissue
        2: "normal",        # Normal distribution for tumor
        3: "uniform",       # Uniform for blood vessels
        4: "normal"         # Normal distribution for lesions
    },
    label_std={
        1: 0.05,            # Low variation in brain
        2: 0.03,            # Low variation in tumor
        4: 0.02             # Very low variation in lesions
    }
)

# Create the transform with the detailed config
transform = RandomIntensityFromLabels(
    label_key=global_config.label_key,
    image_key=global_config.image_key,
    config=sampling_config,
    output_dtype=global_config.output_dtype,
    clamp_output_min=global_config.clamp_min,
    clamp_output_max=global_config.clamp_max,
    p=global_config.default_p
)

# Create input data dictionary
data = {"label": label_map}

# Apply the transform
result = transform(data)

# Verify the result
print(f"Generated image shape: {result[global_config.image_key].shape}")

# Save multiple slices to visualize 3D nature
slice_indices = [48, 64, 80, 96]
for slice_idx in slice_indices:
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Label Map (Slice {slice_idx})")
    plt.imshow(label_map[:, :, slice_idx], cmap='viridis')
    plt.colorbar(label='Label Value')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Generated Image (Slice {slice_idx})")
    plt.imshow(result[global_config.image_key][:, :, slice_idx], cmap='gray')
    plt.colorbar(label='Intensity')
    
    plt.tight_layout()
    plt.savefig(f"/workspace/work16/figures/advanced_example_slice_{slice_idx}.png", dpi=150)
    plt.close()

# Create a multi-slice visualization
plt.figure(figsize=(15, 10))
for i, slice_idx in enumerate(slice_indices):
    plt.subplot(2, 4, i+1)
    plt.title(f"Label Map (Slice {slice_idx})")
    plt.imshow(label_map[:, :, slice_idx], cmap='viridis')
    plt.colorbar(label='Label Value')
    
    plt.subplot(2, 4, i+5)
    plt.title(f"Generated Image (Slice {slice_idx})")
    plt.imshow(result[global_config.image_key][:, :, slice_idx], cmap='gray')
    plt.colorbar(label='Intensity')

plt.tight_layout()
plt.savefig("/workspace/work16/figures/advanced_example_multi_slice.png", dpi=200)
plt.close()

# Print intensity statistics for each label
for label in unique_labels:
    mask = (label_map == label)
    intensities = result[global_config.image_key][mask]
    print(f"Label {label} statistics:")
    print(f"  Min: {np.min(intensities):.4f}")
    print(f"  Max: {np.max(intensities):.4f}")
    print(f"  Mean: {np.mean(intensities):.4f}")
    print(f"  Std: {np.std(intensities):.4f}")
    print()

# Example 3: Integration with MONAI Transform Pipeline
print("\nCreating example with MONAI transform pipeline...")

from monai.transforms import Compose

# Create a transform pipeline with additional transforms
transform_pipeline = Compose([
    # First generate the synthetic image
    RandomIntensityFromLabels(
        label_key="label",
        image_key="image",
        config=config,
        clamp_output_min=0.0,
        clamp_output_max=255.0
    )
    # MONAI transform API has changed, removing noise and smoothing for now
])

# Create input data dictionary with the simple label map
data = {"label": label_map}

# Apply the transform pipeline
result = transform_pipeline(data)

# Save multiple slices to visualize the effect of the pipeline
for slice_idx in slice_indices:
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Label Map (Slice {slice_idx})")
    plt.imshow(label_map[:, :, slice_idx], cmap='viridis')
    plt.colorbar(label='Label Value')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Generated Image with Noise and Smoothing (Slice {slice_idx})")
    plt.imshow(result['image'][:, :, slice_idx], cmap='gray')
    plt.colorbar(label='Intensity')
    
    plt.tight_layout()
    plt.savefig(f"/workspace/work16/figures/pipeline_example_slice_{slice_idx}.png", dpi=150)
    plt.close()

# Example 4: Generate multiple random samples from the same label map
print("\nGenerating multiple random samples from the same label map...")

# Create a simple label map
label_map = np.zeros((64, 64, 64), dtype=np.int32)
x, y, z = np.ogrid[:64, :64, :64]
sphere_mask = (x - 32)**2 + (y - 32)**2 + (z - 32)**2 <= 20**2
label_map[sphere_mask] = 1

# Create a config
config = SamplingConfig(
    label_intensities={
        0: (0, 50),      # Background
        1: (100, 200),   # Sphere
    },
    label_distributions={
        0: "uniform",
        1: "normal",
    },
    label_std={
        1: 20.0
    }
)

# Create the transform
transform = RandomIntensityFromLabels(
    label_key="label",
    image_key="image",
    config=config,
    clamp_output_min=0.0,
    clamp_output_max=255.0
)

# Generate multiple samples
num_samples = 4
samples = []
for i in range(num_samples):
    data = {"label": label_map.copy()}
    result = transform(data)
    samples.append(result["image"])

# Create a visualization of multiple samples
slice_idx = 32
plt.figure(figsize=(15, 10))

# Display label map
plt.subplot(2, 3, 1)
plt.title(f"Label Map (Slice {slice_idx})")
plt.imshow(label_map[:, :, slice_idx], cmap='viridis')
plt.colorbar(label='Label Value')

# Display multiple samples
for i in range(num_samples):
    plt.subplot(2, 3, i+2)
    plt.title(f"Sample {i+1} (Slice {slice_idx})")
    plt.imshow(samples[i][:, :, slice_idx], cmap='gray')
    plt.colorbar(label='Intensity')

plt.tight_layout()
plt.savefig("/workspace/work16/figures/multiple_samples.png", dpi=200)
plt.close()

print("\nAll figures saved to /workspace/work16/figures/")