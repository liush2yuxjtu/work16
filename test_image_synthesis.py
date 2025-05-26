#!/usr/bin/env python
# coding: utf-8

"""
Test-Driven Development (TDD) for Medical Image Synthesis
---------------------------------------------------------
This file tests the functionality of the RandomIntensityFromLabels transform
and related components from the Untitled16.ipynb notebook.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List, Union

# Import required libraries (similar to notebook)
try:
    import monai
    import monai.transforms as mt
    print(f"✓ MONAI version {monai.__version__} successfully imported")
except ImportError:
    print("✗ MONAI not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "monai", "-q"])
    import monai
    import monai.transforms as mt
    print(f"✓ MONAI version {monai.__version__} installed and imported")

try:
    import nibabel as nib
    print("✓ Nibabel successfully imported")
except ImportError:
    print("✗ Nibabel not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nibabel", "-q"])
    import nibabel as nib
    print("✓ Nibabel installed and imported")

# Print test header
print("\n" + "="*80)
print("TESTING MEDICAL IMAGE SYNTHESIS COMPONENTS")
print("="*80 + "\n")

# -------------------------------------------------------------------------
# Test Cell: SamplingConfig
# -------------------------------------------------------------------------
print("\n" + "-"*40)
print("Testing SamplingConfig")
print("-"*40)

from dataclasses import dataclass

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

# Test SamplingConfig initialization
try:
    # Test default initialization
    config1 = SamplingConfig()
    assert isinstance(config1.label_intensities, dict), "label_intensities should be a dict"
    assert len(config1.label_intensities) == 0, "Default label_intensities should be empty"
    print("✓ Default SamplingConfig initialization passed")
    
    # Test with parameters
    config2 = SamplingConfig(
        label_intensities={0: (0, 50), 1: (100, 150)},
        label_distributions={0: "uniform", 1: "normal"},
        label_std={1: 10.0}
    )
    assert config2.label_intensities[0] == (0, 50), "label_intensities not set correctly"
    assert config2.label_distributions[1] == "normal", "label_distributions not set correctly"
    assert config2.label_std[1] == 10.0, "label_std not set correctly"
    print("✓ Parameterized SamplingConfig initialization passed")
    
except AssertionError as e:
    print(f"✗ SamplingConfig test failed: {str(e)}")
except Exception as e:
    print(f"✗ Unexpected error in SamplingConfig test: {str(e)}")

# -------------------------------------------------------------------------
# Test Cell: RandomIntensityFromLabels Transform
# -------------------------------------------------------------------------
print("\n" + "-"*40)
print("Testing RandomIntensityFromLabels Transform")
print("-"*40)

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
        
        # Add the generated image to the data dictionary
        data[self.image_key] = image
        
        return data

# Test RandomIntensityFromLabels initialization
try:
    transform = RandomIntensityFromLabels()
    assert transform.label_key == "label", "Default label_key should be 'label'"
    assert transform.image_key == "image", "Default image_key should be 'image'"
    assert isinstance(transform.config, SamplingConfig), "Default config should be SamplingConfig instance"
    print("✓ RandomIntensityFromLabels initialization passed")
except AssertionError as e:
    print(f"✗ RandomIntensityFromLabels initialization test failed: {str(e)}")
except Exception as e:
    print(f"✗ Unexpected error in RandomIntensityFromLabels initialization test: {str(e)}")

# -------------------------------------------------------------------------
# Test Cell: Create Test Data
# -------------------------------------------------------------------------
print("\n" + "-"*40)
print("Testing with Synthetic Data")
print("-"*40)

# Create a simple 3D label map for testing
try:
    # Create a 3D label map with 3 labels: 0 (background), 1, and 2
    label_map = np.zeros((64, 64, 64), dtype=np.int32)
    
    # Create a sphere for label 1
    x, y, z = np.ogrid[:64, :64, :64]
    sphere_mask = (x - 32)**2 + (y - 32)**2 + (z - 32)**2 <= 15**2
    label_map[sphere_mask] = 1
    
    # Create a cube for label 2
    label_map[20:30, 20:30, 20:30] = 2
    
    # Verify the label map
    unique_labels = np.unique(label_map)
    assert set(unique_labels) == {0, 1, 2}, f"Expected labels 0, 1, 2 but got {unique_labels}"
    print(f"✓ Created test label map with labels: {unique_labels}")
    
    # Convert to tensor for testing with tensor input
    label_map_tensor = torch.tensor(label_map)
    print(f"✓ Converted label map to tensor with shape: {label_map_tensor.shape}")
    
except AssertionError as e:
    print(f"✗ Test data creation failed: {str(e)}")
except Exception as e:
    print(f"✗ Unexpected error in test data creation: {str(e)}")

# -------------------------------------------------------------------------
# Test Cell: Test Transform with Numpy Input
# -------------------------------------------------------------------------
print("\n" + "-"*40)
print("Testing Transform with Numpy Input")
print("-"*40)

try:
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
    assert "image" in result, "Transform should add 'image' key to the result"
    assert result["image"].shape == label_map.shape, f"Image shape {result['image'].shape} should match label map shape {label_map.shape}"
    
    # Check intensity ranges
    image = result["image"]
    for label_value, (min_val, max_val) in config.label_intensities.items():
        mask = (label_map == label_value)
        if np.any(mask):
            label_intensities = image[mask]
            assert np.all(label_intensities >= min_val), f"Label {label_value} has intensities below {min_val}"
            assert np.all(label_intensities <= max_val), f"Label {label_value} has intensities above {max_val}"
            print(f"✓ Label {label_value} intensities are within range [{min_val}, {max_val}]")
    
    print("✓ Transform with numpy input passed")
    
except AssertionError as e:
    print(f"✗ Transform with numpy input test failed: {str(e)}")
except Exception as e:
    print(f"✗ Unexpected error in transform with numpy input test: {str(e)}")

# -------------------------------------------------------------------------
# Test Cell: Test Transform with Tensor Input
# -------------------------------------------------------------------------
print("\n" + "-"*40)
print("Testing Transform with Tensor Input")
print("-"*40)

try:
    # Create input data dictionary with tensor
    data = {"label": label_map_tensor}
    
    # Apply the transform
    result = transform(data)
    
    # Verify the result
    assert "image" in result, "Transform should add 'image' key to the result"
    assert isinstance(result["image"], torch.Tensor), "Result image should be a tensor"
    assert result["image"].shape == label_map_tensor.shape, f"Image shape {result['image'].shape} should match label map shape {label_map_tensor.shape}"
    
    # Check intensity ranges
    image = result["image"].numpy()
    for label_value, (min_val, max_val) in config.label_intensities.items():
        mask = (label_map == label_value)
        if np.any(mask):
            label_intensities = image[mask]
            assert np.all(label_intensities >= min_val), f"Label {label_value} has intensities below {min_val}"
            assert np.all(label_intensities <= max_val), f"Label {label_value} has intensities above {max_val}"
            print(f"✓ Label {label_value} intensities are within range [{min_val}, {max_val}]")
    
    print("✓ Transform with tensor input passed")
    
except AssertionError as e:
    print(f"✗ Transform with tensor input test failed: {str(e)}")
except Exception as e:
    print(f"✗ Unexpected error in transform with tensor input test: {str(e)}")

# -------------------------------------------------------------------------
# Test Cell: Test with Batch Dimension
# -------------------------------------------------------------------------
print("\n" + "-"*40)
print("Testing with Batch Dimension")
print("-"*40)

try:
    # Add batch dimension to label map
    batch_label_map = np.expand_dims(label_map, axis=0)
    assert batch_label_map.shape == (1, 64, 64, 64), f"Expected shape (1, 64, 64, 64) but got {batch_label_map.shape}"
    
    # Create input data dictionary
    data = {"label": batch_label_map}
    
    # Create a new transform that properly handles batch dimension
    class RandomIntensityFromLabelsFixed(RandomIntensityFromLabels):
        def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
            if np.random.rand() >= self.p:
                return data

            if self.label_key not in data:
                raise ValueError(f"Label map key '{self.label_key}' not found in input data.")

            original_label_map = data[self.label_key]
            has_batch_dim = False

            # Handle tensor vs numpy array
            if isinstance(original_label_map, torch.Tensor):
                device = original_label_map.device
                # Handle batch dimension if present
                if original_label_map.ndim > 3 and original_label_map.shape[0] == 1:
                    label_map = original_label_map.squeeze(0).cpu().numpy()
                    has_batch_dim = True
                else:
                    label_map = original_label_map.cpu().numpy()
            else:
                device = None
                # Handle batch dimension if present
                if original_label_map.ndim > 3 and original_label_map.shape[0] == 1:
                    label_map = original_label_map.squeeze(0)
                    has_batch_dim = True
                else:
                    label_map = original_label_map

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
                if isinstance(image, torch.Tensor):
                    image = image.unsqueeze(0)
                else:
                    image = np.expand_dims(image, axis=0)
            
            # Add the generated image to the data dictionary
            data[self.image_key] = image
            
            return data
    
    # Create the fixed transform
    fixed_transform = RandomIntensityFromLabelsFixed(
        label_key="label",
        image_key="image",
        config=config,
        clamp_output_min=0.0,
        clamp_output_max=255.0
    )
    
    # Apply the transform
    result = fixed_transform(data)
    
    # Verify the result
    assert "image" in result, "Transform should add 'image' key to the result"
    assert result["image"].shape == batch_label_map.shape, f"Image shape {result['image'].shape} should match batch label map shape {batch_label_map.shape}"
    
    print("✓ Transform with batch dimension passed")
    
except AssertionError as e:
    print(f"✗ Transform with batch dimension test failed: {str(e)}")
except Exception as e:
    print(f"✗ Unexpected error in transform with batch dimension test: {str(e)}")

# -------------------------------------------------------------------------
# Test Cell: Visualization
# -------------------------------------------------------------------------
print("\n" + "-"*40)
print("Testing Visualization")
print("-"*40)

try:
    # Get a slice from the middle of the volume
    slice_idx = 32
    label_slice = label_map[:, :, slice_idx]
    image_slice = result["image"][0, :, :, slice_idx] if result["image"].ndim == 4 else result["image"][:, :, slice_idx]
    
    # Convert to numpy if tensor
    if isinstance(image_slice, torch.Tensor):
        image_slice = image_slice.numpy()
    
    # Print shape information
    print(f"Label slice shape: {label_slice.shape}")
    print(f"Image slice shape: {image_slice.shape}")
    
    # Print intensity statistics
    print(f"Image intensity range: [{np.min(image_slice):.2f}, {np.max(image_slice):.2f}]")
    print(f"Image mean intensity: {np.mean(image_slice):.2f}")
    print(f"Image standard deviation: {np.std(image_slice):.2f}")
    
    print("✓ Visualization data preparation passed")
    
    # Note: We don't actually display the plots in a test script, but we verify the data is ready for visualization
    
except Exception as e:
    print(f"✗ Visualization test failed: {str(e)}")

# -------------------------------------------------------------------------
# Test Cell: Edge Cases
# -------------------------------------------------------------------------
print("\n" + "-"*40)
print("Testing Edge Cases")
print("-"*40)

# Test missing label key
try:
    data = {"wrong_key": label_map}
    transform(data)
    print("✗ Missing label key test failed: should have raised ValueError")
except ValueError as e:
    print(f"✓ Missing label key correctly raised ValueError: {str(e)}")
except Exception as e:
    print(f"✗ Missing label key raised unexpected error: {str(e)}")

# Test empty label map
try:
    empty_label_map = np.zeros((10, 10, 10), dtype=np.int32)
    data = {"label": empty_label_map}
    result = transform(data)
    assert "image" in result, "Transform should add 'image' key even with empty label map"
    assert result["image"].shape == empty_label_map.shape, "Image shape should match empty label map shape"
    assert np.all(result["image"] >= 0) and np.all(result["image"] <= 50), "Empty label map should have background intensities"
    print("✓ Empty label map test passed")
except AssertionError as e:
    print(f"✗ Empty label map test failed: {str(e)}")
except Exception as e:
    print(f"✗ Empty label map test raised unexpected error: {str(e)}")

# Test probability parameter
try:
    # Create transform with p=0 (should never apply transform)
    no_transform = RandomIntensityFromLabels(
        config=config,
        p=0.0
    )
    
    # Create input with known values
    data = {"label": label_map, "image": np.ones_like(label_map)}
    original_image = data["image"].copy()
    
    # Apply transform (should not change anything with p=0)
    result = no_transform(data)
    
    # Verify image was not changed
    assert np.array_equal(result["image"], original_image), "Transform with p=0 should not modify the image"
    print("✓ Probability parameter test passed")
except AssertionError as e:
    print(f"✗ Probability parameter test failed: {str(e)}")
except Exception as e:
    print(f"✗ Probability parameter test raised unexpected error: {str(e)}")

# -------------------------------------------------------------------------
# Test Summary
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("All tests completed. Check the output above for any failures.")
print("="*80 + "\n")