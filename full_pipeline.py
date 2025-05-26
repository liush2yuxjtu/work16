#!/usr/bin/env python
# coding: utf-8

import monai.transforms as mt
import numpy as np
import torch
import os
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field

# Create output directory for figures
os.makedirs("/workspace/work16/pipeline_figures", exist_ok=True)

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


# Full Pipeline Implementation
print("Starting full pipeline implementation...")

# Step 1: Create a synthetic segmentation mask (simulating a medical image segmentation)
def create_synthetic_segmentation(shape=(128, 128, 128), num_classes=5):
    """Create a synthetic segmentation mask with multiple anatomical structures."""
    print(f"Creating synthetic segmentation with shape {shape} and {num_classes} classes...")
    
    # Initialize empty segmentation
    segmentation = np.zeros(shape, dtype=np.int32)
    
    # Create coordinates grid
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = np.array([s // 2 for s in shape])
    
    # Class 1: Large sphere (e.g., brain)
    brain_radius = min(shape) // 3
    brain_mask = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= brain_radius**2
    segmentation[brain_mask] = 1
    
    # Class 2: Smaller sphere (e.g., tumor)
    tumor_center = center + np.array([brain_radius // 2, 0, 0])
    tumor_radius = brain_radius // 4
    tumor_mask = ((x - tumor_center[0])**2 + (y - tumor_center[1])**2 + (z - tumor_center[2])**2) <= tumor_radius**2
    segmentation[tumor_mask] = 2
    
    # Class 3: Cylinder (e.g., blood vessel)
    vessel_radius = brain_radius // 10
    vessel_mask = ((x - center[0])**2 + (y - center[1])**2 <= vessel_radius**2) & (z >= center[2] - brain_radius//2) & (z <= center[2] + brain_radius//2)
    segmentation[vessel_mask] = 3
    
    # Class 4: Small spheres (e.g., lesions)
    np.random.seed(42)  # For reproducibility
    for i in range(5):
        lesion_center = center + np.random.randint(-brain_radius//2, brain_radius//2, size=3)
        lesion_radius = max(3, brain_radius // 15)
        lesion_mask = ((x - lesion_center[0])**2 + (y - lesion_center[1])**2 + (z - lesion_center[2])**2) <= lesion_radius**2
        # Only place lesions inside the brain
        lesion_mask = lesion_mask & brain_mask
        segmentation[lesion_mask] = 4
    
    # Verify the segmentation
    unique_labels = np.unique(segmentation)
    print(f"Created segmentation with labels: {unique_labels}")
    
    # Count voxels per class
    for label in unique_labels:
        count = np.sum(segmentation == label)
        percentage = 100 * count / np.prod(shape)
        print(f"  Class {label}: {count} voxels ({percentage:.2f}%)")
    
    return segmentation

# Step 2: Create a realistic intensity sampling configuration
def create_intensity_config():
    """Create a configuration for realistic medical image intensities."""
    print("Creating intensity sampling configuration...")
    
    # T1-weighted MRI-like intensities
    t1_config = SamplingConfig(
        label_intensities={
            0: (5, 20),       # Background (dark)
            1: (100, 150),    # Brain tissue (medium intensity)
            2: (180, 220),    # Tumor (bright)
            3: (50, 80),      # Blood vessel (dark)
            4: (160, 200)     # Lesions (bright)
        },
        label_distributions={
            0: "uniform",     # Uniform background
            1: "normal",      # Normal distribution for brain tissue
            2: "normal",      # Normal distribution for tumor
            3: "uniform",     # Uniform for blood vessels
            4: "normal"       # Normal distribution for lesions
        },
        label_std={
            1: 10.0,          # Moderate variation in brain
            2: 8.0,           # Moderate variation in tumor
            4: 5.0            # Low variation in lesions
        }
    )
    
    # T2-weighted MRI-like intensities (roughly inverse of T1)
    t2_config = SamplingConfig(
        label_intensities={
            0: (5, 20),       # Background (dark)
            1: (120, 160),    # Brain tissue (medium-bright in T2)
            2: (200, 240),    # Tumor (very bright in T2)
            3: (180, 220),    # Blood vessel (bright in T2)
            4: (190, 230)     # Lesions (very bright in T2)
        },
        label_distributions={
            0: "uniform",
            1: "normal",
            2: "normal",
            3: "normal",
            4: "normal"
        },
        label_std={
            1: 8.0,
            2: 7.0,
            3: 5.0,
            4: 6.0
        }
    )
    
    return {"t1": t1_config, "t2": t2_config}

# Step 3: Generate synthetic images from segmentation
def generate_synthetic_images(segmentation, configs):
    """Generate multiple synthetic images from the same segmentation."""
    print("Generating synthetic images from segmentation...")
    
    results = {}
    
    # Generate T1-weighted image
    t1_transform = RandomIntensityFromLabels(
        label_key="label",
        image_key="t1",
        config=configs["t1"],
        clamp_output_min=0.0,
        clamp_output_max=255.0
    )
    
    # Generate T2-weighted image
    t2_transform = RandomIntensityFromLabels(
        label_key="label",
        image_key="t2",
        config=configs["t2"],
        clamp_output_min=0.0,
        clamp_output_max=255.0
    )
    
    # Apply transforms
    data = {"label": segmentation}
    data = t1_transform(data)
    data = t2_transform(data)
    
    return data

# Step 4: Add noise and artifacts to make images more realistic
def add_noise_and_artifacts(data):
    """Add realistic noise and artifacts to synthetic images."""
    print("Adding noise and artifacts to synthetic images...")
    
    # Add Gaussian noise to T1
    t1_image = data["t1"]
    if isinstance(t1_image, torch.Tensor):
        t1_image = t1_image.cpu().numpy()
    
    t1_noise = np.random.normal(0, 5.0, t1_image.shape)
    t1_image_noisy = t1_image + t1_noise
    t1_image_noisy = np.clip(t1_image_noisy, 0, 255)
    
    # Add Gaussian noise to T2
    t2_image = data["t2"]
    if isinstance(t2_image, torch.Tensor):
        t2_image = t2_image.cpu().numpy()
    
    t2_noise = np.random.normal(0, 7.0, t2_image.shape)
    t2_image_noisy = t2_image + t2_noise
    t2_image_noisy = np.clip(t2_image_noisy, 0, 255)
    
    # Add bias field to T1 (simulating MRI inhomogeneity)
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, t1_image.shape[0]),
        np.linspace(-1, 1, t1_image.shape[1]),
        np.linspace(-1, 1, t1_image.shape[2]),
        indexing='ij'
    )
    
    # Create a smooth bias field
    bias_field = 1.0 + 0.1 * (x**2 + y**2 - 0.2 * z)
    
    # Apply bias field to T1
    t1_image_biased = t1_image_noisy * bias_field
    t1_image_biased = np.clip(t1_image_biased, 0, 255)
    
    # Update data dictionary
    data["t1_noisy"] = t1_image_noisy
    data["t2_noisy"] = t2_image_noisy
    data["t1_biased"] = t1_image_biased
    
    return data

# Step 5: Visualize results
def visualize_results(data, save_dir="/workspace/work16/pipeline_figures"):
    """Visualize the segmentation and generated images."""
    print("Visualizing results...")
    
    # Get the data
    segmentation = data["label"]
    t1_image = data["t1"]
    t2_image = data["t2"]
    t1_noisy = data["t1_noisy"]
    t2_noisy = data["t2_noisy"]
    t1_biased = data["t1_biased"]
    
    # Create a directory to save figures
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize multiple slices
    slices = [
        segmentation.shape[0] // 4,
        segmentation.shape[0] // 2,
        3 * segmentation.shape[0] // 4
    ]
    
    for slice_idx in slices:
        # Create a figure with all modalities
        plt.figure(figsize=(20, 10))
        
        # Plot segmentation
        plt.subplot(2, 3, 1)
        plt.title(f"Segmentation (Slice {slice_idx})")
        plt.imshow(segmentation[slice_idx], cmap='viridis')
        plt.colorbar(label='Label')
        
        # Plot T1
        plt.subplot(2, 3, 2)
        plt.title(f"T1-weighted (Slice {slice_idx})")
        plt.imshow(t1_image[slice_idx], cmap='gray')
        plt.colorbar(label='Intensity')
        
        # Plot T2
        plt.subplot(2, 3, 3)
        plt.title(f"T2-weighted (Slice {slice_idx})")
        plt.imshow(t2_image[slice_idx], cmap='gray')
        plt.colorbar(label='Intensity')
        
        # Plot T1 with noise
        plt.subplot(2, 3, 4)
        plt.title(f"T1 with noise (Slice {slice_idx})")
        plt.imshow(t1_noisy[slice_idx], cmap='gray')
        plt.colorbar(label='Intensity')
        
        # Plot T2 with noise
        plt.subplot(2, 3, 5)
        plt.title(f"T2 with noise (Slice {slice_idx})")
        plt.imshow(t2_noisy[slice_idx], cmap='gray')
        plt.colorbar(label='Intensity')
        
        # Plot T1 with bias field
        plt.subplot(2, 3, 6)
        plt.title(f"T1 with bias field (Slice {slice_idx})")
        plt.imshow(t1_biased[slice_idx], cmap='gray')
        plt.colorbar(label='Intensity')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/full_pipeline_slice_{slice_idx}.png", dpi=200)
        plt.close()
    
    # Create a multi-slice visualization for T1 and segmentation overlay
    plt.figure(figsize=(15, 10))
    
    for i, slice_idx in enumerate(slices):
        # Plot segmentation
        plt.subplot(2, 3, i+1)
        plt.title(f"Segmentation (Slice {slice_idx})")
        plt.imshow(segmentation[slice_idx], cmap='viridis')
        plt.colorbar(label='Label')
        
        # Plot T1 with segmentation overlay
        plt.subplot(2, 3, i+4)
        plt.title(f"T1 with segmentation overlay (Slice {slice_idx})")
        
        # Create a masked array for the segmentation
        masked_seg = np.ma.masked_where(segmentation[slice_idx] == 0, segmentation[slice_idx])
        
        # Plot T1 image
        plt.imshow(t1_image[slice_idx], cmap='gray')
        
        # Overlay segmentation with alpha
        plt.imshow(masked_seg, cmap='viridis', alpha=0.5)
        
        plt.colorbar(label='Intensity')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/segmentation_overlay.png", dpi=200)
    plt.close()
    
    # Create a 3D visualization (multiple slices in a grid)
    num_slices = 9
    slice_indices = np.linspace(10, segmentation.shape[0]-10, num_slices, dtype=int)
    
    plt.figure(figsize=(20, 20))
    
    for i, slice_idx in enumerate(slice_indices):
        plt.subplot(3, 3, i+1)
        plt.title(f"T1 (Slice {slice_idx})")
        plt.imshow(t1_image[slice_idx], cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/t1_3d_visualization.png", dpi=200)
    plt.close()
    
    # Create a comparison of all modalities for a single slice
    middle_slice = segmentation.shape[0] // 2
    
    plt.figure(figsize=(20, 10))
    
    # Plot all modalities for the middle slice
    images = [
        (segmentation[middle_slice], "Segmentation", "viridis"),
        (t1_image[middle_slice], "T1-weighted", "gray"),
        (t2_image[middle_slice], "T2-weighted", "gray"),
        (t1_noisy[middle_slice], "T1 with noise", "gray"),
        (t2_noisy[middle_slice], "T2 with noise", "gray"),
        (t1_biased[middle_slice], "T1 with bias field", "gray")
    ]
    
    for i, (img, title, cmap) in enumerate(images):
        plt.subplot(2, 3, i+1)
        plt.title(title)
        plt.imshow(img, cmap=cmap)
        plt.colorbar(label='Intensity' if i > 0 else 'Label')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/modality_comparison.png", dpi=200)
    plt.close()
    
    print(f"All visualizations saved to {save_dir}")

# Step 6: Save the generated data as NIfTI files
def save_as_nifti(data, save_dir="/workspace/work16/pipeline_figures"):
    """Save the generated data as NIfTI files."""
    print("Saving data as NIfTI files...")
    
    # Create a directory to save NIfTI files
    os.makedirs(save_dir, exist_ok=True)
    
    # Create an affine matrix (identity for simplicity)
    affine = np.eye(4)
    
    # Save each modality as a NIfTI file
    for key, image in data.items():
        # Convert to numpy if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(image, affine)
        
        # Save the image
        nib.save(nifti_img, f"{save_dir}/{key}.nii.gz")
    
    print(f"All NIfTI files saved to {save_dir}")

# Run the full pipeline
def run_full_pipeline():
    """Run the full pipeline from segmentation to visualization."""
    print("Running full pipeline...")
    
    # Step 1: Create a synthetic segmentation
    segmentation = create_synthetic_segmentation()
    
    # Step 2: Create intensity configurations
    configs = create_intensity_config()
    
    # Step 3: Generate synthetic images
    data = generate_synthetic_images(segmentation, configs)
    
    # Step 4: Add noise and artifacts
    data = add_noise_and_artifacts(data)
    
    # Step 5: Visualize results
    visualize_results(data)
    
    # Step 6: Save as NIfTI
    save_as_nifti(data)
    
    print("Full pipeline completed successfully!")
    return data

# Execute the full pipeline
if __name__ == "__main__":
    data = run_full_pipeline()