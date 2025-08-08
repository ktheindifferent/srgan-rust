#!/usr/bin/env python3
"""
Create Training Dataset for SRGAN-Rust

This script helps create training datasets from your own images.
It can:
- Crop images into patches
- Create low-resolution pairs
- Apply data augmentation
- Split into train/validation sets
"""

import os
import sys
import argparse
import random
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional

def create_patches(image: Image.Image, patch_size: int, stride: Optional[int] = None) -> List[Image.Image]:
    """Extract patches from an image."""
    if stride is None:
        stride = patch_size
    
    patches = []
    width, height = image.size
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
    
    return patches

def downscale_image(image: Image.Image, scale: int, method: str = 'bicubic') -> Image.Image:
    """Downscale an image by a factor."""
    width, height = image.size
    new_size = (width // scale, height // scale)
    
    resample_methods = {
        'bicubic': Image.BICUBIC,
        'bilinear': Image.BILINEAR,
        'lanczos': Image.LANCZOS,
        'nearest': Image.NEAREST
    }
    
    method_enum = resample_methods.get(method, Image.BICUBIC)
    return image.resize(new_size, method_enum)

def augment_image(image: Image.Image, augmentations: List[str]) -> List[Image.Image]:
    """Apply data augmentations to an image."""
    augmented = [image]
    
    if 'flip_h' in augmentations:
        augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    
    if 'flip_v' in augmentations:
        augmented.append(image.transpose(Image.FLIP_TOP_BOTTOM))
    
    if 'rotate_90' in augmentations:
        augmented.append(image.rotate(90, expand=True))
        augmented.append(image.rotate(180, expand=True))
        augmented.append(image.rotate(270, expand=True))
    
    return augmented

def process_dataset(
    input_dir: Path,
    output_dir: Path,
    patch_size: int,
    scale: int,
    stride: Optional[int] = None,
    augmentations: List[str] = [],
    train_split: float = 0.8,
    min_patch_std: float = 0.0
):
    """Process a directory of images into a training dataset."""
    
    # Create output directories
    train_hr_dir = output_dir / 'train' / 'HR'
    train_lr_dir = output_dir / 'train' / f'LR_x{scale}'
    val_hr_dir = output_dir / 'validation' / 'HR'
    val_lr_dir = output_dir / 'validation' / f'LR_x{scale}'
    
    for dir_path in [train_hr_dir, train_lr_dir, val_hr_dir, val_lr_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle for train/val split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train: {len(train_files)} images, Validation: {len(val_files)} images")
    
    patch_count = 0
    rejected_count = 0
    
    for idx, (files, is_train) in enumerate([(train_files, True), (val_files, False)]):
        hr_dir = train_hr_dir if is_train else val_hr_dir
        lr_dir = train_lr_dir if is_train else val_lr_dir
        dataset_name = "training" if is_train else "validation"
        
        print(f"\nProcessing {dataset_name} set...")
        
        for img_path in files:
            try:
                image = Image.open(img_path).convert('RGB')
                
                # Skip small images
                if image.width < patch_size or image.height < patch_size:
                    print(f"  Skipping {img_path.name} (too small)")
                    continue
                
                # Extract patches
                patches = create_patches(image, patch_size, stride)
                
                # Apply augmentations
                if is_train and augmentations:
                    augmented_patches = []
                    for patch in patches:
                        augmented_patches.extend(augment_image(patch, augmentations))
                    patches = augmented_patches
                
                # Save patches
                for patch_idx, patch in enumerate(patches):
                    # Check patch variance (skip uniform patches)
                    if min_patch_std > 0:
                        patch_array = np.array(patch)
                        if np.std(patch_array) < min_patch_std:
                            rejected_count += 1
                            continue
                    
                    # Create filenames
                    base_name = f"{img_path.stem}_{patch_idx:04d}.png"
                    hr_path = hr_dir / base_name
                    lr_path = lr_dir / base_name
                    
                    # Save HR patch
                    patch.save(hr_path, 'PNG')
                    
                    # Create and save LR patch
                    lr_patch = downscale_image(patch, scale)
                    lr_patch.save(lr_path, 'PNG')
                    
                    patch_count += 1
                
                print(f"  Processed {img_path.name}: {len(patches)} patches")
                
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
    
    print(f"\nDataset creation complete!")
    print(f"Total patches created: {patch_count}")
    if rejected_count > 0:
        print(f"Patches rejected (low variance): {rejected_count}")
    
    # Create info file
    info_file = output_dir / 'dataset_info.txt'
    with open(info_file, 'w') as f:
        f.write(f"Dataset Information\n")
        f.write(f"==================\n\n")
        f.write(f"Source directory: {input_dir}\n")
        f.write(f"Patch size: {patch_size}x{patch_size}\n")
        f.write(f"Downscale factor: {scale}x\n")
        f.write(f"Stride: {stride if stride else patch_size}\n")
        f.write(f"Augmentations: {', '.join(augmentations) if augmentations else 'None'}\n")
        f.write(f"Train/Val split: {train_split:.0%}/{(1-train_split):.0%}\n")
        f.write(f"Total patches: {patch_count}\n")
        f.write(f"Training patches: {len(list(train_hr_dir.iterdir()))}\n")
        f.write(f"Validation patches: {len(list(val_hr_dir.iterdir()))}\n")
    
    print(f"\nDataset info saved to: {info_file}")

def create_test_images(output_dir: Path, count: int = 10):
    """Create synthetic test images for testing the pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {count} synthetic test images...")
    
    for i in range(count):
        # Create random image with patterns
        size = random.randint(256, 512)
        image = Image.new('RGB', (size, size))
        pixels = image.load()
        
        # Add some patterns
        for y in range(size):
            for x in range(size):
                # Gradient with noise
                r = int((x / size) * 255 + random.randint(-20, 20))
                g = int((y / size) * 255 + random.randint(-20, 20))
                b = int(((x + y) / (2 * size)) * 255 + random.randint(-20, 20))
                
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                pixels[x, y] = (r, g, b)
        
        # Add some geometric shapes
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        for _ in range(random.randint(3, 8)):
            shape_type = random.choice(['rectangle', 'ellipse'])
            color = tuple(random.randint(0, 255) for _ in range(3))
            x1, y1 = random.randint(0, size//2), random.randint(0, size//2)
            x2, y2 = random.randint(size//2, size), random.randint(size//2, size)
            
            if shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=color)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=color)
        
        image.save(output_dir / f"test_image_{i:03d}.png")
    
    print(f"Test images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create training dataset for SRGAN-Rust')
    parser.add_argument('input_dir', type=Path, help='Input directory containing images')
    parser.add_argument('output_dir', type=Path, help='Output directory for dataset')
    parser.add_argument('--patch-size', type=int, default=96, help='Size of patches to extract')
    parser.add_argument('--scale', type=int, default=4, help='Downscaling factor (2 or 4)')
    parser.add_argument('--stride', type=int, help='Stride for patch extraction (default: patch_size)')
    parser.add_argument('--augment', nargs='+', choices=['flip_h', 'flip_v', 'rotate_90'],
                       help='Data augmentation methods')
    parser.add_argument('--train-split', type=float, default=0.8, 
                       help='Proportion of data for training (0-1)')
    parser.add_argument('--min-std', type=float, default=10.0,
                       help='Minimum standard deviation for patches (filters uniform patches)')
    parser.add_argument('--create-test', action='store_true',
                       help='Create synthetic test images in input directory')
    
    args = parser.parse_args()
    
    if args.create_test:
        create_test_images(args.input_dir)
    
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    print(f"Creating dataset from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    process_dataset(
        args.input_dir,
        args.output_dir,
        args.patch_size,
        args.scale,
        args.stride,
        args.augment or [],
        args.train_split,
        args.min_std
    )

if __name__ == '__main__':
    main()