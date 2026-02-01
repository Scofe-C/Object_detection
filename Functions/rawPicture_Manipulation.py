#!/usr/bin/env python3
"""
Image Augmentation Script (No Annotations)
Generates 100 transformed images from input photos with various transformations
All output images will be 224x224 pixels with no white padding
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()


def load_images(input_folder):
    """Load all images from the input folder"""
    supported_formats = ('.jpg', '.jpeg', '.png', '.heic', '.HEIC')
    image_files = []

    for file in Path(input_folder).glob('*'):
        if file.suffix.lower() in supported_formats:
            image_files.append(file)

    return image_files


def apply_transformations(img, transform_type, params):
    """Apply various transformations to the image"""

    if transform_type == 'rotate':
        angle = params.get('angle', random.randint(-30, 30))
        img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    elif transform_type == 'flip_horizontal':
        img = ImageOps.mirror(img)

    elif transform_type == 'flip_vertical':
        img = ImageOps.flip(img)

    elif transform_type == 'scale':
        scale_factor = params.get('scale', random.uniform(0.8, 1.2))
        w, h = img.size
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    elif transform_type == 'brightness':
        factor = params.get('factor', random.uniform(0.7, 1.3))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)

    elif transform_type == 'contrast':
        factor = params.get('factor', random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)

    elif transform_type == 'crop_zoom':
        w, h = img.size
        crop_percent = random.uniform(0.7, 0.9)
        crop_w, crop_h = int(w * crop_percent), int(h * crop_percent)
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        img = img.crop((left, top, left + crop_w, top + crop_h))

    return img


def resize_to_target(img, target_size=(224, 224)):
    """Resize image to target size by center cropping to fill the entire frame
    No white padding - fills entire 224x224"""
    orig_width, orig_height = img.size
    target_width, target_height = target_size

    # Calculate scale to cover the entire target size
    scale_w = target_width / orig_width
    scale_h = target_height / orig_height
    scale = max(scale_w, scale_h)  # Use max to ensure image covers entire target

    # Resize image
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center crop to target size
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    img_cropped = img_resized.crop((left, top, right, bottom))

    return img_cropped


def generate_augmented_dataset(input_folder, output_folder, target_count=100, target_size=(224, 224)):
    """Generate augmented dataset with various transformations"""

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load input images
    image_files = load_images(input_folder)

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} input images")
    print(f"Generating {target_count} augmented images...")

    # Define transformation types
    transformations = [
        'rotate',
        'flip_horizontal',
        'flip_vertical',
        'scale',
        'brightness',
        'contrast',
        'crop_zoom'
    ]

    generated_count = 0

    # First, save original images resized to target size
    for idx, img_path in enumerate(image_files):
        if generated_count >= target_count:
            break

        try:
            img = Image.open(img_path).convert('RGB')
            img_resized = resize_to_target(img, target_size)
            output_path = os.path.join(output_folder, f'OBJ028_{generated_count:03d}.jpg')
            img_resized.save(output_path, 'JPEG', quality=95)
            generated_count += 1
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Generate augmented images until we reach target_count
    while generated_count < target_count:
        # Randomly select an input image
        img_path = random.choice(image_files)

        try:
            img = Image.open(img_path).convert('RGB')

            # Apply 1-3 random transformations
            num_transforms = random.randint(1, 3)
            transform_names = random.sample(transformations, num_transforms)

            for transform in transform_names:
                img = apply_transformations(img, transform, {})

            # Resize to target size
            img_final = resize_to_target(img, target_size)

            # Save the augmented image
            output_path = os.path.join(output_folder, f'OBJ028_{generated_count:03d}.jpg')
            img_final.save(output_path, 'JPEG', quality=95)

            generated_count += 1

            if generated_count % 10 == 0:
                print(f"Progress: {generated_count}/{target_count} images generated")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nCompleted! Generated {generated_count} images in {output_folder}")
    print(f"All images are {target_size[0]}x{target_size[1]} pixels")


def main():
    print("=" * 60)
    print("Image Augmentation Start")
    print("=" * 60)

    # Configuration
    input_folder = r"D:\NEU\IE7615\data\raw1"
    output_folder = r"D:\NEU\IE7615\data\raw_1_train"
    target_count = 100  # Number of images to generate
    target_size = (224, 224)  # Output image size

    # Check if input folder has images
    if not os.path.exists(input_folder):
        print(f"\nError: Input folder '{input_folder}' does not exist!")
        print("Please upload your photos first.")
        return

    image_files = load_images(input_folder)
    if not image_files:
        print(f"\nNo images found in '{input_folder}'")
        print("Please upload your photos to this location.")
        print("\nSupported formats: .jpg, .jpeg, .png, .heic")
        return

    print(f"\nConfiguration:")
    print(f"  Input folder: {input_folder}")
    print(f"  Output folder: {output_folder}")
    print(f"  Target count: {target_count}")
    print(f"  Target size: {target_size}")
    print(f"  Found {len(image_files)} input images")
    print()

    generate_augmented_dataset(input_folder, output_folder, target_count, target_size)


if __name__ == "__main__":
    main()