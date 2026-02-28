import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from PIL import Image, ImageEnhance

def enhance_image(image_path, output_path, brightness=1.2, contrast=1.2, noise_str=3):
    """
    Processes a single image: Denoising, LAB-based Brightness/Contrast, 
    and Color Balancing.
    """
    # Load image using OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    # 1. NOISE REDUCTION
    # fastNlMeansDenoisingColored preserves edges better than Gaussian blur
    # h = strength for luminance, hColor = strength for color components
    denoised = cv2.fastNlMeansDenoisingColored(img, None, noise_str, noise_str, 7, 21)

    # 2. BRIGHTNESS & CONTRAST (LAB Space)
    # We convert to LAB to adjust 'L' (Lightness) without shifting colors (A/B)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for natural look
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge back and convert to BGR
    enhanced_lab = cv2.merge((l, a, b))
    img_final = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 3. FINE-TUNING WITH PILLOW (Brightness/Contrast/Color)
    # Convert BGR (OpenCV) to RGB (Pillow)
    pil_img = Image.fromarray(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
    
    # Adjust Brightness
    enhancer_b = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer_b.enhance(brightness)
    
    # Adjust Contrast
    enhancer_c = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer_c.enhance(contrast)

    # Subtle Saturation boost for natural skies/tones
    enhancer_s = ImageEnhance.Color(pil_img)
    pil_img = enhancer_s.enhance(1.1) 

    # Save output
    pil_img.save(output_path, quality=95, subsampling=0)

def main():
    parser = argparse.ArgumentParser(description="Automated JPEG Enhancement Pipeline")
    parser.add_argument("--input", required=True, help="Input file or folder path")
    parser.add_argument("--output", required=True, help="Output folder path")
    parser.add_argument("--brightness", type=float, default=1.1, help="Brightness factor (default 1.1)")
    parser.add_argument("--contrast", type=float, default=1.1, help="Contrast factor (default 1.1)")
    parser.add_argument("--noise", type=int, default=3, help="Denoise strength 1-10 (default 3)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather files
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))

    print(f"Processing {len(files)} images...")
    for f in files:
        out_file = output_dir / f.name
        enhance_image(f, out_file, args.brightness, args.contrast, args.noise)
        print(f"Enhanced: {f.name}")

if __name__ == "__main__":
    main()