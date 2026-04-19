import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm   # pip install tqdm  (progress bar)

#directories
INPUT_DIR  = "data/SkinLesionData/images"        # folder with your 300 raw images
OUTPUT_DIR = "data/SkinLesionData/images_no_hair"# processed images saved here

# Supported image extensions
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#parameters 
KERNEL_SIZE    = 17     # BlackHat kernel size (must be ODD)
                        # larger → detects thicker hair
                        # smaller → only catches fine hair
HAIR_THRESHOLD = 10     # pixel intensity threshold for hair mask
                        # lower → more aggressive detection
                        # higher → only catches very dark hair
INPAINT_RADIUS = 6      # inpainting fill radius in pixels
                        # larger → smoother fill, slightly blurry
BLUR_KSIZE     = 3      # optional light Gaussian blur on mask
                        # to smooth jagged hair edges (set 0 to skip)




def remove_hair(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes hair artifacts from a single dermoscopic image.

    Steps:
      1. Convert to grayscale
      2. BlackHat morphological transform → highlights dark thin
         structures (hair) against a bright background
      3. Threshold the result → binary hair mask
      4. (Optional) slight blur on mask to smooth hair edges
      5. Inpaint (Telea) → fill hair regions using surrounding pixels

    Args:
        image : BGR image as a numpy array (H, W, 3)

    Returns:
        result    : hair-removed BGR image
        hair_mask : binary mask of detected hair (for inspection)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── Step 1: BlackHat transform ───────────────────────────
    # BlackHat = closing(image) - image
    # Enhances dark features smaller than the kernel (i.e. hairs)
    kernel   = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (KERNEL_SIZE, KERNEL_SIZE)
               )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # ── Step 2: Threshold 
    _, hair_mask = cv2.threshold(
                        blackhat,
                        HAIR_THRESHOLD,
                        255,
                        cv2.THRESH_BINARY
                   )

    # ── Step 3: (Optional) smooth the mask slightly ──────────
    # Fills tiny gaps in detected hair strands so inpainting
    # covers the full strand width, not just the dark core
    if BLUR_KSIZE > 0:
        hair_mask = cv2.GaussianBlur(
                        hair_mask,
                        (BLUR_KSIZE, BLUR_KSIZE), 0
                    )
        _, hair_mask = cv2.threshold(hair_mask, 10, 255, cv2.THRESH_BINARY)

  #inpainting
    result = cv2.inpaint(
                 image,
                 hair_mask,
                 inpaintRadius=INPAINT_RADIUS,
                 flags=cv2.INPAINT_TELEA
             )

    return result, hair_mask



# BATCH PROCESSOR


def process_dataset(input_dir: str, output_dir: str) -> None:
    """
    Reads all images from input_dir, applies hair removal,
    and saves each result to output_dir with the SAME filename.
    Skips already-processed files so you can resume safely.

    Also saves hair masks to output_dir/masks/ for inspection.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    mask_path   = output_path / "masks"   # optional mask folder

    # Validate input directory
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    # Create output folders
    output_path.mkdir(parents=True, exist_ok=True)
    mask_path.mkdir(parents=True, exist_ok=True)

    # Gather all valid image files
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in VALID_EXTS
    ])

    if not image_files:
        print(f"No images found in {input_path}")
        return

    print(f"\nFound {len(image_files)} images in '{input_path}'")
    print(f"Saving processed images to '{output_path}'\n")

    # Tracking counters 
    success  = 0
    skipped  = 0
    failed   = 0
    errors   = []

    # Process each image 
    for img_file in tqdm(image_files, desc="Removing hair", unit="img"):

        out_file  = output_path / img_file.name        # same filename
        mask_file = mask_path   / img_file.name        # mask with same name

        # Skip if already processed 
        if out_file.exists():
            skipped += 1
            continue

        # Read image
        image = cv2.imread(str(img_file))
        if image is None:
            tqdm.write(f"  [WARN] Could not read: {img_file.name}")
            failed += 1
            errors.append(img_file.name)
            continue

        try:
            # Apply hair removal
            cleaned, hair_mask = remove_hair(image)

            # Save processed image (same format as input)
            cv2.imwrite(str(out_file),  cleaned)

            # Save mask for visual inspection
            cv2.imwrite(str(mask_file), hair_mask)

            success += 1

        except Exception as e:
            tqdm.write(f"  [ERROR] {img_file.name}: {e}")
            failed += 1
            errors.append(img_file.name)

  
    print(f"\n{'─'*45}")
    print(f"  Done!")
    print(f"  Processed : {success}")
    print(f"  Skipped   : {skipped}  (already existed)")
    print(f"  Failed    : {failed}")
    if errors:
        print(f"\n  Failed files:")
        for e in errors:
            print(f"    - {e}")
    print(f"{'─'*45}\n")
    print(f"  Cleaned images → {output_path}/")
    print(f"  Hair masks     → {mask_path}/")
    print(f"  (Check masks/ to verify detection quality)\n")


# SINGLE IMAGE PREVIEW  


def preview_single(image_path: str) -> None:
    """
    Runs hair removal on one image and shows before/after/mask
    side-by-side using OpenCV. Press any key to close.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read: {image_path}")
        return

    cleaned, mask = remove_hair(image)

    # Resize for display if image is large
    h, w = image.shape[:2]
    scale = min(1.0, 600 / max(h, w))
    def rs(img):
        return cv2.resize(img, (int(w*scale), int(h*scale)))

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)   # make 3-channel
    panel    = np.hstack([rs(image), rs(mask_bgr), rs(cleaned)])

    cv2.imshow("Original  |  Hair mask  |  Cleaned", panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":


    process_dataset(INPUT_DIR, OUTPUT_DIR)