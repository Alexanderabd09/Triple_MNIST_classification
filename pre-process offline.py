import cv2
import os
from tqdm import tqdm


def offline_clean_dataset(src_root, dest_root):
    """
    Applies denoising and adaptive thresholding to all images in a directory tree.
    Preserves the original directory structure in the destination.

    Args:
        src_root: Source directory containing raw images
        dest_root: Destination directory for cleaned images
    """
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    for subdir, dirs, files in os.walk(src_root):
        # Create corresponding subdirectory in destination
        relative_path = os.path.relpath(subdir, src_root)
        dest_dir = os.path.join(dest_root, relative_path)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Process image files
        image_files = [f for f in files if f.endswith(".png")]
        if image_files:
            print(f"\nProcessing folder: {relative_path}")
            for file in tqdm(image_files, desc="Images"):
                img_path = os.path.join(subdir, file)
                save_path = os.path.join(dest_dir, file)

                try:
                    # Read grayscale image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        print(f"  Warning: Could not read {img_path}")
                        continue

                    # Apply non-local means denoising
                    denoised = cv2.fastNlMeansDenoising(img, h=10)

                    # Apply adaptive thresholding
                    thresh = cv2.adaptiveThreshold(
                        denoised,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        11,
                        2
                    )

                    # Save cleaned image
                    cv2.imwrite(save_path, thresh)

                except Exception as e:
                    print(f"  Error processing {file}: {e}")


def main():
    """
    Main function to clean all dataset splits.
    """
    print("=" * 60)
    print("Offline Dataset Cleaning")
    print("=" * 60)

    splits = [
        ("triple_mnist/train", "triple_mnist/train"),
        ("triple_mnist/val", "triple_mnist/val"),
        ("triple_mnist/test", "triple_mnist/test")
    ]

    for src, dest in splits:
        if os.path.exists(src):
            print(f"\nCleaning {src} -> {dest}")
            offline_clean_dataset(src, dest)
        else:
            print(f"\nWarning: Source directory not found: {src}")

    print("\n" + "=" * 60)
    print("Dataset cleaning completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()