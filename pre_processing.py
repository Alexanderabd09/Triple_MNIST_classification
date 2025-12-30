import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def prepare_dataset(path, IMG_SIZE, BATCH_SIZE, split=False):
    """
    Prepares a TensorFlow dataset from a directory of images.

    Args:
        path: Path to directory containing class subdirectories
        IMG_SIZE: Tuple of (height, width) for resizing images
        BATCH_SIZE: Number of samples per batch
        split: If True, format data for split/Siamese model with separate digit labels

    Returns:
        TensorFlow dataset ready for training/evaluation
    """
    # Get sorted list of existing folders to maintain consistent class ordering
    existing_folders = sorted([
        f for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f))
    ])

    # Load dataset from directory
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=existing_folders,
        shuffle=True if "train" in path else False
    )

    # Map inferred indices to actual folder numbers (e.g., folder '005' -> integer 5)
    folder_to_actual_int = np.array([int(f) for f in existing_folders])

    def transform_data(x, y):
        """Normalize images and map labels to actual class numbers."""
        x = tf.cast(x, tf.float32) / 255.0
        y_actual = tf.gather(folder_to_actual_int, y)
        return x, y_actual

    ds = ds.map(transform_data, num_parallel_calls=tf.data.AUTOTUNE)

    # If split mode, format for multi-output model
    if split:
        def split_transform(image, label_idx):
            """Transform for split model."""
            label_idx = tf.cast(label_idx, tf.int32)

            # Split 3-digit number into individual digits
            digit1 = label_idx // 100
            digit2 = (label_idx % 100) // 10
            digit3 = label_idx % 10

            return (
                {"img_in": image},
                {"out_1": digit1, "out_2": digit2, "out_3": digit3}
            )

        ds = ds.map(split_transform, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def split_dataset(dataset):
    """
    Converts a TensorFlow dataset into NumPy arrays.

    Args:
        dataset: TensorFlow dataset

    Returns:
        Tuple of (X, y) as NumPy arrays
    """
    X_list, y_list = [], []
    for images, labels in dataset:
        X_list.append(images.numpy())
        y_list.append(labels.numpy())
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def show_samples(dataset_path, num_samples=16, img_size=(84, 84), save_path='dataset_samples.png'):
    """
    Display and save random samples from the Triple-MNIST dataset.

    Args:
        dataset_path: Path to dataset directory (e.g., "triple_mnist_cleaned/train/")
        num_samples: Number of samples to display (default: 16, must be perfect square)
        img_size: Image size tuple (height, width)
        save_path: Path to save the visualization

    Returns:
        None (displays and saves visualization)
    """
    print(f"\n{'=' * 70}")
    print(f"LOADING SAMPLES FROM: {dataset_path}")
    print(f"{'=' * 70}\n")

    # Load dataset
    ds = prepare_dataset(dataset_path, img_size, batch_size=num_samples, split=False)

    # Get one batch of samples
    for images, labels in ds.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        break

    # Calculate grid size
    grid_size = int(np.sqrt(num_samples))

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'Random Samples from Triple-MNIST Dataset\nPath: {dataset_path}',
                 fontsize=16, weight='bold', y=0.995)

    # Display images
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            # Display image
            ax.imshow(images_np[idx, :, :, 0], cmap='gray')

            # Format label (3-digit number)
            label = int(labels_np[idx])
            digit1 = label // 100
            digit2 = (label % 100) // 10
            digit3 = label % 10

            # Add title with label
            ax.set_title(f'Label: {label:03d} ({digit1}-{digit2}-{digit3})',
                         fontsize=10, pad=5)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved as '{save_path}'")
    plt.show()

    # Print statistics
    print(f"\n{'=' * 70}")
    print("DATASET STATISTICS")
    print(f"{'=' * 70}")
    print(f"Number of samples shown: {num_samples}")
    print(f"Image shape: {images_np[0].shape}")
    print(f"Image data type: {images_np[0].dtype}")
    print(f"Pixel value range: [{images_np.min():.3f}, {images_np.max():.3f}]")
    print(f"\nLabels shown: {', '.join([f'{int(l):03d}' for l in labels_np])}")

    # Analyze label distribution
    unique_labels = np.unique(labels_np)
    print(f"\nUnique labels in batch: {len(unique_labels)}")
    print(f"Label range: {int(labels_np.min()):03d} - {int(labels_np.max()):03d}")
    print(f"{'=' * 70}\n")


def analyze_dataset_structure(base_path="triple_mnist"):
    """
    Analyze and report on the dataset structure.

    Args:
        base_path: Base path to the dataset

    Returns:
        Dictionary with dataset statistics
    """
    print(f"\n{'=' * 70}")
    print("DATASET STRUCTURE ANALYSIS")
    print(f"{'=' * 70}\n")

    splits = ['train', 'val', 'test']
    stats = {}

    for split in splits:
        split_path = os.path.join(base_path, split)

        if not os.path.exists(split_path):
            print(f"❌ {split_path} does not exist!")
            continue

        # Get all class folders
        class_folders = sorted([
            f for f in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, f))
        ])

        # Count images per class
        total_images = 0
        for folder in class_folders:
            folder_path = os.path.join(split_path, folder)
            images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
            total_images += len(images)

        stats[split] = {
            'num_classes': len(class_folders),
            'total_images': total_images,
            'avg_per_class': total_images / len(class_folders) if class_folders else 0
        }

        print(f"{split.upper()} SET:")
        print(f"  Classes: {stats[split]['num_classes']}")
        print(f"  Total images: {stats[split]['total_images']}")
        print(f"  Avg images/class: {stats[split]['avg_per_class']:.1f}\n")

    print(f"{'=' * 70}\n")
    return stats