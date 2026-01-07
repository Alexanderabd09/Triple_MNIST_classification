import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def prepare_dataset(path, IMG_SIZE, BATCH_SIZE=128, split=False, gan_mode=False):

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
        if gan_mode:
            x = (x -127.5)/127.5
        else:
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

    X_list, y_list = [], []
    for images, labels in dataset:
        X_list.append(images.numpy())
        y_list.append(labels.numpy())
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def show_samples(dataset_path, num_samples=16, img_size=(84, 84), save_path='dataset_samples.png'):


    print(f"Loading Samples from: {dataset_path}")


    # Load dataset
    ds = prepare_dataset(dataset_path, img_size, BATCH_SIZE= num_samples, split=False)

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
    print(f"Visualization saved as '{save_path}'")
    plt.show()

    # Analyze label distribution
    unique_labels = np.unique(labels_np)
    print(f"\nUnique labels in batch: {len(unique_labels)}")
    print(f"Label range: {int(labels_np.min()):03d} - {int(labels_np.max()):03d}")



def analyze_dataset_structure(base_path="triple_mnist"):


    print("DATASET STRUCTURE ANALYSIS")


    splits = ['train', 'val', 'test']
    stats = {}

    for split in splits:
        split_path = os.path.join(base_path, split)

        if not os.path.exists(split_path):
            print(f"{split_path} does not exist!")
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


    return stats