import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


def prepare_dataset(path, IMG_SIZE, BATCH_SIZE, split=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        label_mode='int',
        color_mode='grayscale',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True if "train" in path else False
    )
    # Only map the split function if requested
    if split:
        return ds.map(split_image_data).prefetch(tf.data.AUTOTUNE)
    # Otherwise just normalize for standard models
    return ds.map(pre_processing).prefetch(tf.data.AUTOTUNE)


def pre_processing(img, labels):
    img = tf.cast(img, tf.float32) / 255.0
    return img, labels


def show_samples(dataset, class_names=None):
    for images, labels in dataset.take(1):
        # If images is a dict (from split_image_data), extract one part to show
        if isinstance(images, dict):
            display_images = images["left_in"]
            title_prefix = "Left Strip"
        else:
            display_images = images
            title_prefix = "Full Image"

        plt.figure(figsize=(12, 8))
        # Use the batch size of the actual tensor, not the dict length
        num_to_show = min(len(display_images), 12)

        for i in range(num_to_show):
            plt.subplot(3, 4, i + 1)
            img = display_images[i]

            if img.shape[-1] == 1:
                plt.imshow(img[..., 0], cmap='gray')
            else:
                plt.imshow(img)

            # Handle dict labels or integer labels
            if isinstance(labels, dict):
                label_val = f"{labels['out_1'][i]}-{labels['out_2'][i]}-{labels['out_3'][i]}"
            else:
                label_val = class_names[labels[i]] if class_names else labels[i].numpy()

            plt.title(f"{title_prefix}\nLabel: {label_val}", fontsize=8)
            plt.axis("off")
    plt.show()


def split_dataset(dataset):
    X_batches = []
    Y_batches = []
    for images, labels in dataset:
        # split_dataset only works easily with non-dict data
        if isinstance(images, dict):
            raise ValueError("split_dataset does not support dictionary inputs. Use split=False in prepare_dataset.")
        X_batches.append(images.numpy())
        Y_batches.append(labels.numpy())
    X = np.concatenate(X_batches, axis=0)
    y = np.concatenate(Y_batches, axis=0)
    return X, y


def split_image_data(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    # Axis 1 is width for (Height, Width, Channels)
    pieces = tf.split(img, num_or_size_splits=3, axis=1)
    d1 = label // 100
    d2 = (label % 100) // 10
    d3 = label % 10
    return {"left_in": pieces[0], "mid_in": pieces[1], "right_in": pieces[2]}, \
        {"out_1": d1, "out_2": d2, "out_3": d3}