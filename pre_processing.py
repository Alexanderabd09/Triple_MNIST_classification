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

    # Apply the appropriate transformation
    if split:
        ds = ds.map(split_image_data, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(pre_processing, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


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
    # Normalize the image pixels to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0

    # Image input shape is (84, 84, 1)
    # Split across the width (axis=1) into 3 vertical strips of (84, 28, 1)
    pieces = tf.split(img, num_or_size_splits=3, axis=1)

    # Ensure the tensors are explicitly shaped for the model input layers
    left = tf.reshape(pieces[0], (84, 28, 1))
    mid = tf.reshape(pieces[1], (84, 28, 1))
    right = tf.reshape(pieces[2], (84, 28, 1))

    # --- Label Splitting Logic ---
    # Convert integer label (e.g., 123) to a zero-padded string "123"
    # we use {:03d} to ensure labels like '5' become '005'
    label_str = tf.strings.format("{:03d}", [label])

    # Get the single string from the batch tensor
    label_str_scalar = label_str[0]

    # Slice the string to get individual digit characters
    d1_str = tf.strings.substr(label_str_scalar, 0, 1)  # Hundreds place
    d2_str = tf.strings.substr(label_str_scalar, 1, 1)  # Tens place
    d3_str = tf.strings.substr(label_str_scalar, 2, 1)  # Units place

    # Convert characters back to integers (0-9)
    d1 = tf.strings.to_number(d1_str, out_type=tf.int32)
    d2 = tf.strings.to_number(d2_str, out_type=tf.int32)
    d3 = tf.strings.to_number(d3_str, out_type=tf.int32)

    # Return a tuple: (inputs dictionary, outputs dictionary)
    # Keys must match the 'name' attributes of the layers in your Keras model
    return (
        {"left_in": left, "mid_in": mid, "right_in": right},
        {"out_1": d1, "out_2": d2, "out_3": d3}
    )


def build_split_cnn(num_classes=10):
    # Define the 3 separate inputs (Strips are 84 high x 28 wide)
    input_left = layers.Input(shape=(84, 28, 1), name="left_in")
    input_mid = layers.Input(shape=(84, 28, 1), name="mid_in")
    input_right = layers.Input(shape=(84, 28, 1), name="right_in")

    # Define a shared feature extractor (Siamese approach)
    # This allows the model to learn "what a digit looks like" using all three strips
    shared_conv = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])

    # Pass each input through the same shared layers
    feat_left = shared_conv(input_left)
    feat_mid = shared_conv(input_mid)
    feat_right = shared_conv(input_right)

    # 3 Separate Output Heads (one for each digit position)
    output_left = layers.Dense(num_classes, activation='softmax', name="out_1")(feat_left)
    output_mid = layers.Dense(num_classes, activation='softmax', name="out_2")(feat_mid)
    output_right = layers.Dense(num_classes, activation='softmax', name="out_3")(feat_right)

    # Build the final multi-input, multi-output model
    model = models.Model(
        inputs=[input_left, input_mid, input_right],
        outputs=[output_left, output_mid, output_right]
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model