
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

def prepare_dataset(path, IMG_SIZE, BATCH_SIZE):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        label_mode='int',
        color_mode='grayscale',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True if "train" in path else False
    )
    return ds.map(split_image_data).prefetch(tf.data.AUTOTUNE)

def pre_processing(img, labels):
    img = tf.cast(img, tf.float32) / 255.0
    return img, labels


def show_samples(dataset, class_names):
    # Fixed: Moved plt.figure inside the loop or ensured it only runs once
    for images, labels in dataset.take(1):
        plt.figure(figsize=(12, 8))
        for i in range(min(len(images), 12)):  # Added safety check for batch size
            plt.subplot(3, 4, i + 1)
            img = images[i]
            if img.shape[-1] == 1:
                plt.imshow(img[..., 0], cmap='gray')
            else:
                plt.imshow(img)

            label_idx = labels[i].numpy()
            # Ensure label_idx is treated correctly if it's a multi-digit integer
            plt.title(str(label_idx), fontsize=8)
            plt.axis("off")
    plt.show()


def split_dataset(dataset):
    X_batches = []
    Y_batches = []
    for images, labels in dataset:
        X_batches.append(images.numpy())
        Y_batches.append(labels.numpy())
    X = np.concatenate(X_batches, axis=0)
    y = np.concatenate(Y_batches, axis=0)
    return X, y


def split_image_data(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    pieces = tf.split(img, num_or_size_splits=3, axis=1)
    d1 = label // 100
    d2 = (label % 100) // 10
    d3 = label % 10

    return {"left_in": pieces[0], "mid_in": pieces[1], "right_in": pieces[2]}, \
        {"out_1": d1, "out_2": d2, "out_3": d3}


def build_split_cnn(num_classes=10):
    # Shared Backbone
    shared_backbone = models.Sequential([
        layers.Input(shape=(84, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3)
    ], name="digit_feature_extractor")

    # Inputs
    in_left = layers.Input(shape=(84, 28, 1), name="left_in")
    in_mid = layers.Input(shape=(84, 28, 1), name="mid_in")
    in_right = layers.Input(shape=(84, 28, 1), name="right_in")

    # Pass through shared weights
    feat_left = shared_backbone(in_left)
    feat_mid = shared_backbone(in_mid)
    feat_right = shared_backbone(in_right)

    # Outputs
    out1 = layers.Dense(num_classes, activation='softmax', name="out_1")(feat_left)
    out2 = layers.Dense(num_classes, activation='softmax', name="out_2")(feat_mid)
    out3 = layers.Dense(num_classes, activation='softmax', name="out_3")(feat_right)

    model = models.Model(inputs=[in_left, in_mid, in_right], outputs=[out1, out2, out3])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model