"""
Builder.py - All model architectures and training functions
Contains models and utilities for Tasks 2, 4, and 5

FIXED: The previous version had digit 1 dominating the shared backbone.
This version uses THREE INDEPENDENT CNNs (one per digit position).
"""

import time
import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import os


# ============================================================================
# TASK 2: BASELINE MODELS
# ============================================================================

def create_cnn_model(img_size, num_classes):

    augmentation = tf.keras.Sequential([
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomRotation(0.05),
    ])

    model = models.Sequential([
        layers.Input(shape=(*img_size, 1)),
        augmentation,
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def run_benchmark_lr(X_train, X_test, y_train, y_test, C=1.0):
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    start = time.time()
    lr = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', multi_class='multinomial')
    lr.fit(X_train_flat, y_train)
    train_time = time.time() - start

    preds = lr.predict(X_test_flat)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds, average='macro'),
        "training_time": train_time
    }, lr


def run_benchmark_cnn(model, X_train, X_test, y_train, y_test, X_val, y_val, epochs, batch_size):
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    train_time = time.time() - start

    preds = np.argmax(model.predict(X_test), axis=1)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds, average='macro'),
        "training_time": train_time,
        "history": history.history
    }


# ============================================================================
# TASK 4: SPLIT CNN - REDESIGNED WITH INDEPENDENT CNNS
# ============================================================================

class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.95):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        acc1 = logs.get('val_out_1_accuracy', 0)
        acc2 = logs.get('val_out_2_accuracy', 0)
        acc3 = logs.get('val_out_3_accuracy', 0)

        if acc1 >= self.threshold and acc2 >= self.threshold and acc3 >= self.threshold:
            print(f"\nReached {self.threshold * 100}% accuracy on all heads. Stopping training!")
            self.model.stop_training = True


def build_dual_path_cnn(num_classes=10, learning_rate=0.001):
    """
    Split CNN - REDESIGNED for balanced learning.

    Previous problem: With a shared backbone, digit 1's gradients dominated,
    causing digits 2 and 3 to stay at ~10% (random chance).

    Solution: THREE INDEPENDENT CNNs (one per digit position).
    Each digit gets its own complete feature extractor.

    Architecture:
        Input (84x84) → Split into 3 strips (84x28 each)
        Strip 1 → CNN_1 → Softmax (out_1: 0-9)
        Strip 2 → CNN_2 → Softmax (out_2: 0-9)
        Strip 3 → CNN_3 → Softmax (out_3: 0-9)
    """

    # Input: full 84x84 grayscale image
    img_in = layers.Input(shape=(84, 84, 1), name='img_in')

    # Split into three 84x28 vertical strips using Cropping2D
    left_strip = layers.Cropping2D(cropping=((0, 0), (0, 56)), name='crop_left')(img_in)
    mid_strip = layers.Cropping2D(cropping=((0, 0), (28, 28)), name='crop_mid')(img_in)
    right_strip = layers.Cropping2D(cropping=((0, 0), (56, 0)), name='crop_right')(img_in)

    def create_digit_cnn(name):
        """Create an independent CNN for one digit position."""
        return models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
        ], name=name)

    # THREE SEPARATE CNNs - one for each digit
    cnn_left = create_digit_cnn('cnn_digit1')
    cnn_mid = create_digit_cnn('cnn_digit2')
    cnn_right = create_digit_cnn('cnn_digit3')

    # Process each strip with its own CNN
    feat_left = cnn_left(left_strip)
    feat_mid = cnn_mid(mid_strip)
    feat_right = cnn_right(right_strip)

    # Output heads
    out1 = layers.Dense(num_classes, activation='softmax', name='out_1')(feat_left)
    out2 = layers.Dense(num_classes, activation='softmax', name='out_2')(feat_mid)
    out3 = layers.Dense(num_classes, activation='softmax', name='out_3')(feat_right)

    model = Model(inputs=img_in, outputs=[out1, out2, out3])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={
            'out_1': 'sparse_categorical_crossentropy',
            'out_2': 'sparse_categorical_crossentropy',
            'out_3': 'sparse_categorical_crossentropy'
        },
        loss_weights={'out_1': 1.0, 'out_2': 1.0, 'out_3': 1.0},
        metrics={
            'out_1': 'accuracy',
            'out_2': 'accuracy',
            'out_3': 'accuracy'
        }
    )

    return model


def run_benchmark_split_cnn(model, train_ds, val_ds, test_ds, epochs):
    """Train and evaluate the split CNN model."""

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    train_time = time.time() - start

    # Evaluate on test set
    print("\nEvaluating on test set...")

    all_images = []
    all_labels_1 = []
    all_labels_2 = []
    all_labels_3 = []

    for batch_images, batch_labels in test_ds:
        all_images.append(batch_images.numpy())
        all_labels_1.append(batch_labels['out_1'].numpy())
        all_labels_2.append(batch_labels['out_2'].numpy())
        all_labels_3.append(batch_labels['out_3'].numpy())

    X_test = np.concatenate(all_images, axis=0)
    y1_true = np.concatenate(all_labels_1, axis=0)
    y2_true = np.concatenate(all_labels_2, axis=0)
    y3_true = np.concatenate(all_labels_3, axis=0)

    y_true = y1_true * 100 + y2_true * 10 + y3_true

    preds_raw = model.predict(X_test, verbose=0)
    p1 = np.argmax(preds_raw[0], axis=1)
    p2 = np.argmax(preds_raw[1], axis=1)
    p3 = np.argmax(preds_raw[2], axis=1)
    y_pred = (p1 * 100) + (p2 * 10) + p3

    acc_1 = accuracy_score(y1_true, p1)
    acc_2 = accuracy_score(y2_true, p2)
    acc_3 = accuracy_score(y3_true, p3)

    print(f"\nPer-digit test accuracies:")
    print(f"  Digit 1 (left):   {acc_1:.4f}")
    print(f"  Digit 2 (middle): {acc_2:.4f}")
    print(f"  Digit 3 (right):  {acc_3:.4f}")
    print(f"  Combined (product): {acc_1 * acc_2 * acc_3:.4f}")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='macro'),
        "training_time": train_time,
        "history": history.history,
        "per_digit_accuracy": {
            "digit_1": acc_1,
            "digit_2": acc_2,
            "digit_3": acc_3
        }
    }, y_true, y_pred


# ============================================================================
# TASK 5A: MULTI-LABEL CNN WITHOUT SPLITTING
# ============================================================================



def build_multilabel_cnn(img_size=(84, 84), learning_rate=0.001):
    inputs = layers.Input(shape=(*img_size, 1), name='img_in')

    x = layers.RandomTranslation(0.1, 0.1)(inputs)
    x = layers.RandomRotation(0.05)(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    shared = layers.Dense(512, activation='relu', name='shared_dense')(x)

    out1 = layers.Dense(10, activation='softmax', name='out_1')(shared)
    out2 = layers.Dense(10, activation='softmax', name='out_2')(shared)
    out3 = layers.Dense(10, activation='softmax', name='out_3')(shared)

    model = models.Model(inputs=inputs, outputs=[out1, out2, out3])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_multilabel_cnn(train_ds, val_ds, test_ds, epochs=15):
    model = build_multilabel_cnn()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    high_accuracy_stop = AccuracyThresholdCallback(threshold=0.89)

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop, high_accuracy_stop],
        verbose=1
    )
    training_time = time.time() - start_time

    all_images = []
    all_labels_1 = []
    all_labels_2 = []
    all_labels_3 = []

    for batch_images, batch_labels in test_ds:
        all_images.append(batch_images.numpy())
        all_labels_1.append(batch_labels['out_1'].numpy())
        all_labels_2.append(batch_labels['out_2'].numpy())
        all_labels_3.append(batch_labels['out_3'].numpy())

    X_test = np.concatenate(all_images, axis=0)
    y1_true = np.concatenate(all_labels_1, axis=0)
    y2_true = np.concatenate(all_labels_2, axis=0)
    y3_true = np.concatenate(all_labels_3, axis=0)

    y_true = y1_true * 100 + y2_true * 10 + y3_true

    preds_raw = model.predict(X_test, verbose=0)
    p1 = np.argmax(preds_raw[0], axis=1)
    p2 = np.argmax(preds_raw[1], axis=1)
    p3 = np.argmax(preds_raw[2], axis=1)
    y_pred = (p1 * 100) + (p2 * 10) + p3

    return {
        'model': model,
        'history': history.history,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='macro'),
        'training_time': training_time
    }


# ============================================================================
# TASK 5B: GAN (DCGAN)
# ============================================================================

def build_generator(latent_dim=100):
    model = models.Sequential(name='generator')
    model.add(layers.Dense(11 * 11 * 256, use_bias=False, input_dim=latent_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((11, 11, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(1, (5, 5), padding='valid', activation='tanh'))
    return model


def build_discriminator(img_size=(84, 84)):
    model = models.Sequential(name='discriminator')
    model.add(layers.Input(shape=(*img_size, 1)))
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


class DCGAN:
    def __init__(self, latent_dim=100, img_size=(84, 84)):
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator(img_size)
        self.discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(latent_dim,))
        generated_image = self.generator(gan_input)
        gan_output = self.discriminator(generated_image)
        self.combined = models.Model(gan_input, gan_output, name='dcgan')
        self.combined.compile(
            optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )

    def train_step(self, real_images, batch_size):
        real_labels = np.ones((batch_size, 1)) * 0.9
        fake_labels = np.zeros((batch_size, 1))
        self.discriminator.trainable = True
        d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        fake_images = self.generator.predict(noise, verbose=0)
        d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        self.discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        g_loss = self.combined.train_on_batch(noise, real_labels)
        return d_loss[0], g_loss, d_loss[1]

    def generate_images(self, num_images):
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        generated = self.generator.predict(noise, verbose=0)
        generated = (generated + 1) / 2.0
        return generated


def train_gan(X_all, epochs=50, batch_size=128, latent_dim=100):
    gan = DCGAN(latent_dim=latent_dim, img_size=(84, 84))
    batches_per_epoch = len(X_all) // batch_size
    d_losses, g_losses, d_accuracies = [], [], []

    for epoch in range(epochs):
        epoch_d_loss, epoch_g_loss, epoch_d_acc = [], [], []
        indices = np.random.permutation(len(X_all))
        X_shuffled = X_all[indices]

        for batch_idx in range(batches_per_epoch):
            real_images = X_shuffled[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            d_loss, g_loss, d_acc = gan.train_step(real_images, batch_size)
            epoch_d_loss.append(d_loss); epoch_g_loss.append(g_loss); epoch_d_acc.append(d_acc)

        d_losses.append(np.mean(epoch_d_loss))
        g_losses.append(np.mean(epoch_g_loss))
        d_accuracies.append(np.mean(epoch_d_acc))
        print(f"Epoch {epoch+1}/{epochs} - D_loss: {d_losses[-1]:.4f}, G_loss: {g_losses[-1]:.4f}, D_acc: {d_accuracies[-1]:.4f}")

    visualize_gan_training(gan, d_losses, g_losses, d_accuracies)
    return gan, d_losses, g_losses, d_accuracies


def visualize_gan_training(gan, d_losses, g_losses, d_accuracies):
    num_samples = 16
    generated_images = gan.generate_images(num_samples)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i, :, :, 0], cmap='gray')
        ax.axis('off')
    plt.savefig('generated_samples.png')
    plt.close()


def train_with_augmented_data(gan, train_ds, val_ds, test_ds, X_train_real, y_train_real,
                              num_synthetic=10000, epochs=15, batch_size=128):
    print(f"\n[1/4] Generating {num_synthetic} synthetic images...")
    synthetic_images = gan.generate_images(num_synthetic)

    print("\n[2/4] Labeling synthetic images...")
    if os.path.exists('multilabel_cnn.keras'):
        labeling_model = tf.keras.models.load_model('multilabel_cnn.keras')
        preds = labeling_model.predict(synthetic_images, verbose=0)
        p1, p2, p3 = np.argmax(preds[0], axis=1), np.argmax(preds[1], axis=1), np.argmax(preds[2], axis=1)
        y_synthetic = (p1 * 100) + (p2 * 10) + p3
    else:
        y_synthetic = np.random.randint(0, 1000, num_synthetic)

    X_train_augmented = np.concatenate([X_train_real, synthetic_images], axis=0)
    y_train_augmented = np.concatenate([y_train_real, y_synthetic], axis=0)

    def create_multilabel_dataset(X, y, batch_size, shuffle=True):
        digit1, digit2, digit3 = y // 100, (y % 100) // 10, y % 10
        dataset = tf.data.Dataset.from_tensor_slices((X, {'out_1': digit1, 'out_2': digit2, 'out_3': digit3}))
        if shuffle: dataset = dataset.shuffle(10000)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds_augmented = create_multilabel_dataset(X_train_augmented, y_train_augmented, batch_size)

    def extract_images(inputs, labels):
        return (inputs['img_in'] if isinstance(inputs, dict) else inputs), labels

    val_ds_extracted = val_ds.map(extract_images)
    test_ds_extracted = test_ds.map(extract_images)

    model_augmented = build_multilabel_cnn()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    high_accuracy_stop = AccuracyThresholdCallback(threshold=0.89)

    print("\n[3/4] Training on augmented data...")
    start_time = time.time()
    history = model_augmented.fit(
        train_ds_augmented,
        validation_data=val_ds_extracted,
        epochs=epochs,
        callbacks=[early_stop, high_accuracy_stop],
        verbose=1
    )
    training_time_augmented = time.time() - start_time

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Augmented Total Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['out_1_accuracy'], label='Train Acc'); plt.plot(history.history['val_out_1_accuracy'], label='Val Acc')
    plt.legend(); plt.title('Digit 1 Accuracy')
    plt.savefig('augmented_training_curves.png'); plt.close()

    print("\n[4/4] Evaluating augmented model...")

    all_images = []
    all_labels_1 = []
    all_labels_2 = []
    all_labels_3 = []

    for batch_images, batch_labels in test_ds:
        all_images.append(batch_images.numpy())
        all_labels_1.append(batch_labels['out_1'].numpy())
        all_labels_2.append(batch_labels['out_2'].numpy())
        all_labels_3.append(batch_labels['out_3'].numpy())

    X_test = np.concatenate(all_images, axis=0)
    y1_true = np.concatenate(all_labels_1, axis=0)
    y2_true = np.concatenate(all_labels_2, axis=0)
    y3_true = np.concatenate(all_labels_3, axis=0)

    y_true = y1_true * 100 + y2_true * 10 + y3_true

    preds_raw = model_augmented.predict(X_test, verbose=0)
    p1 = np.argmax(preds_raw[0], axis=1)
    p2 = np.argmax(preds_raw[1], axis=1)
    p3 = np.argmax(preds_raw[2], axis=1)
    y_pred = (p1 * 100) + (p2 * 10) + p3

    return {
        'model': model_augmented,
        'history': history.history,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='macro'),
        'training_time': training_time_augmented
    }