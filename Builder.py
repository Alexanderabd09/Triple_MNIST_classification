"""
Builder.py - All model architectures and training functions
Contains models and utilities for Tasks 2, 4, and 5
"""

import time
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import os


# ============================================================================
# TASK 2: BASELINE MODELS
# ============================================================================

def create_cnn_model(img_size, num_classes):
    """
    Creates a standard CNN model with data augmentation.
    """
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
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def run_benchmark_lr(X_train, X_test, y_train, y_test, C=1.0):
    """
    Train and evaluate a Logistic Regression model.
    """
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
    """
    Train and evaluate a CNN model.
    """
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
# TASK 4: SPLIT/SIAMESE CNN
# ============================================================================

def build_dual_path_cnn(num_classes=10, learning_rate=0.001):
    """
    Creates a Siamese-style CNN that processes three 84x28 strips.
    """
    img_in = layers.Input(shape=(84, 84, 1), name="img_in")

    shared_backbone = models.Sequential([
        layers.Input(shape=(84, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ], name="siamese_backbone")

    left_v = layers.Lambda(lambda x: x[:, :, 0:28, :])(img_in)
    mid_v = layers.Lambda(lambda x: x[:, :, 28:56, :])(img_in)
    right_v = layers.Lambda(lambda x: x[:, :, 56:84, :])(img_in)

    f_left = shared_backbone(left_v)
    f_mid = shared_backbone(mid_v)
    f_right = shared_backbone(right_v)

    out1 = layers.Dense(num_classes, activation='softmax', name='out_1')(f_left)
    out2 = layers.Dense(num_classes, activation='softmax', name='out_2')(f_mid)
    out3 = layers.Dense(num_classes, activation='softmax', name='out_3')(f_right)

    model = models.Model(inputs=img_in, outputs=[out1, out2, out3])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics={'out_1': 'accuracy', 'out_2': 'accuracy', 'out_3': 'accuracy'}
    )
    return model


def run_benchmark_split_cnn(model, train_ds, val_ds, test_ds, epochs):
    """
    Train and evaluate the split/Siamese CNN model.
    """
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )
    train_time = time.time() - start

    y_true = []
    for _, batch_labels in test_ds:
        combined = batch_labels['out_1'] * 100 + batch_labels['out_2'] * 10 + batch_labels['out_3']
        y_true.append(combined.numpy())
    y_true = np.concatenate(y_true)

    preds_raw = model.predict(test_ds)
    p1 = np.argmax(preds_raw[0], axis=1)
    p2 = np.argmax(preds_raw[1], axis=1)
    p3 = np.argmax(preds_raw[2], axis=1)
    y_pred = (p1 * 100) + (p2 * 10) + p3

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='macro'),
        "training_time": train_time,
        "history": history.history
    }, y_true, y_pred


# ============================================================================
# TASK 5A: MULTI-LABEL CNN WITHOUT SPLITTING
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


def build_multilabel_cnn(img_size=(84, 84), learning_rate=0.001):
    """
    Multi-label CNN that predicts three digits simultaneously without splitting.
    """
    inputs = layers.Input(shape=(*img_size, 1), name='img_in')

    x = layers.RandomTranslation(0.1, 0.1)(inputs)
    x = layers.RandomRotation(0.05)(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    shared_features = layers.Dense(512, activation='relu')(x)
    shared_features = layers.Dropout(0.5)(shared_features)

    digit1_dense = layers.Dense(128, activation='relu', name='digit1_features')(shared_features)
    digit1_dropout = layers.Dropout(0.3)(digit1_dense)
    output_digit1 = layers.Dense(10, activation='softmax', name='out_1')(digit1_dropout)

    digit2_dense = layers.Dense(128, activation='relu', name='digit2_features')(shared_features)
    digit2_dropout = layers.Dropout(0.3)(digit2_dense)
    output_digit2 = layers.Dense(10, activation='softmax', name='out_2')(digit2_dropout)

    digit3_dense = layers.Dense(128, activation='relu', name='digit3_features')(shared_features)
    digit3_dropout = layers.Dropout(0.3)(digit3_dense)
    output_digit3 = layers.Dense(10, activation='softmax', name='out_3')(digit3_dropout)

    model = models.Model(
        inputs=inputs,
        outputs=[output_digit1, output_digit2, output_digit3],
        name='multilabel_cnn'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={
            'out_1': 'sparse_categorical_crossentropy',
            'out_2': 'sparse_categorical_crossentropy',
            'out_3': 'sparse_categorical_crossentropy'
        },
        metrics={
            'out_1': 'accuracy',
            'out_2': 'accuracy',
            'out_3': 'accuracy'
        }
    )
    return model


def train_multilabel_cnn(train_ds, val_ds, test_ds, epochs=15):
    """
    Train and evaluate multi-label CNN.
    """
    print("\n[1/3] Building multi-label CNN...")
    model = build_multilabel_cnn()

    def extract_images(inputs, labels):
        if isinstance(inputs, dict):
            return inputs['img_in'], labels
        return inputs, labels

    train_ds_extracted = train_ds.map(extract_images)
    val_ds_extracted = val_ds.map(extract_images)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    high_accuracy_stop = AccuracyThresholdCallback(threshold=0.89)

    print("\n[2/3] Training model...")
    start_time = time.time()
    history = model.fit(
        train_ds_extracted,
        validation_data=val_ds_extracted,
        epochs=epochs,
        callbacks=[early_stop, high_accuracy_stop],
        verbose=1
    )
    training_time = time.time() - start_time

    print("\n[3/3] Evaluating on test set...")
    y_true = []
    for _, batch_labels in test_ds:
        combined = batch_labels['out_1'] * 100 + batch_labels['out_2'] * 10 + batch_labels['out_3']
        y_true.append(combined.numpy())
    y_true = np.concatenate(y_true)

    test_ds_extracted = test_ds.map(extract_images)
    preds_raw = model.predict(test_ds_extracted, verbose=0)
    p1 = np.argmax(preds_raw[0], axis=1)
    p2 = np.argmax(preds_raw[1], axis=1)
    p3 = np.argmax(preds_raw[2], axis=1)
    y_pred = (p1 * 100) + (p2 * 10) + p3

    return {
        'model': model,
        'history': history.history,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='macro'),
        'training_time': training_time,
        'y_true': y_true,
        'y_pred': y_pred
    }


# ============================================================================
# TASK 5B: DCGAN IMPLEMENTATION
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
    """
    Train multi-label CNN with GAN-augmented data using Early Stopping and Accuracy Threshold logic.
    """
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

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Augmented Total Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['out_1_accuracy'], label='Train Acc'); plt.plot(history.history['val_out_1_accuracy'], label='Val Acc')
    plt.legend(); plt.title('Digit 1 Accuracy')
    plt.savefig('augmented_training_curves.png'); plt.close()

    print("\n[4/4] Evaluating augmented model...")
    y_true = []
    for _, batch_labels in test_ds:
        y_true.append((batch_labels['out_1'] * 100 + batch_labels['out_2'] * 10 + batch_labels['out_3']).numpy())
    y_true = np.concatenate(y_true)

    preds_raw = model_augmented.predict(test_ds_extracted, verbose=0)
    p1, p2, p3 = np.argmax(preds_raw[0], axis=1), np.argmax(preds_raw[1], axis=1), np.argmax(preds_raw[2], axis=1)
    y_pred = (p1 * 100) + (p2 * 10) + p3

    return {
        'model': model_augmented,
        'history': history.history,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='macro'),
        'training_time': training_time_augmented
    }