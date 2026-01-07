

import time
import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import os



# Helper Functions


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
def extract_split_dataset(dataset):
    """
    Extract images and labels from a split dataset.
    """
    all_images = []
    all_labels_1 = []
    all_labels_2 = []
    all_labels_3 = []

    for batch_inputs, batch_labels in dataset:
        if isinstance(batch_inputs, dict):
            all_images.append(batch_inputs['img_in'].numpy())
        else:
            all_images.append(batch_inputs.numpy())
        all_labels_1.append(batch_labels['out_1'].numpy())
        all_labels_2.append(batch_labels['out_2'].numpy())
        all_labels_3.append(batch_labels['out_3'].numpy())

    X = np.concatenate(all_images, axis=0)
    y1 = np.concatenate(all_labels_1, axis=0)
    y2 = np.concatenate(all_labels_2, axis=0)
    y3 = np.concatenate(all_labels_3, axis=0)

    return X, y1, y2, y3



# TASK 2: Baseline Models


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


# TASK 4: Split CNN

def build_dual_path_cnn(num_classes=10, learning_rate=0.001):
    img_in = layers.Input(shape=(84, 84, 1), name='img_in')

    left_strip = layers.Cropping2D(cropping=((0, 0), (0, 56)), name='crop_left')(img_in)
    mid_strip = layers.Cropping2D(cropping=((0, 0), (28, 28)), name='crop_mid')(img_in)
    right_strip = layers.Cropping2D(cropping=((0, 0), (56, 0)), name='crop_right')(img_in)

    def create_digit_cnn(name):
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

    cnn_left = create_digit_cnn('cnn_digit1')
    cnn_mid = create_digit_cnn('cnn_digit2')
    cnn_right = create_digit_cnn('cnn_digit3')

    feat_left = cnn_left(left_strip)
    feat_mid = cnn_mid(mid_strip)
    feat_right = cnn_right(right_strip)

    out1 = layers.Dense(num_classes, activation='softmax', name='out_1')(feat_left)
    out2 = layers.Dense(num_classes, activation='softmax', name='out_2')(feat_mid)
    out3 = layers.Dense(num_classes, activation='softmax', name='out_3')(feat_right)

    model = Model(inputs=img_in, outputs=[out1, out2, out3])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def run_benchmark_split_cnn(model, train_ds, val_ds, test_ds, epochs):
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

    print("\nEvaluating on test set...")
    X_test, y1_true, y2_true, y3_true = extract_split_dataset(test_ds)
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

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='macro'),
        "training_time": train_time,
        "history": history.history,
        "per_digit_accuracy": {"digit_1": acc_1, "digit_2": acc_2, "digit_3": acc_3}
    }, y_true, y_pred



# TASK 5A: Multi-label CNN

def build_multilabel_cnn(input_shape=(84, 84, 1)):
    inputs = layers.Input(shape=input_shape, name='img_in')

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    out_1 = layers.Dense(10, activation='softmax', name='out_1')(x)
    out_2 = layers.Dense(10, activation='softmax', name='out_2')(x)
    out_3 = layers.Dense(10, activation='softmax', name='out_3')(x)

    model = models.Model(inputs=inputs, outputs=[out_1, out_2, out_3])

    model.compile(
        optimizer='adam',
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


def train_multilabel_cnn(train_ds, val_ds, test_ds, epochs=15, quick_test=False, batch_size=128):
    """
    Train multi-label CNN for Task 5.
    """
    print("Extracting data from datasets...")
    X_train, y1_train, y2_train, y3_train = extract_split_dataset(train_ds)
    X_val, y1_val, y2_val, y3_val = extract_split_dataset(val_ds)
    X_test, y1_test, y2_test, y3_test = extract_split_dataset(test_ds)

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    if quick_test:
        subset_train = min(20000, len(X_train))
        subset_val = min(4000, len(X_val))

        indices = np.random.choice(len(X_train), subset_train, replace=False)
        X_train = X_train[indices]
        y1_train, y2_train, y3_train = y1_train[indices], y2_train[indices], y3_train[indices]

        indices_val = np.random.choice(len(X_val), subset_val, replace=False)
        X_val = X_val[indices_val]
        y1_val, y2_val, y3_val = y1_val[indices_val], y2_val[indices_val], y3_val[indices_val]

        print(f"  Quick test subset: {subset_train} train, {subset_val} val")

    model = build_multilabel_cnn()

    accuracyThreshold = AccuracyThresholdCallback(0.89)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_out_1_accuracy',
        mode='max',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_out_1_accuracy',
        mode='max',
        factor=0.5,
        patience=2,
        verbose=1
    )

    start_time = time.time()

    history = model.fit(
        {'img_in': X_train},
        {'out_1': y1_train, 'out_2': y2_train, 'out_3': y3_train},
        validation_data=(
            {'img_in': X_val},
            {'out_1': y1_val, 'out_2': y2_val, 'out_3': y3_val}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr,accuracyThreshold ],
        verbose=1
    )

    training_time = time.time() - start_time

    print("\nEvaluating on test set...")
    preds = model.predict({'img_in': X_test}, verbose=0)
    p1 = np.argmax(preds[0], axis=1)
    p2 = np.argmax(preds[1], axis=1)
    p3 = np.argmax(preds[2], axis=1)

    y_true_combined = y1_test * 100 + y2_test * 10 + y3_test
    y_pred_combined = p1 * 100 + p2 * 10 + p3

    accuracy = accuracy_score(y_true_combined, y_pred_combined)
    f1 = f1_score(y_true_combined, y_pred_combined, average='macro')

    acc1 = accuracy_score(y1_test, p1)
    acc2 = accuracy_score(y2_test, p2)
    acc3 = accuracy_score(y3_test, p3)

    print(f"\nPer-digit accuracy: D1={acc1:.4f}, D2={acc2:.4f}, D3={acc3:.4f}")

    return {
        'model': model,
        'history': history.history,
        'accuracy': accuracy,
        'f1_score': f1,
        'per_digit_accuracy': [acc1, acc2, acc3],
        'training_time': training_time,
        'y_true': y_true_combined,
        'y_pred': y_pred_combined
    }



# TASK 5B: GAN early stopping


class GANEarlyStopping:
    """
    Early stopping for GAN training based on multiple criteria.

    Criteria:
    1. Discriminator accuracy reaches target threshold (balanced training)
    2. Mode collapse detected (low diversity)
    3. Loss ratio becomes too imbalanced

    """

    def __init__(self,
                 d_acc_threshold=0.5,
                 d_acc_tolerance=0.1,
                 d_acc_patience=10,
                 diversity_threshold=10.0,
                 diversity_patience=5,
                 loss_ratio_threshold=10.0,
                 verbose=True):

        self.d_acc_threshold = d_acc_threshold
        self.d_acc_tolerance = d_acc_tolerance
        self.d_acc_patience = d_acc_patience
        self.diversity_threshold = diversity_threshold
        self.diversity_patience = diversity_patience
        self.loss_ratio_threshold = loss_ratio_threshold
        self.verbose = verbose

        # Counters
        self.acc_counter = 0
        self.diversity_counter = 0
        self.stop_reason = None
        self.stopped = False

    def should_stop(self, d_acc, d_loss, g_loss, diversity=None):
        '''Check if training should stop'''
        if self.stopped:
            return True

        # Check 1: Discriminator accuracy at target (GOOD - converged)
        if abs(d_acc - self.d_acc_threshold) <= self.d_acc_tolerance:
            self.acc_counter += 1
            if self.acc_counter >= self.d_acc_patience:
                self.stop_reason = (
                    f"CONVERGED: D accuracy stable at {d_acc:.4f} "
                    f"(target: {self.d_acc_threshold}±{self.d_acc_tolerance}) "
                    f"for {self.d_acc_patience} epochs"
                )
                self.stopped = True
                return True
        else:
            self.acc_counter = 0

        # Check 2: Mode collapse (BAD - low diversity)
        if diversity is not None and diversity < self.diversity_threshold:
            self.diversity_counter += 1
            if self.diversity_counter >= self.diversity_patience:
                self.stop_reason = (
                    f"MODE COLLAPSE: Diversity {diversity:.2f} below threshold "
                    f"{self.diversity_threshold} for {self.diversity_patience} epochs"
                )
                self.stopped = True
                return True
        else:
            self.diversity_counter = 0

        # Check 3: Extremely imbalanced loss ratio (BAD)
        if d_loss > 0:
            loss_ratio = g_loss / (d_loss + 1e-8)
            if loss_ratio > self.loss_ratio_threshold:
                self.stop_reason = (
                    f"IMBALANCED: G_loss/D_loss ratio {loss_ratio:.2f} "
                    f"exceeds threshold {self.loss_ratio_threshold}"
                )
                self.stopped = True
                return True

        return False

    def reset(self):
        """Reset all counters."""
        self.acc_counter = 0
        self.diversity_counter = 0
        self.stop_reason = None
        self.stopped = False



# TASK 5B: optimised DCGAN


class DCGAN:


    def __init__(self, latent_dim=128, img_size=(84, 84)):
        self.latent_dim = latent_dim
        self.latent_dim_tensor = tf.constant(latent_dim, dtype=tf.int32)
        self.img_size = img_size

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Optimizers - lower LR for discriminator
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0002,
            beta_1=0.5,
            beta_2=0.999
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.5,
            beta_2=0.999
        )

        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)


        self.fixed_noise = tf.random.normal([16, latent_dim])


        self._train_step = tf.function(self._train_step_impl)

    def _build_generator(self):
        """
        Generator: latent vector → 84×84×1 image
        Architecture: 6→12→24→48→96 then crop to 84
        """
        model = models.Sequential(name='generator')

        # Project and reshape: latent_dim → 6×6×512
        model.add(layers.Dense(6 * 6 * 512, use_bias=False, input_dim=self.latent_dim))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Reshape((6, 6, 512)))

        # 6×6 → 12×12
        model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))

        # 12×12 → 24×24
        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))

        # 24×24 → 48×48
        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))

        # 48×48 → 96×96
        model.add(layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))

        # Final conv to 1 channel
        model.add(layers.Conv2D(1, (3, 3), padding='same', activation='tanh'))

        # Crop 96×96 → 84×84
        model.add(layers.Cropping2D(cropping=((6, 6), (6, 6))))

        return model

    def _build_discriminator(self):
        """
        Discriminator: 84×84×1 image → real/fake probability
        No batch normalization (causes training issues).
        """
        model = models.Sequential(name='discriminator')

        model.add(layers.Input(shape=(84, 84, 1)))

        # Add noise for stability
        model.add(layers.GaussianNoise(0.1))

        # 84×84 → 42×42
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.25))

        # 42×42 → 21×21
        model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.25))

        # 21×21 → 11×11
        model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.25))

        # 11×11 → 6×6
        model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def _train_step_impl(self, real_images):
        """
        Internal training step implementation.
        """
        batch_size = tf.shape(real_images)[0]

        # Label smoothing for stability
        real_labels = tf.ones((batch_size, 1)) * 0.9
        fake_labels = tf.zeros((batch_size, 1)) + 0.1

        # Train Discriminator
        noise = tf.random.normal((batch_size, self.latent_dim))

        with tf.GradientTape() as d_tape:
            fake_images = self.generator(noise, training=True)
            real_pred = self.discriminator(real_images, training=True)
            fake_pred = self.discriminator(fake_images, training=True)

            d_loss_real = self.bce(real_labels, real_pred)
            d_loss_fake = self.bce(fake_labels, fake_pred)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        #  Train Generator
        noise = tf.random.normal((batch_size, self.latent_dim))

        with tf.GradientTape() as g_tape:
            fake_images = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake_images, training=False)
            g_loss = self.bce(tf.ones((batch_size, 1)), fake_pred)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        # Calculate discriminator accuracy
        real_acc = tf.reduce_mean(tf.cast(real_pred > 0.5, tf.float32))
        fake_acc = tf.reduce_mean(tf.cast(fake_pred < 0.5, tf.float32))
        d_acc = 0.5 * (real_acc + fake_acc)

        return d_loss, g_loss, d_acc

    def train_step(self, real_images):
        """Public training step that uses compiled function."""
        return self._train_step(real_images)

    def generate_images(self, num_images):
        """Generate images for visualization/evaluation."""
        noise = tf.random.normal([num_images, self.latent_dim])
        generated = self.generator(noise, training=False)
        generated = (generated + 1.0) / 2.0  # [-1,1] → [0,1]
        return generated.numpy()

    def generate_from_fixed_noise(self):
        """Generate images from fixed noise for consistent visualization."""
        generated = self.generator(self.fixed_noise, training=False)
        generated = (generated + 1.0) / 2.0
        return generated.numpy()

    def check_mode_collapse(self, num_samples=100):
        """Check for mode collapse by measuring generation diversity."""
        images = self.generate_images(num_samples)
        images = (images * 2.0) - 1.0

        pixel_std = np.std(images, axis=0).mean()

        flat_images = images.reshape(num_samples, -1)
        indices = np.random.choice(num_samples, size=(min(100, num_samples), 2), replace=True)
        distances = np.linalg.norm(
            flat_images[indices[:, 0]] - flat_images[indices[:, 1]],
            axis=1
        )
        mean_distance = np.mean(distances)

        return {
            'pixel_std': float(pixel_std),
            'mean_pairwise_distance': float(mean_distance),
            'is_collapsed': pixel_std < 0.1 or mean_distance < 10
        }


def train_gan(X_all, epochs=50, batch_size=128, latent_dim=128,
              save_interval=10, early_stop_patience=15,
              d_acc_threshold=0.5, d_acc_patience=10):
    """
    Optimized GAN training function with accuracy-based early stopping.
    """

    print("DCGAN TRAINING ")


    gan = DCGAN(latent_dim=latent_dim, img_size=(84, 84))

    print("\nGenerator architecture:")
    gan.generator.summary()
    print("\nDiscriminator architecture:")
    gan.discriminator.summary()

    # Create optimized dataset with caching
    dataset = tf.data.Dataset.from_tensor_slices(X_all)
    dataset = dataset.shuffle(buffer_size=min(10000, len(X_all)))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.cache()  # Cache in memory for faster epochs
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = len(X_all) // batch_size
    print(f"\nTraining configuration:")
    print(f"  Samples: {len(X_all)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Epochs: {epochs}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  D accuracy threshold: {d_acc_threshold} (±0.1)")
    print(f"  D accuracy patience: {d_acc_patience} epochs")

    d_losses, g_losses, d_accuracies = [], [], []
    best_diversity = 0
    patience_counter = 0
    acc_threshold_counter = 0  # Counter for accuracy-based stopping


    print("Starting training...")


    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Use lists to accumulate losses
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_d_acc = []

        for step, batch in enumerate(dataset):
            d_loss, g_loss, d_acc = gan.train_step(batch)

            # Only convert to float at the end of accumulation
            epoch_d_loss.append(d_loss)
            epoch_g_loss.append(g_loss)
            epoch_d_acc.append(d_acc)

            # Print progress less frequently to reduce overhead
            if (step + 1) % 200 == 0:
                print(f"  Epoch {epoch+1} - Step {step+1}/{steps_per_epoch}", end='\r')

        # Convert to numpy only once per epoch
        mean_d_loss = float(tf.reduce_mean(epoch_d_loss))
        mean_g_loss = float(tf.reduce_mean(epoch_g_loss))
        mean_d_acc = float(tf.reduce_mean(epoch_d_acc))

        d_losses.append(mean_d_loss)
        g_losses.append(mean_g_loss)
        d_accuracies.append(mean_d_acc)

        epoch_time = time.time() - epoch_start

        # Accuracy-based early stopping

        if abs(mean_d_acc - d_acc_threshold) <= 0.1:
            acc_threshold_counter += 1
            if acc_threshold_counter >= d_acc_patience:
                print(f"\n✓ Discriminator accuracy stable at {mean_d_acc:.4f} for {d_acc_patience} epochs.")
                print(f"  Training converged! Stopping early at epoch {epoch + 1}.")
                # Save final samples before stopping
                _save_epoch_samples(gan, epoch + 1)
                break
        else:
            acc_threshold_counter = 0  # Reset counter if accuracy moves away

        # Mode collapse check every 5 epochs
        if (epoch + 1) % 5 == 0:
            collapse_check = gan.check_mode_collapse()
            diversity = collapse_check['mean_pairwise_distance']

            print(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s) - "
                  f"D_loss: {mean_d_loss:.4f}, G_loss: {mean_g_loss:.4f}, "
                  f"D_acc: {mean_d_acc:.4f}, Diversity: {diversity:.2f}")

            if collapse_check['is_collapsed']:
                print("Possible mode collapse detected!")
                patience_counter += 1
                if patience_counter >= early_stop_patience // 5:
                    print("Stopping early due to mode collapse")
                    break
            else:
                patience_counter = 0
                if diversity > best_diversity:
                    best_diversity = diversity
                    print(f"New best diversity: {diversity:.2f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s) - "
                  f"D_loss: {mean_d_loss:.4f}, G_loss: {mean_g_loss:.4f}, "
                  f"D_acc: {mean_d_acc:.4f}")

        # Save samples periodically
        if (epoch + 1) % save_interval == 0:
            _save_epoch_samples(gan, epoch + 1)

    total_time = time.time() - start_time


    return gan, d_losses, g_losses, d_accuracies


def _save_epoch_samples(gan, epoch, num_samples=16):
    """Save generated samples at checkpoint."""
    images = gan.generate_from_fixed_noise()

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i, :, :, 0], cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Generated Samples - Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'generated_epoch_{epoch}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved samples: generated_epoch_{epoch}.png")



# TASK 5B: Augumented Training


def train_with_augmented_data(gan, train_ds, val_ds, test_ds,
                              X_train_real, y_train_real,
                              num_synthetic=10000, epochs=15,
                              batch_size=128,
                              confidence_threshold=0.8):


    print("Augumented Training Pipeline")


    # Step 1: Generate synthetic images
    print(f"\nGenerating {num_synthetic} synthetic images...")
    synthetic_images = gan.generate_images(num_synthetic)
    print(f"Generated shape: {synthetic_images.shape}")

    # Step 2: Quality assessment
    print("\nAssessing synthetic image quality...")
    syn_mean, syn_std = np.mean(synthetic_images), np.std(synthetic_images)
    real_mean, real_std = np.mean(X_train_real), np.std(X_train_real)

    print(f"  Synthetic - Mean: {syn_mean:.4f}, Std: {syn_std:.4f}")
    print(f"  Real      - Mean: {real_mean:.4f}, Std: {real_std:.4f}")

    # Step 3: Pseudo-labeling
    print(f"\nPseudo-labeling with confidence threshold {confidence_threshold}...")

    if not os.path.exists('multilabel_cnn.keras'):
        raise FileNotFoundError("Baseline model 'multilabel_cnn.keras' not found!")

    labeling_model = tf.keras.models.load_model('multilabel_cnn.keras')
    preds = labeling_model.predict({'img_in': synthetic_images}, verbose=0)

    p1 = np.argmax(preds[0], axis=1)
    p2 = np.argmax(preds[1], axis=1)
    p3 = np.argmax(preds[2], axis=1)

    conf1 = np.max(preds[0], axis=1)
    conf2 = np.max(preds[1], axis=1)
    conf3 = np.max(preds[2], axis=1)

    min_confidence = np.minimum(np.minimum(conf1, conf2), conf3)
    high_conf_mask = min_confidence >= confidence_threshold
    num_high_conf = np.sum(high_conf_mask)

    print(f"  High-confidence samples: {num_high_conf}/{num_synthetic} "
          f"({100 * num_high_conf / num_synthetic:.1f}%)")

    if num_high_conf < 1000:
        print(f"Only {num_high_conf} high-confidence samples.")
        print(f"Lowering threshold to 0.5...")
        confidence_threshold = 0.5
        high_conf_mask = min_confidence >= confidence_threshold
        num_high_conf = np.sum(high_conf_mask)
        print(f"New high-confidence count: {num_high_conf}")

    synthetic_images_filtered = synthetic_images[high_conf_mask]
    p1_filtered = p1[high_conf_mask]
    p2_filtered = p2[high_conf_mask]
    p3_filtered = p3[high_conf_mask]

    # Step 4: Combine datasets
    print(f"\nCombining datasets...")

    if len(y_train_real.shape) == 1:
        y1_real = y_train_real // 100
        y2_real = (y_train_real % 100) // 10
        y3_real = y_train_real % 10
    else:
        y1_real = y_train_real[:, 0]
        y2_real = y_train_real[:, 1]
        y3_real = y_train_real[:, 2]

    X_augmented = np.concatenate([X_train_real, synthetic_images_filtered], axis=0)
    y1_augmented = np.concatenate([y1_real, p1_filtered], axis=0)
    y2_augmented = np.concatenate([y2_real, p2_filtered], axis=0)
    y3_augmented = np.concatenate([y3_real, p3_filtered], axis=0)

    print(f"  Real samples:      {len(X_train_real)}")
    print(f"  Synthetic samples: {len(synthetic_images_filtered)}")
    print(f"  Total augmented:   {len(X_augmented)}")

    # Step 5: Train augmented model
    print(f"\nTraining augmented model...")

    X_val, y1_val, y2_val, y3_val = extract_split_dataset(val_ds)
    X_test, y1_test, y2_test, y3_test = extract_split_dataset(test_ds)

    model_augmented = build_multilabel_cnn()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_out_1_accuracy',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_out_1_accuracy',
            mode='max',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    start_time = time.time()
    history = model_augmented.fit(
        {'img_in': X_augmented},
        {'out_1': y1_augmented, 'out_2': y2_augmented, 'out_3': y3_augmented},
        validation_data=(
            {'img_in': X_val},
            {'out_1': y1_val, 'out_2': y2_val, 'out_3': y3_val}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time

    # Evaluation
    print("\nTesting augmented model...")

    preds_test = model_augmented.predict({'img_in': X_test}, verbose=0)
    p1_test = np.argmax(preds_test[0], axis=1)
    p2_test = np.argmax(preds_test[1], axis=1)
    p3_test = np.argmax(preds_test[2], axis=1)

    y_true_combined = y1_test * 100 + y2_test * 10 + y3_test
    y_pred_combined = p1_test * 100 + p2_test * 10 + p3_test

    combined_accuracy = accuracy_score(y_true_combined, y_pred_combined)
    f1 = f1_score(y_true_combined, y_pred_combined, average='macro')

    acc1 = accuracy_score(y1_test, p1_test)
    acc2 = accuracy_score(y2_test, p2_test)
    acc3 = accuracy_score(y3_test, p3_test)


    print("AUGMENTED MODEL RESULTS")

    print(f"Combined Accuracy: {combined_accuracy:.4f}")
    print(f"F1 Score (macro):  {f1:.4f}")
    print(f"Per-digit Accuracy: D1={acc1:.4f}, D2={acc2:.4f}, D3={acc3:.4f}")
    print(f"Training Time: {training_time:.1f}s")
    print(f"Synthetic samples used: {len(synthetic_images_filtered)}")

    return {
        'model': model_augmented,
        'history': history.history,
        'accuracy': combined_accuracy,
        'f1_score': f1,
        'per_digit_accuracy': [acc1, acc2, acc3],
        'training_time': training_time,
        'num_synthetic_used': len(synthetic_images_filtered),
        'y_true': y_true_combined,
        'y_pred': y_pred_combined,
        'synthetic_images': synthetic_images_filtered,
        'confidence_stats': {
            'threshold': confidence_threshold,
            'num_high_conf': num_high_conf,
            'total_generated': num_synthetic
        }
    }