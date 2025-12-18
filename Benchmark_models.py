import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression


def create_cnn_model(img_size, num_classes):
    model = models.Sequential([
        layers.Input(shape=(*img_size, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def run_benchmark_lr(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', verbose=0)
    results = {}

    start = time.time()
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    results["training_time"] = time.time() - start

    start = time.time()
    preds = model.predict(X_test.reshape(len(X_test), -1))
    results["prediction_time"] = time.time() - start

    results["accuracy"] = accuracy_score(y_test, preds)
    results["f1_score"] = f1_score(y_test, preds, average='macro')
    return results, model


def run_benchmark_cnn(model, X_train, X_test, y_train, y_test, X_val, y_val, epochs, batch_size):
    results = {}

    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose= 1
    )
    results["training_time"] = time.time() - start

    start = time.time()
    preds = model.predict(X_test, verbose=0).argmax(axis=1)
    results["prediction_time"] = time.time() - start

    results["accuracy"] = accuracy_score(y_test, preds)
    results["f1_score"] = f1_score(y_test, preds, average='macro')
    results["history"] = history.history
    return results


def run_benchmark_split_cnn(model, train_ds, val_ds, test_ds, epochs):
    """
    Benchmarks the Siamese/Split CNN architecture.
    Handles dictionary inputs/outputs directly from tf.data.Dataset.
    """
    results = {}

    print(f"Training Split CNN for {epochs} epochs...")
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose= 1 )
    results["training_time"] = time.time() - start

    print("Evaluating Split CNN...")
    start = time.time()
    # Predictions will be a list of 3 arrays: [out1, out2, out3]
    preds_raw = model.predict(test_ds, verbose=0)
    results["prediction_time"] = time.time() - start

    # Convert softmax outputs to class labels for each head
    p1 = np.argmax(preds_raw[0], axis=1)
    p2 = np.argmax(preds_raw[1], axis=1)
    p3 = np.argmax(preds_raw[2], axis=1)

    # Reconstruct the 3-digit number: (p1 * 100) + (p2 * 10) + p3
    y_pred_total = (p1 * 100) + (p2 * 10) + p3

    # Extract true labels from the dataset for comparison
    y_true_batches = []
    for _, labels in test_ds:
        # Reconstruct true 3-digit labels from dict: {'out_1': d1, ...}
        batch_true = (labels['out_1'].numpy() * 100) + \
                     (labels['out_2'].numpy() * 10) + \
                     labels['out_3'].numpy()
        y_true_batches.append(batch_true)

    y_true_total = np.concatenate(y_true_batches)

    results["accuracy"] = accuracy_score(y_true_total, y_pred_total)
    results["f1_score"] = f1_score(y_true_total, y_pred_total, average='macro')
    results["history"] = history.history

    return results, y_true_total, y_pred_total