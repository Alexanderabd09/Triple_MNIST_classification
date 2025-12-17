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
    return results


def run_benchmark_cnn(model, X_train, X_test, y_train, y_test, X_val, y_val, epochs, batch_size):
    results = {}

    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    results["training_time"] = time.time() - start

    start = time.time()
    preds = model.predict(X_test, verbose=0).argmax(axis=1)
    results["prediction_time"] = time.time() - start

    results["accuracy"] = accuracy_score(y_test, preds)
    results["f1_score"] = f1_score(y_test, preds, average='macro')
    results["history"] = history.history
    return results