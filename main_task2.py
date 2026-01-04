import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import Builder as bm
import pre_processing as pp
import visualise as vs
from visualise import plot_hyperparameter_tuning

devices = tf.config.list_physical_devices()
print(f"Total available devices: {devices}")

gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.set_visible_devices(gpu_devices[0], 'GPU')
        print(f"SUCCESS: M2 GPU ('{gpu_devices[0].name}') is active.")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")
else:
    print("GPU NOT found. Using CPU.")

tf.config.set_soft_device_placement(True)


def main():

    # Configuration
    img_size = (84, 84)
    batch_size = 128
    epochs = 10

    # For testing/debugging, set to True to use subset of data
    USE_SUBSET = True  # Set to True for quick testing (~15 min vs 1-2 hours)
    SUBSET_SIZE = 5000


    print(" Model Benchmarking (Logistic Regression vs CNN)")


    # 1. Load and prepare datasets
    print("\n[1/5] Loading datasets...")
    train_ds = pp.prepare_dataset("triple_mnist/train/", img_size, batch_size)
    val_ds = pp.prepare_dataset("triple_mnist/val/", img_size, batch_size)
    test_ds = pp.prepare_dataset("triple_mnist/test/", img_size, batch_size)

    # Convert to NumPy arrays for scikit-learn compatibility
    X_train, y_train = pp.split_dataset(train_ds)
    X_val, y_val = pp.split_dataset(val_ds)
    X_test, y_test = pp.split_dataset(test_ds)

    # Optional: Use subset for faster testing/debugging
    if USE_SUBSET:
        print(f"DEBUG MODE ACTIVE")
        X_train = X_train[:SUBSET_SIZE]
        y_train = y_train[:SUBSET_SIZE]
        X_val = X_val[:SUBSET_SIZE // 4]
        y_val = y_val[:SUBSET_SIZE // 4]
        X_test = X_test[:SUBSET_SIZE // 4]
        y_test = y_test[:SUBSET_SIZE // 4]

    # Determine number of classes from the data
    num_classes = 1000

    # 2. Hyperparameter tuning for Logistic Regression
    print("\n[2/5] Tuning Logistic Regression hyperparameters...")
    lr_tune = []
    for c in [0.1, 1.0]:
        print(f"  Testing C={c}...")
        res, _ = bm.run_benchmark_lr(X_train, X_val, y_train, y_val, C=c)
        lr_tune.append({'param': c, 'val_accuracy': res['accuracy']})
        print(f"    Validation Accuracy: {res['accuracy']:.4f}")

    best_c = max(lr_tune, key=lambda x: x['val_accuracy'])['param']
    print(f"  Best C parameter: {best_c}")

    # 3. Hyperparameter tuning for CNN
    print("\n[3/5] Tuning CNN hyperparameters...")
    cnn_tune = []
    for lr_rate in [1e-3, 5e-4, 1e-4]:
        print(f"  Testing learning rate={lr_rate}...")
        model = bm.create_cnn_model(img_size, num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        h = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=2,
            batch_size=batch_size,
            verbose=0
        )
        cnn_tune.append({'param': lr_rate, 'val_accuracy': h.history['val_accuracy'][-1]})
        print(f"    Validation Accuracy: {h.history['val_accuracy'][-1]:.4f}")

    best_lr = max(cnn_tune, key=lambda x: x['val_accuracy'])['param']
    print(f"  Best learning rate: {best_lr}")

    # 4. Train final models with best hyperparameters
    print("\n[4/5] Training final models on full training set...")

    # Logistic Regression
    print("  Training Logistic Regression...")
    res_lr, model_lr = bm.run_benchmark_lr(X_train, X_test, y_train, y_test, C=best_c)
    print(f"    Test Accuracy: {res_lr['accuracy']:.4f}")
    print(f"    Test F1 Score: {res_lr['f1_score']:.4f}")
    print(f"    Training Time: {res_lr['training_time']:.2f}s")

    # CNN
    print("  Training CNN...")
    final_cnn = bm.create_cnn_model(img_size, num_classes)
    final_cnn.compile(
        optimizer=tf.keras.optimizers.Adam(best_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    res_cnn = bm.run_benchmark_cnn(
        final_cnn, X_train, X_test, y_train, y_test, X_val, y_val, epochs, batch_size
    )
    print(f"    Test Accuracy: {res_cnn['accuracy']:.4f}")
    print(f"    Test F1 Score: {res_cnn['f1_score']:.4f}")
    print(f"    Training Time: {res_cnn['training_time']:.2f}s")

    # 5. Create comparison DataFrame
    df_results = pd.DataFrame(
        [res_lr, res_cnn],
        index=['Logistic Regression', 'Standard CNN']
    )

    print("RESULTS SUMMARY")

    print(df_results.to_string())


    # Save the best model
    print("\n[5/5] Saving best model...")
    if res_lr['accuracy'] > res_cnn['accuracy']:
        joblib.dump(model_lr, 'best_model.pkl')
        print("Logistic Regression performed better.")
        print("Saved: best_model.pkl")
    else:
        final_cnn.save('best_model.keras')
        print("CNN performed better.")
        print("Saved: best_model.keras")

    # Generate visualizations
    print("\nGenerating visualizations...")
    cnn_preds = np.argmax(final_cnn.predict(X_test, verbose=0), axis=1)
    vs.plot_benchmark_results(df_results, res_cnn['history'], lr_tune, y_test, cnn_preds)
    vs,plot_hyperparameter_tuning(lr_tune, cnn_tune)



if __name__ == "__main__":
    main()