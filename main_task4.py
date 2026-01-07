import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
import Builder as bm
import pre_processing as pp
import visualise as vs

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
    IMG_SIZE = (84, 84)
    BATCH_SIZE = 128
    EPOCHS = 15

    # For testing/debugging, set to True to use subset of data
    USE_SUBSET = False  # Set to True for quick testing
    SUBSET_SIZE = 5000

    # For testing/debugging, set to True to use fewer epochs
    QUICK_TEST = False  # Set to True for quick testing (5 epochs instead of 15)
    if QUICK_TEST:
        EPOCHS = 5
        print("\nQUICK TEST MODE: Using only 5 epochs")


    print("TASK 4: Split/Siamese CNN Architecture")


    # Try to load baseline model from Task 2 for comparison
    res_baseline = None
    baseline_history = None
    y_true_baseline = None
    y_pred_baseline = None

    if os.path.exists("best_model.keras"):
        print("\n[Baseline] Loading saved model from Task 2...")
        try:
            baseline_model = tf.keras.models.load_model("best_model.keras")

            # Load test data in standard format
            print("Evaluating baseline model on test set...")
            test_ds_standard = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=False)
            X_test_baseline, y_test_baseline = pp.split_dataset(test_ds_standard)

            # Get predictions
            print("Generating baseline predictions...")
            preds_baseline = np.argmax(baseline_model.predict(X_test_baseline, verbose=0), axis=1)

            # Calculate metrics
            res_baseline = {
                "accuracy": accuracy_score(y_test_baseline, preds_baseline),
                "f1_score": f1_score(y_test_baseline, preds_baseline, average='macro'),
                "training_time": 0
            }
            y_true_baseline = y_test_baseline
            y_pred_baseline = preds_baseline

            # Get training history by loading validation data
            print("Loading baseline training data for loss comparison...")
            train_ds_standard = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=False)
            val_ds_standard = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=False)
            X_train_baseline, y_train_baseline = pp.split_dataset(train_ds_standard)
            X_val_baseline, y_val_baseline = pp.split_dataset(val_ds_standard)

            # Evaluate on train and val sets to get loss
            train_results = baseline_model.evaluate(X_train_baseline, y_train_baseline, verbose=0)
            val_results = baseline_model.evaluate(X_val_baseline, y_val_baseline, verbose=0)

            # Create minimal history for plotting
            baseline_history = {
                'loss': [train_results[0]],
                'val_loss': [val_results[0]],
                'accuracy': [train_results[1]],
                'val_accuracy': [val_results[1]]
            }

            print(f"  Baseline Test Accuracy: {res_baseline['accuracy']:.4f}")
            print(f"  Baseline Test F1 Score: {res_baseline['f1_score']:.4f}")

        except Exception as e:
            print(f"  Warning: Could not load baseline model: {e}")
            print(f"  Error details: {str(e)}")
            res_baseline = None
    else:
        print("\n[Baseline] No saved model found from Task 2.")
        print("  Run main_task2.py first to enable full comparison.")
        print("  Continuing with split model training only...")


    # 1. Load datasets with split=True for multi-output format
    print("\nLoading datasets for split model...")


    train_ds_split = pp.prepare_dataset(
        "triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=True
    )
    val_ds_split = pp.prepare_dataset(
        "triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=True
    )
    test_ds_split = pp.prepare_dataset(
        "triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=True
    )

    # Extract data from datasets (needed for hyperparameter tuning)
    X_train, y1_train, y2_train, y3_train = bm.extract_split_dataset(train_ds_split)
    X_val, y1_val, y2_val, y3_val = bm.extract_split_dataset(val_ds_split)
    X_test, y1_test, y2_test, y3_test = bm.extract_split_dataset(test_ds_split)

    # Helper function to create dataset for split format
    def create_split_dataset(images, labels_dict, batch_size):
        """Create a tf.data.Dataset from numpy arrays in split format."""
        dataset = tf.data.Dataset.from_tensor_slices((
            {"img_in": images},
            labels_dict
        ))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Use subset for faster testing/debugging
    if USE_SUBSET:
        print(f"WARNING: DEBUG MODE ACTIVE")

        # Subset the data
        X_train = X_train[:SUBSET_SIZE]
        y1_train, y2_train, y3_train = y1_train[:SUBSET_SIZE], y2_train[:SUBSET_SIZE], y3_train[:SUBSET_SIZE]

        X_val = X_val[:SUBSET_SIZE // 4]
        y1_val, y2_val, y3_val = y1_val[:SUBSET_SIZE // 4], y2_val[:SUBSET_SIZE // 4], y3_val[:SUBSET_SIZE // 4]

        X_test = X_test[:SUBSET_SIZE // 4]
        y1_test, y2_test, y3_test = y1_test[:SUBSET_SIZE // 4], y2_test[:SUBSET_SIZE // 4], y3_test[:SUBSET_SIZE // 4]

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

    # Recreate datasets from (possibly subsetted) numpy arrays
    train_ds_split = create_split_dataset(
        X_train, {"out_1": y1_train, "out_2": y2_train, "out_3": y3_train}, BATCH_SIZE
    )
    val_ds_split = create_split_dataset(
        X_val, {"out_1": y1_val, "out_2": y2_val, "out_3": y3_val}, BATCH_SIZE
    )
    test_ds_split = create_split_dataset(
        X_test, {"out_1": y1_test, "out_2": y2_test, "out_3": y3_test}, BATCH_SIZE
    )

    print("Datasets loaded and formatted for split model.")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")


    print("Datasets loaded and formatted for split model.")

    print("\n Tuning CNN hyperparameters...")

    cnn_tune = []
    for lr_rate in [1e-3, 5e-4, 1e-4]:
        # Using the already extracted numpy arrays
        print(f"  Testing learning rate={lr_rate}...")
        model = bm.build_dual_path_cnn(num_classes=10, learning_rate=lr_rate)


        h = model.fit(
            {"img_in": X_train},
            {"out_1": y1_train, "out_2": y2_train, "out_3": y3_train},
            validation_data=(
                {"img_in": X_val},
                {"out_1": y1_val, "out_2": y2_val, "out_3": y3_val}
            ),
            epochs=2,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        # Get average accuracy across all 3 outputs
        val_acc = np.mean([
            h.history.get('out_1_accuracy', h.history.get('val_out_1_accuracy', [0]))[-1],
            h.history.get('out_2_accuracy', h.history.get('val_out_2_accuracy', [0]))[-1],
            h.history.get('out_3_accuracy', h.history.get('val_out_3_accuracy', [0]))[-1]
        ])
        cnn_tune.append({'param': lr_rate, 'val_accuracy': val_acc})
        print(f"    Validation Accuracy: {val_acc:.4f}")

    best_lr = max(cnn_tune, key=lambda x: x['val_accuracy'])['param']
    print(f"  Best learning rate: {best_lr}")

    # 2. Build and train the split/Siamese CNN model
    print("\nBuilding and training Split CNN model...")

    split_model = bm.build_dual_path_cnn(num_classes=10, learning_rate=0.001)

    print("\nModel Summary:")
    split_model.summary()

    print(f"\nTraining for {EPOCHS} epochs...")
    print("Each output predicts one digit (0-9)")
    results_split, y_true_split, y_pred_split = bm.run_benchmark_split_cnn(
        split_model, train_ds_split, val_ds_split, test_ds_split, EPOCHS
    )


    print("SPLIT MODEL RESULTS")

    print(f"Test Accuracy (3-digit combination): {results_split['accuracy']:.4f}")
    print(f"Test F1 Score (macro): {results_split['f1_score']:.4f}")
    print(f"Training Time: {results_split['training_time']:.2f}s")


    # 3. Compare with baseline if available
    if res_baseline:

        print("MODEL COMPARISON (Task 2 vs Task 4)")

        comparison_df = pd.DataFrame(
            [res_baseline, results_split],
            index=['Task 2: Standard CNN', 'Task 4: Split CNN']
        )
        print(comparison_df.to_string())


        improvement = (results_split['accuracy'] - res_baseline['accuracy']) * 100
        if improvement > 0:
            print(f"\nSplit CNN improved accuracy by {improvement:.2f} percentage points")
        elif improvement < 0:
            print(f"\nSplit CNN decreased accuracy by {abs(improvement):.2f} percentage points")
        else:
            print(f"\nBoth models achieved the same accuracy")



    # Save the split model
    print("\nSaving split model...")
    split_model.save('best_split_model.keras')
    print("Saved: best_split_model.keras")

    # Generate visualizations
    print("\nGenerating visualizations...")
    if res_baseline and baseline_history:
        print("Creating comparison plots between baseline and split models...")
        vs.plot_split_model_comparison(
            results_split,
            res_baseline,
            baseline_history,
            y_true_split,
            y_pred_split,
            y_true_baseline,
            y_pred_baseline
        )
    else:
        print("Baseline model not available for comparison plots.")
        print("Only split model results will be visualized.")


if __name__ == "__main__":
    main()