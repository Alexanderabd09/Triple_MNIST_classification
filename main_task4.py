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
    """
    Main script for Task 4: Split/Siamese CNN implementation.
    Splits each 84x84 image into three 84x28 strips (one per digit).
    Trains a shared CNN to predict each digit independently.
    """
    # Configuration
    IMG_SIZE = (84, 84)
    BATCH_SIZE = 128
    EPOCHS = 15

    # For testing/debugging, set to True to use fewer epochs
    QUICK_TEST = False  # Set to True for quick testing (5 epochs instead of 15)
    if QUICK_TEST:
        EPOCHS = 5
        print("\n⚠️  QUICK TEST MODE: Using only 5 epochs")

    print("=" * 60)
    print("TASK 4: Split/Siamese CNN Architecture")
    print("=" * 60)

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
    print("\n[1/3] Loading datasets for split model...")
    print("Each 84x84 image will be split into three 84x28 strips (left, middle, right)")

    train_ds_split = pp.prepare_dataset(
        "triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=True
    )
    val_ds_split = pp.prepare_dataset(
        "triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=True
    )
    test_ds_split = pp.prepare_dataset(
        "triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=True
    )
    print("Datasets loaded and formatted for split model.")

    # 2. Build and train the split/Siamese CNN model
    print("\n[2/3] Building and training Split CNN model...")
    print("Architecture: Siamese network with shared weights for each digit strip")
    split_model = bm.build_dual_path_cnn(num_classes=10, learning_rate=0.001)

    print("\nModel Summary:")
    split_model.summary()

    print(f"\nTraining for {EPOCHS} epochs...")
    print("Each output predicts one digit (0-9)")
    results_split, y_true_split, y_pred_split = bm.run_benchmark_split_cnn(
        split_model, train_ds_split, val_ds_split, test_ds_split, EPOCHS
    )

    print("\n" + "=" * 60)
    print("SPLIT MODEL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy (3-digit combination): {results_split['accuracy']:.4f}")
    print(f"Test F1 Score (macro): {results_split['f1_score']:.4f}")
    print(f"Training Time: {results_split['training_time']:.2f}s")
    print("=" * 60)

    # 3. Compare with baseline if available
    if res_baseline:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON (Task 2 vs Task 4)")
        print("=" * 60)
        comparison_df = pd.DataFrame(
            [res_baseline, results_split],
            index=['Task 2: Standard CNN', 'Task 4: Split CNN']
        )
        print(comparison_df.to_string())
        print("=" * 60)

        improvement = (results_split['accuracy'] - res_baseline['accuracy']) * 100
        if improvement > 0:
            print(f"\n✓ Split CNN improved accuracy by {improvement:.2f} percentage points")
        elif improvement < 0:
            print(f"\n✗ Split CNN decreased accuracy by {abs(improvement):.2f} percentage points")
        else:
            print(f"\n= Both models achieved the same accuracy")

        print("\nKey Insight: The split approach treats each digit independently,")
        print("which can be more efficient than predicting all 1000 combinations directly.")

    # Save the split model
    print("\n[3/3] Saving split model...")
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