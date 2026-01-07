

import numpy as np
import tensorflow as tf
import Builder as bm
import pre_processing as pp
import visualise as vis
import os


# GPU SETUP

devices = tf.config.list_physical_devices()
print(f"Total available devices: {devices}")

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpu_devices[0], 'GPU')
        print(f"SUCCESS: GPU ('{gpu_devices[0].name}') is active.")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")
else:
    print("GPU NOT found. Using CPU.")

tf.config.set_soft_device_placement(True)


# CONFIGURATION


IMG_SIZE = (84, 84)
QUICK_TEST = False  # Set to False for full training

if QUICK_TEST:

    print("QUICK TEST MODE - Reduced parameters for testing")

    BATCH_SIZE = 64
    EPOCHS_MULTILABEL = 5
    EPOCHS_GAN = 20          # Reduced for testing
    LATENT_DIM = 128
    NUM_SYNTHETIC = 5000
    GAN_SUBSET_SIZE = 20000  # Use subset for quick testing
else:

    print("FULL TRAINING MODE")

    BATCH_SIZE = 128
    EPOCHS_MULTILABEL = 15
    EPOCHS_GAN = 50          # Full training
    LATENT_DIM = 128
    NUM_SYNTHETIC = 10000
    GAN_SUBSET_SIZE = None   # Use all data



def main():



    # PHASE 1: Multi-label CNN Training


    print(" Multi-label CNN Training")


    print("\nLoading datasets...")
    train_ds = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=True)
    val_ds = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=True)
    test_ds = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=True)

    results_multilabel = bm.train_multilabel_cnn(
        train_ds, val_ds, test_ds,
        epochs=EPOCHS_MULTILABEL,
        quick_test=QUICK_TEST
    )


    print("Multi-lable CNN Results")
    print(f"Test Accuracy: {results_multilabel['accuracy']:.4f}")
    print(f"Test F1 Score: {results_multilabel['f1_score']:.4f}")
    print(f"Per-digit:     D1={results_multilabel['per_digit_accuracy'][0]:.4f}, "
          f"D2={results_multilabel['per_digit_accuracy'][1]:.4f}, "
          f"D3={results_multilabel['per_digit_accuracy'][2]:.4f}")
    print(f"Training Time: {results_multilabel['training_time']:.2f}s")

    # Save baseline model (needed for pseudo-labeling)
    print("\nSaving baseline model...")
    results_multilabel['model'].save('multilabel_cnn.keras')
    print("Model saved as 'multilabel_cnn.keras'")

    # Visualize baseline results
    vis.plot_task5_multilabel_results(results_multilabel)


    # PHASE 2: DCGAN Training on entire Dataset


    print(" DCGAN TRAINING")
    print("\nLoading ALL datasets for GAN training...")

    # Load each split with GAN mode (normalizes to [-1, 1])
    train_ds_gan = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=False, gan_mode=True)
    val_ds_gan = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=False, gan_mode=True)
    test_ds_gan = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=False, gan_mode=True)

    # Extract numpy arrays
    X_train, _ = pp.split_dataset(train_ds_gan)
    X_val, _ = pp.split_dataset(val_ds_gan)
    X_test, _ = pp.split_dataset(test_ds_gan)

    # Combine ALL data for GAN training
    X_all_gan = np.concatenate([X_train, X_val, X_test], axis=0)
    print(f"\nCombined dataset sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    print(f"  TOTAL: {len(X_all_gan)} images for GAN training")

    # Use subset if in quick test mode
    if GAN_SUBSET_SIZE is not None and len(X_all_gan) > GAN_SUBSET_SIZE:
        print(f"\nUsing subset: {GAN_SUBSET_SIZE} of {len(X_all_gan)} images")
        indices = np.random.choice(len(X_all_gan), GAN_SUBSET_SIZE, replace=False)
        X_all_gan = X_all_gan[indices]

    print(f"\nFinal GAN training set: {len(X_all_gan)} images")
    print(f"Epochs: {EPOCHS_GAN}")
    print(f"Latent dimension: {LATENT_DIM}")

    # Train GAN
    gan, d_losses, g_losses, d_accuracies = bm.train_gan(
        X_all_gan,
        epochs=EPOCHS_GAN,
        batch_size=BATCH_SIZE,
        latent_dim=LATENT_DIM,
        save_interval=10
    )

    # Visualize GAN training progress
    vis.plot_gan_training_progress(d_losses, g_losses, d_accuracies,
                                   save_path='gan_training_curves.png')

    # Quality assessment
    print("GAN QUALITY ASSESSMENT")


    generated_samples = gan.generate_images(100)
    real_samples = (X_train[:100] + 1) / 2  # Convert back to [0,1] for comparison

    print(f"Generated images - Mean: {np.mean(generated_samples):.4f}, Std: {np.std(generated_samples):.4f}")
    print(f"Real images      - Mean: {np.mean(real_samples):.4f}, Std: {np.std(real_samples):.4f}")

    # Check for mode collapse
    collapse_check = gan.check_mode_collapse()
    print(f"Diversity (pairwise distance): {collapse_check['mean_pairwise_distance']:.2f}")
    print(f"Mode collapse detected: {collapse_check['is_collapsed']}")

    # Save generator
    gan.generator.save('generator.keras')
    print("Generator saved as 'generator.keras'")


    # PHASE 3: Generate and Visualize Synthetic Images


    print("PHASE 3: SYNTHETIC IMAGE GENERATION & VISUALIZATION")


    # Generate synthetic images for visualization
    print(f"\nGenerating {NUM_SYNTHETIC} synthetic images...")
    synthetic_images = gan.generate_images(NUM_SYNTHETIC)
    print(f"Generated shape: {synthetic_images.shape}")

    # Visualize generated samples
    vis.plot_generated_samples(
        synthetic_images[:16],
        title='GAN Generated Synthetic Images',
        save_path='synthetic_samples.png',
        grid_size=(4, 4)
    )

    # Real vs Synthetic comparison
    real_for_comparison = (X_train[:8] + 1) / 2  # Convert to [0,1]
    vis.plot_real_vs_synthetic_comparison(
        real_for_comparison,
        synthetic_images[:8],
        save_path='real_vs_synthetic_comparison.png',
        num_samples=8
    )

    # Generate more samples with pseudo-labels for visualization
    print("\nGenerating pseudo-labels for visualization...")
    labeling_model = tf.keras.models.load_model('multilabel_cnn.keras')
    sample_preds = labeling_model.predict({'img_in': synthetic_images[:25]}, verbose=0)
    p1 = np.argmax(sample_preds[0], axis=1)
    p2 = np.argmax(sample_preds[1], axis=1)
    p3 = np.argmax(sample_preds[2], axis=1)

    vis.plot_synthetic_image_grid(
        synthetic_images[:25],
        labels=(p1, p2, p3),
        save_path='synthetic_with_labels.png',
        grid_size=(5, 5)
    )


    # PHASE 4: Augmented Training


    print("PHASE 4: TRAINING WITH AUGMENTED DATA")


    # Get real training data (non-GAN normalized)
    train_ds_real = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=False)
    X_train_real, y_train_real = pp.split_dataset(train_ds_real)

    print(f"\nReal training data: {len(X_train_real)} images")
    print(f"Synthetic to generate: {NUM_SYNTHETIC}")

    # Train with augmented data
    results_augmented = bm.train_with_augmented_data(
        gan,
        train_ds,
        val_ds,
        test_ds,
        X_train_real,
        y_train_real,
        num_synthetic=NUM_SYNTHETIC,
        epochs=EPOCHS_MULTILABEL,
        batch_size=BATCH_SIZE,
        confidence_threshold=0.8
    )


    print("AUGMENTED MODEL RESULTS")

    print(f"Test Accuracy: {results_augmented['accuracy']:.4f}")
    print(f"Test F1 Score: {results_augmented['f1_score']:.4f}")
    print(f"Per-digit:     D1={results_augmented['per_digit_accuracy'][0]:.4f}, "
          f"D2={results_augmented['per_digit_accuracy'][1]:.4f}, "
          f"D3={results_augmented['per_digit_accuracy'][2]:.4f}")
    print(f"Training Time: {results_augmented['training_time']:.2f}s")
    print(f"Synthetic samples used: {results_augmented['num_synthetic_used']}")

    # Save augmented model
    results_augmented['model'].save('multilabel_cnn_augmented.keras')
    print("Augmented model saved as 'multilabel_cnn_augmented.keras'")


    # PHASE 5: Performance Comparison & Visualization


    print("PHASE 5: PERFORMANCE COMPARISON")


    # Visualize augmentation impact
    vis.plot_augmentation_impact(
        results_multilabel,
        results_augmented,
        save_path='augmentation_impact.png'
    )

    # Create complete summary visualization
    vis.plot_task5_complete_summary(
        results_multilabel,
        results_augmented,
        gan_history=(d_losses, g_losses, d_accuracies),
        synthetic_images=synthetic_images[:16],
        real_images=real_for_comparison[:8],
        save_path='task5_complete_summary.png'
    )


    # FINAL SUMMARY


    print("FINAL COMPARISON SUMMARY")


    print(f"\n{'Metric':<25} {'Baseline':<15} {'Augmented':<15} {'Change':<15}")


    acc_change = (results_augmented['accuracy'] - results_multilabel['accuracy']) * 100
    f1_change = (results_augmented['f1_score'] - results_multilabel['f1_score']) * 100

    print(f"{'Test Accuracy':<25} {results_multilabel['accuracy']:<15.4f} "
          f"{results_augmented['accuracy']:<15.4f} {acc_change:+.2f}%")
    print(f"{'Test F1 Score':<25} {results_multilabel['f1_score']:<15.4f} "
          f"{results_augmented['f1_score']:<15.4f} {f1_change:+.2f}%")

    for i, digit in enumerate(['Digit 1', 'Digit 2', 'Digit 3']):
        base_acc = results_multilabel['per_digit_accuracy'][i]
        aug_acc = results_augmented['per_digit_accuracy'][i]
        change = (aug_acc - base_acc) * 100
        print(f"{digit + ' Accuracy':<25} {base_acc:<15.4f} {aug_acc:<15.4f} {change:+.2f}%")


    print(f"{'Synthetic samples used':<25} {'-':<15} {results_augmented['num_synthetic_used']:<15,}")
    print(f"{'Training time (s)':<25} {results_multilabel['training_time']:<15.1f} "
          f"{results_augmented['training_time']:<15.1f}")


    if acc_change > 0:
        print(f"GAN augmentation IMPROVED accuracy by {acc_change:.2f} percentage points")
    elif acc_change == 0:
        print("GAN augmentation had NO EFFECT on accuracy")
    else:
        print(f"GAN augmentation DECREASED accuracy by {abs(acc_change):.2f} percentage points")




    return results_multilabel, results_augmented, gan


if __name__ == "__main__":
    results_baseline, results_augmented, trained_gan = main()