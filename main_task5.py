"""
Task 5: Multi-label CNN and GAN-based Data Augmentation
"""

import numpy as np
import tensorflow as tf
import Builder as bm
import pre_processing as pp

# GPU Setup
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

# ============================================
# CONFIGURATION
# ============================================

IMG_SIZE = (84, 84)
QUICK_TEST = False  # Set to False for full training

if QUICK_TEST:
    print("\nQUICK TEST MODE")
    BATCH_SIZE = 64
    EPOCHS_MULTILABEL = 5
    EPOCHS_GAN = 15
    LATENT_DIM = 100
    NUM_SYNTHETIC = 5000
    GAN_SUBSET_SIZE = 20000
else:
    print("\nFULL TRAINING MODE")
    BATCH_SIZE = 128
    EPOCHS_MULTILABEL = 15
    EPOCHS_GAN = 100
    LATENT_DIM = 100
    NUM_SYNTHETIC = 10000
    GAN_SUBSET_SIZE = None


def main():

    # =========================================
    # PHASE 1: Multi-label CNN Training
    # =========================================
    print("\n" + "="*60)
    print("PHASE 1: MULTI-LABEL CNN TRAINING")
    print("="*60)

    print("\nLoading datasets...")
    train_ds = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=True)
    val_ds = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=True)
    test_ds = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=True)

    # Pass quick_test flag to limit steps per epoch
    results_multilabel = bm.train_multilabel_cnn(
        train_ds, val_ds, test_ds,
        epochs=EPOCHS_MULTILABEL,
        quick_test=QUICK_TEST
    )

    print("\n" + "-"*40)
    print("MULTI-LABEL CNN RESULTS")
    print("-"*40)
    print(f"Test Accuracy: {results_multilabel['accuracy']:.4f}")
    print(f"Test F1 Score: {results_multilabel['f1_score']:.4f}")
    print(f"Training Time: {results_multilabel['training_time']:.2f}s")

    print("\nSaving model...")
    results_multilabel['model'].save('multilabel_cnn.keras')
    print("✓ Model saved as 'multilabel_cnn.keras'")

    # =========================================
    # PHASE 2: DCGAN Training
    # =========================================
    print("\n" + "="*60)
    print("PHASE 2: DCGAN TRAINING")
    print("="*60)

    # Load data for GAN (WITHOUT split - we need raw images)
    train_ds_gan = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=False)
    X_train_gan, _ = pp.split_dataset(train_ds_gan)

    # Use subset in quick test mode
    if GAN_SUBSET_SIZE is not None and len(X_train_gan) > GAN_SUBSET_SIZE:
        print(f"\nUsing subset: {GAN_SUBSET_SIZE} of {len(X_train_gan)} images")
        indices = np.random.choice(len(X_train_gan), GAN_SUBSET_SIZE, replace=False)
        X_train_gan = X_train_gan[indices]

    # Normalize to [-1, 1] for tanh activation
    X_train_gan = (X_train_gan * 2.0) - 1.0

    print(f"\nTraining images for GAN: {len(X_train_gan)}")
    print(f"Epochs: {EPOCHS_GAN}")

    gan, d_losses, g_losses, d_accuracies = bm.train_gan(
        X_train_gan,
        epochs=EPOCHS_GAN,
        batch_size=BATCH_SIZE,
        latent_dim=LATENT_DIM
    )

    # Quality check
    generated_samples = gan.generate_images(100)
    print("\nQuality Metrics:")
    print(f"  Generated mean: {np.mean(generated_samples):.4f}")
    print(f"  Generated std:  {np.std(generated_samples):.4f}")
    print(f"  Real mean:      {np.mean((X_train_gan + 1) / 2):.4f}")
    print(f"  Real std:       {np.std((X_train_gan + 1) / 2):.4f}")

    gan.generator.save('generator.keras')
    print("✓ Generator saved as 'generator.keras'")

    # =========================================
    # PHASE 3: Augmented Training
    # =========================================
    print("\n" + "="*60)
    print("PHASE 3: TRAINING WITH AUGMENTED DATA")
    print("="*60)

    # Get real training data
    train_ds_real = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=False)
    X_train_real, y_train_real = pp.split_dataset(train_ds_real)

    results_augmented = bm.train_with_augmented_data(
        gan,
        train_ds,
        val_ds,
        test_ds,
        X_train_real,
        y_train_real,
        num_synthetic=NUM_SYNTHETIC,
        epochs=EPOCHS_MULTILABEL,
        batch_size=BATCH_SIZE
    )

    print("\n" + "-"*40)
    print("AUGMENTED MODEL RESULTS")
    print("-"*40)
    print(f"Test Accuracy: {results_augmented['accuracy']:.4f}")
    print(f"Test F1 Score: {results_augmented['f1_score']:.4f}")
    print(f"Training Time: {results_augmented['training_time']:.2f}s")

    results_augmented['model'].save('multilabel_cnn_augmented.keras')
    print("✓ Augmented model saved")

    # =========================================
    # FINAL COMPARISON
    # =========================================
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    print(f"\nBaseline Model:")
    print(f"  Test Accuracy: {results_multilabel['accuracy']:.4f}")
    print(f"  Test F1 Score: {results_multilabel['f1_score']:.4f}")

    print(f"\nAugmented Model (+{NUM_SYNTHETIC} synthetic):")
    print(f"  Test Accuracy: {results_augmented['accuracy']:.4f}")
    print(f"  Test F1 Score: {results_augmented['f1_score']:.4f}")

    acc_diff = (results_augmented['accuracy'] - results_multilabel['accuracy']) * 100
    print(f"\nAccuracy Change: {acc_diff:+.2f} percentage points")

    print("\n" + "="*60)
    print("TASK 5 COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()