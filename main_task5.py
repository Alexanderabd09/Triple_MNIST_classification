"""
Task 5: Multi-label CNN and GAN-based Data Augmentation
Main execution script - calls functions from Builder.py
"""

import numpy as np
import tensorflow as tf
import Builder as bm
import pre_processing as pp


devices = tf.config.list_physical_devices()
print(f"Total available devices: {devices}")


gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    try:

        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Explicitly set the default device to GPU
        tf.config.set_visible_devices(gpu_devices[0], 'GPU')
        print(f"SUCCESS: M2 GPU ('{gpu_devices[0].name}') is active.")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")
else:
    print(" GPU NOT found. Using CPU.")

tf.config.set_soft_device_placement(True)

# CONFIGURATION

IMG_SIZE = (84, 84)
BATCH_SIZE = 128
EPOCHS_MULTILABEL = 15
EPOCHS_GAN = 50
LATENT_DIM = 100

# For quick testing
QUICK_TEST = False
if QUICK_TEST:
    EPOCHS_MULTILABEL = 5
    EPOCHS_GAN = 10
    print("\nTesting mode- ON")



def main():

    print("\nLoading datasets for multi-label training...")
    train_ds = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=True)
    val_ds = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=True)
    test_ds = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=True)

    results_multilabel = bm.train_multilabel_cnn(train_ds, val_ds, test_ds, epochs=EPOCHS_MULTILABEL)


    print("MULTI-LABEL CNN RESULTS")
    print(f"Test Accuracy: {results_multilabel['accuracy']:.4f}")
    print(f"Test F1 Score: {results_multilabel['f1_score']:.4f}")
    print(f"Training Time: {results_multilabel['training_time']:.2f}s")


    print("\n Saving model...")
    results_multilabel['model'].save('multilabel_cnn.keras')
    print(" Model saved as 'multilabel_cnn.keras'")




    print("DCGAN TRAINING FOR SYNTHETIC IMAGE GENERATION")


    train_ds_gan = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=False)
    val_ds_gan = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=False)
    test_ds_gan = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=False)

    X_train, _ = pp.split_dataset(train_ds_gan)
    X_val, _ = pp.split_dataset(val_ds_gan)
    X_test, _ = pp.split_dataset(test_ds_gan)

    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    X_all = (X_all * 2.0) - 1.0  # Normalize to [-1, 1] for tanh

    print(f" Total images for GAN: {len(X_all)}")
    print(f"\nThis will take approximately {EPOCHS_GAN * len(X_all) / (BATCH_SIZE * 3000):.0f}-{EPOCHS_GAN * len(X_all) / (BATCH_SIZE * 2000):.0f} minutes...")

    gan, d_losses, g_losses, d_accuracies = bm.train_gan(
        X_all,
        epochs=EPOCHS_GAN,
        batch_size=BATCH_SIZE,
        latent_dim=LATENT_DIM
    )

    generated_samples = gan.generate_images(100)

    print("\nQuality Metrics:")
    print(f"  Generated mean: {np.mean(generated_samples):.4f}")
    print(f"  Generated std:  {np.std(generated_samples):.4f}")
    print(f"  Real mean:      {np.mean((X_all + 1) / 2):.4f}")
    print(f"  Real std:       {np.std((X_all + 1) / 2):.4f}")

    print("\n✓ Generated samples saved as 'generated_samples.png'")
    print("✓ Training curves saved as 'gan_training_curves.png'")

    print("\n[4/4] Saving generator...")
    gan.generator.save('generator.keras')
    print("✓ Generator saved as 'generator.keras'")



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
        num_synthetic=10000,
        epochs=EPOCHS_MULTILABEL,
        batch_size=BATCH_SIZE
    )


    print("AUGMENTED MODEL RESULTS")

    print(f"Test Accuracy: {results_augmented['accuracy']:.4f}")
    print(f"Test F1 Score: {results_augmented['f1_score']:.4f}")
    print(f"Training Time: {results_augmented['training_time']:.2f}s")


    results_augmented['model'].save('multilabel_cnn_augmented.keras')
    print("\n✓ Augmented model saved as 'multilabel_cnn_augmented.keras'")



    print(f"\nBaseline Model (Real Data Only):")
    print(f"  Training samples: {len(X_train_real)}")
    print(f"  Test Accuracy:    {results_multilabel['accuracy']:.4f}")
    print(f"  Test F1 Score:    {results_multilabel['f1_score']:.4f}")
    print(f"  Training Time:    {results_multilabel['training_time']/60:.1f} minutes")

    print(f"\nAugmented Model (Real + 10,000 Synthetic):")
    print(f"  Training samples: {len(X_train_real) + 10000}")
    print(f"  Test Accuracy:    {results_augmented['accuracy']:.4f}")
    print(f"  Test F1 Score:    {results_augmented['f1_score']:.4f}")
    print(f"  Training Time:    {results_augmented['training_time']/60:.1f} minutes")

    # Calculate improvements
    acc_improvement = (results_augmented['accuracy'] - results_multilabel['accuracy']) * 100
    f1_improvement = (results_augmented['f1_score'] - results_multilabel['f1_score']) * 100


    print("IMPACT OF GAN AUGMENTATION")

    print(f"Accuracy Improvement: {acc_improvement:+.2f} percentage points")
    print(f"F1 Score Improvement: {f1_improvement:+.2f} percentage points")

    if acc_improvement > 2:
        print("\n✓ SIGNIFICANT IMPROVEMENT from GAN augmentation!")
        print("  Analysis:")
        print("  - Synthetic data successfully enhanced generalization")
        print("  - Model benefits from additional training examples")
        print("  - GAN-generated images are sufficiently realistic")
    elif acc_improvement > 0:
        print("\n✓ MODEST IMPROVEMENT from GAN augmentation")
        print("  Analysis:")
        print("  - Synthetic data provides some benefit")
        print("  - Generated images add value but with limitations")
        print("  - Consider generating more samples or improving GAN quality")
    elif acc_improvement > -1:
        print("\n≈ NEUTRAL IMPACT from GAN augmentation")
        print("  Analysis:")
        print("  - Synthetic data neither helps nor hurts significantly")
        print("  - Possible reasons:")
        print("    * Original dataset already sufficient")
        print("    * Generated images too similar to training data")
        print("    * Pseudo-labeling accuracy cancels out benefits")
    else:
        print("\n✗ NEGATIVE IMPACT from GAN augmentation")
        print("  Analysis:")
        print("  - Synthetic data decreased performance")
        print("  - Possible reasons:")
        print("    * Generated images not realistic enough")
        print("    * High pseudo-labeling error rate")
        print("    * GAN mode collapse (limited variety)")
        print("    * Need longer GAN training or different architecture")

    # ========================================================================
    # SUMMARY FOR REPORT
    # ========================================================================

    print("\n" + "=" * 70)
    print("TASK 5 COMPLETE - SUMMARY FOR REPORT")
    print("=" * 70)

    print("\nGenerated Files:")
    print("  Models:")
    print("    - multilabel_cnn.keras (Part A baseline)")
    print("    - generator.keras (trained GAN generator)")
    print("    - multilabel_cnn_augmented.keras (Part B augmented)")
    print("\n  Visualizations:")
    print("    - generated_samples.png (16 synthetic images)")
    print("    - gan_training_curves.png (GAN loss/accuracy)")

    print("\nKey Results to Report:")
    print(f"  Part A - Multi-label CNN:")
    print(f"    • Architecture: Full image processing with 3 output heads")
    print(f"    • Accuracy: {results_multilabel['accuracy']:.4f}")
    print(f"    • Key insight: Learns implicit spatial attention")

    print(f"\n  Part B - DCGAN:")
    print(f"    • Training epochs: {EPOCHS_GAN}")
    print(f"    • Final D_loss: {d_losses[-1]:.4f}, G_loss: {g_losses[-1]:.4f}")
    print(f"    • Discriminator accuracy: {d_accuracies[-1]:.4f}")
    print(f"    • Image quality: [See generated_samples.png]")

    print(f"\n  Part C - Augmentation Impact:")
    print(f"    • Synthetic images added: 10,000")
    print(f"    • Accuracy change: {acc_improvement:+.2f}%")
    print(f"    • Conclusion: [Based on improvement above]")

    print("\nDiscussion Points for Report:")
    print("  1. Compare multi-label CNN with Task 4 split approach")
    print("  2. Explain GAN architecture choices (DCGAN principles)")
    print("  3. Assess quality of generated images (visual + statistical)")
    print("  4. Analyze impact of augmentation (why it helped/didn't help)")
    print("  5. Discuss trade-offs: training time vs. performance gain")
    print("  6. Suggest improvements (e.g., conditional GAN, more epochs)")

    print("\n" + "=" * 70)
    print("All Task 5 experiments completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()