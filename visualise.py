

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns



# TASK 5B: GAN VISUALIZATIONS


def plot_gan_training_progress(d_losses, g_losses, d_accuracies, save_path='gan_training_curves.png'):
    """
    Visualize GAN training progress with loss curves and metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(d_losses) + 1)

    # 1. Loss Curves
    axes[0].plot(epochs, d_losses, 'b-', label='Discriminator', linewidth=2, alpha=0.8)
    axes[0].plot(epochs, g_losses, 'r-', label='Generator', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('GAN Training Losses', fontsize=14, weight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 2. Discriminator Accuracy
    axes[1].plot(epochs, d_accuracies, 'g-', linewidth=2, alpha=0.8)
    axes[1].axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Ideal (0.5)')
    axes[1].fill_between(epochs, 0.4, 0.6, alpha=0.2, color='green', label='Healthy Range')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Discriminator Accuracy', fontsize=14, weight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 3. Loss Ratio (G/D)
    loss_ratio = np.array(g_losses) / (np.array(d_losses) + 1e-8)
    axes[2].plot(epochs, loss_ratio, 'purple', linewidth=2, alpha=0.8)
    axes[2].axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Balanced (1.0)')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('G_loss / D_loss', fontsize=12)
    axes[2].set_title('Loss Ratio (Training Balance)', fontsize=14, weight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"GAN training curves saved as '{save_path}'")
    plt.close()


def plot_generated_samples(images, title='Generated Samples', save_path='generated_samples.png',
                           grid_size=(4, 4)):
    """Display a grid of generated images."""
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i]
            if len(img.shape) == 3:
                img = img[:, :, 0]
            ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.suptitle(title, fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Generated samples saved as '{save_path}'")
    plt.close()


def plot_real_vs_synthetic_comparison(real_images, synthetic_images,
                                      save_path='real_vs_synthetic.png', num_samples=8):
    """Side-by-side comparison of real and synthetic images."""
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 5))

    for i in range(num_samples):
        img = real_images[i]
        if len(img.shape) == 3:
            img = img[:, :, 0]
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Real', fontsize=12, weight='bold')

    for i in range(num_samples):
        img = synthetic_images[i]
        if len(img.shape) == 3:
            img = img[:, :, 0]
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Synthetic', fontsize=12, weight='bold')

    plt.suptitle('Real vs Synthetic Images Comparison', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Comparison saved as '{save_path}'")
    plt.close()


def plot_synthetic_image_grid(images, labels=None, save_path='synthetic_grid.png', grid_size=(5, 5)):
    """Display synthetic images with optional pseudo-labels."""
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i]
            if len(img.shape) == 3:
                img = img[:, :, 0]
            ax.imshow(img, cmap='gray')
            if labels is not None:
                d1, d2, d3 = labels
                ax.set_title(f"{d1[i]}{d2[i]}{d3[i]}", fontsize=10)
        ax.axis('off')

    title = 'Synthetic Images with Pseudo-Labels' if labels else 'Synthetic Images'
    plt.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Synthetic grid saved as '{save_path}'")
    plt.close()


def plot_augmentation_impact(results_baseline, results_augmented, save_path='augmentation_impact.png'):
    """Visualize the impact of GAN augmentation on model performance."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Accuracy Comparison
    ax1 = axes[0, 0]
    models = ['Baseline', 'Augmented']
    accuracies = [results_baseline['accuracy'], results_augmented['accuracy']]
    colors = ['#3498db', '#2ecc71']
    bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Overall Accuracy Comparison', fontsize=13, weight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=11, weight='bold')

    improvement = (results_augmented['accuracy'] - results_baseline['accuracy']) * 100
    ax1.annotate(f'{improvement:+.2f}%', xy=(1, results_augmented['accuracy']),
                 xytext=(1.3, results_augmented['accuracy']),
                 fontsize=12, color='green' if improvement > 0 else 'red', weight='bold')

    # 2. F1 Score Comparison
    ax2 = axes[0, 1]
    f1_scores = [results_baseline['f1_score'], results_augmented['f1_score']]
    bars = ax2.bar(models, f1_scores, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score Comparison', fontsize=13, weight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=11, weight='bold')

    # 3. Per-Digit Accuracy
    ax3 = axes[0, 2]
    x = np.arange(3)
    width = 0.35
    baseline_per_digit = results_baseline['per_digit_accuracy']
    augmented_per_digit = results_augmented['per_digit_accuracy']
    bars1 = ax3.bar(x - width / 2, baseline_per_digit, width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x + width / 2, augmented_per_digit, width, label='Augmented', color='#2ecc71', alpha=0.8)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Per-Digit Accuracy', fontsize=13, weight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Digit 1', 'Digit 2', 'Digit 3'])
    ax3.legend()
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)

    # 4. Augmented Model Loss
    ax4 = axes[1, 0]
    if 'history' in results_augmented:
        history = results_augmented['history']
        epochs = range(1, len(history['loss']) + 1)
        ax4.plot(epochs, history['loss'], 'b-o', label='Train Loss', linewidth=2)
        ax4.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Loss', fontsize=12)
        ax4.set_title('Augmented Model Loss', fontsize=13, weight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Data Composition
    ax5 = axes[1, 1]
    if 'num_synthetic_used' in results_augmented:
        real_samples = 60000
        synthetic_samples = results_augmented['num_synthetic_used']
        sizes = [real_samples, synthetic_samples]
        labels_pie = [f'Real\n({real_samples:,})', f'Synthetic\n({synthetic_samples:,})']
        colors_pie = ['#3498db', '#2ecc71']
        ax5.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Training Data Composition', fontsize=13, weight='bold')

    # 6. Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_data = [
        ['Metric', 'Baseline', 'Augmented', 'Change'],
        ['Accuracy', f"{results_baseline['accuracy']:.4f}",
         f"{results_augmented['accuracy']:.4f}",
         f"{(results_augmented['accuracy'] - results_baseline['accuracy']) * 100:+.2f}%"],
        ['F1 Score', f"{results_baseline['f1_score']:.4f}",
         f"{results_augmented['f1_score']:.4f}",
         f"{(results_augmented['f1_score'] - results_baseline['f1_score']) * 100:+.2f}%"],
    ]
    if 'num_synthetic_used' in results_augmented:
        summary_data.append(['Synthetic', '-', f"{results_augmented['num_synthetic_used']:,}", '-'])

    table = ax6.table(cellText=summary_data[1:], colLabels=summary_data[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    ax6.set_title('Summary', fontsize=13, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Augmentation impact saved as '{save_path}'")
    plt.close()


def plot_task5_complete_summary(results_multilabel, results_augmented, gan_history,
                                synthetic_images, real_images, save_path='task5_complete_summary.png'):
    """Create a comprehensive summary visualization for Task 5."""
    fig = plt.figure(figsize=(24, 16))
    d_losses, g_losses, d_accuracies = gan_history

    # Row 1: GAN Training
    ax1 = plt.subplot(3, 4, 1)
    epochs = range(1, len(d_losses) + 1)
    ax1.plot(epochs, d_losses, 'b-', label='D', linewidth=2)
    ax1.plot(epochs, g_losses, 'r-', label='G', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('GAN Losses', weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(epochs, d_accuracies, 'g-', linewidth=2)
    ax2.axhline(y=0.5, color='r', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('D Accuracy', weight='bold')
    ax2.grid(True, alpha=0.3)

    # Real vs Synthetic
    ax3 = plt.subplot(3, 4, 3)
    comparison = np.zeros((84 * 2, 84 * 4))
    for i in range(min(4, len(real_images), len(synthetic_images))):
        r_img = real_images[i][:, :, 0] if len(real_images[i].shape) == 3 else real_images[i]
        s_img = synthetic_images[i][:, :, 0] if len(synthetic_images[i].shape) == 3 else synthetic_images[i]
        comparison[0:84, i * 84:(i + 1) * 84] = r_img
        comparison[84:168, i * 84:(i + 1) * 84] = s_img
    ax3.imshow(comparison, cmap='gray')
    ax3.set_title('Real (top) vs Synthetic', weight='bold')
    ax3.axis('off')

    ax4 = plt.subplot(3, 4, 4)
    syn_grid = np.zeros((84 * 2, 84 * 4))
    for i in range(min(8, len(synthetic_images))):
        row, col = i // 4, i % 4
        img = synthetic_images[i][:, :, 0] if len(synthetic_images[i].shape) == 3 else synthetic_images[i]
        syn_grid[row * 84:(row + 1) * 84, col * 84:(col + 1) * 84] = img
    ax4.imshow(syn_grid, cmap='gray')
    ax4.set_title('Synthetic Samples', weight='bold')
    ax4.axis('off')

    # Row 2: Performance
    ax5 = plt.subplot(3, 4, 5)
    models = ['Baseline', 'Augmented']
    accs = [results_multilabel['accuracy'], results_augmented['accuracy']]
    colors = ['#3498db', '#2ecc71']
    bars = ax5.bar(models, accs, color=colors)
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Accuracy', weight='bold')
    ax5.set_ylim([0, 1])
    for bar in bars:
        ax5.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01, f'{bar.get_height():.4f}', ha='center',
                 weight='bold')

    ax6 = plt.subplot(3, 4, 6)
    f1s = [results_multilabel['f1_score'], results_augmented['f1_score']]
    bars = ax6.bar(models, f1s, color=colors)
    ax6.set_ylabel('F1')
    ax6.set_title('F1 Score', weight='bold')
    ax6.set_ylim([0, 1])
    for bar in bars:
        ax6.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01, f'{bar.get_height():.4f}', ha='center',
                 weight='bold')

    ax7 = plt.subplot(3, 4, 7)
    x = np.arange(3)
    width = 0.35
    ax7.bar(x - width / 2, results_multilabel['per_digit_accuracy'], width, label='Base', color='#3498db')
    ax7.bar(x + width / 2, results_augmented['per_digit_accuracy'], width, label='Aug', color='#2ecc71')
    ax7.set_xticks(x)
    ax7.set_xticklabels(['D1', 'D2', 'D3'])
    ax7.set_title('Per-Digit Acc', weight='bold')
    ax7.legend()
    ax7.set_ylim([0, 1])

    ax8 = plt.subplot(3, 4, 8)
    if 'num_synthetic_used' in results_augmented:
        ax8.pie([60000, results_augmented['num_synthetic_used']],
                labels=['Real', 'Synthetic'], colors=colors, autopct='%1.1f%%')
        ax8.set_title('Data Mix', weight='bold')

    # Row 3: Training curves
    ax9 = plt.subplot(3, 4, 9)
    h = results_multilabel['history']
    ax9.plot(h['loss'], 'b-', label='Train')
    ax9.plot(h['val_loss'], 'r--', label='Val')
    ax9.set_title('Baseline Loss', weight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    ax10 = plt.subplot(3, 4, 10)
    h = results_augmented['history']
    ax10.plot(h['loss'], 'b-', label='Train')
    ax10.plot(h['val_loss'], 'r--', label='Val')
    ax10.set_title('Augmented Loss', weight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # Summary
    ax11 = plt.subplot(3, 4, (11, 12))
    ax11.axis('off')
    acc_chg = (results_augmented['accuracy'] - results_multilabel['accuracy']) * 100
    f1_chg = (results_augmented['f1_score'] - results_multilabel['f1_score']) * 100
    summary = f"""
    ══════════════════════════════════════════════════════
                      TASK 5 SUMMARY
    ══════════════════════════════════════════════════════

    BASELINE MODEL
      Accuracy:  {results_multilabel['accuracy']:.4f}
      F1 Score:  {results_multilabel['f1_score']:.4f}

    AUGMENTED MODEL  
      Accuracy:  {results_augmented['accuracy']:.4f}  ({acc_chg:+.2f}%)
      F1 Score:  {results_augmented['f1_score']:.4f}  ({f1_chg:+.2f}%)
      Synthetic: {results_augmented.get('num_synthetic_used', 'N/A'):,}

    RESULT: {'Augmentation improved performance!' if acc_chg > 0 else ' No improvement'}
    ══════════════════════════════════════════════════════
    """
    ax11.text(0.1, 0.5, summary, fontsize=12, fontfamily='monospace', verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Task 5: GAN-Based Data Augmentation Analysis', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Complete summary saved as '{save_path}'")
    plt.close()




def plot_benchmark_results(df, cnn_history=None, lr_tuning_history=None, y_true=None, y_pred=None):
    """Creates a comprehensive visualization comparing Logistic Regression and CNN."""
    fig = plt.figure(figsize=(20, 12))

    if cnn_history and 'loss' in cnn_history:
        ax1 = plt.subplot(2, 3, 1)
        epochs = range(1, len(cnn_history['loss']) + 1)
        ax1.plot(epochs, cnn_history['loss'], 'b-o', label='Training Loss', linewidth=2)
        ax1.plot(epochs, cnn_history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('CNN Loss Curves', fontsize=13, weight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    models = df.index.tolist()
    accuracies = df['accuracy'].tolist()
    colors = ['skyblue', 'lightcoral']
    bars = ax2.bar(models, accuracies, color=colors, edgecolor='navy', alpha=0.7)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Model Accuracy Comparison', fontsize=13, weight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    ax3 = plt.subplot(2, 3, 3)
    f1_scores = df['f1_score'].tolist()
    bars = ax3.bar(models, f1_scores, color=colors, edgecolor='darkred', alpha=0.7)
    ax3.set_ylabel('F1 Score', fontsize=11)
    ax3.set_title('Model F1 Score Comparison', fontsize=13, weight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    if cnn_history:
        ax4 = plt.subplot(2, 3, 4)
        epochs = range(1, len(cnn_history['accuracy']) + 1)
        ax4.plot(epochs, cnn_history['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
        ax4.plot(epochs, cnn_history['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Accuracy', fontsize=11)
        ax4.set_title('CNN Accuracy Curves', fontsize=13, weight='bold')
        ax4.legend(loc='lower right')
        ax4.grid(True, alpha=0.3)

    if y_true is not None and y_pred is not None:
        ax5 = plt.subplot(2, 3, 5)
        cm = confusion_matrix(y_true, y_pred)
        im = ax5.imshow(cm, cmap='Blues', interpolation='nearest', aspect='auto')
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        ax5.set_xlabel('Predicted Label', fontsize=11)
        ax5.set_ylabel('True Label', fontsize=11)
        ax5.set_title('Confusion Matrix (1000x1000)', fontsize=13, weight='bold')
        tick_positions = np.arange(0, 1000, 100)
        ax5.set_xticks(tick_positions)
        ax5.set_yticks(tick_positions)

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    numeric_cols = ['accuracy', 'f1_score', 'training_time']
    df_display = df[numeric_cols].copy()
    table_data = [[f'{df_display.loc[idx, col]:.4f}' if col != 'training_time' else f'{df_display.loc[idx, col]:.2f}s'
                   for col in df_display.columns] for idx in df_display.index]
    table = ax6.table(cellText=table_data, colLabels=df_display.columns, rowLabels=df_display.index, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    ax6.set_title('Detailed Model Comparison', pad=20, fontsize=13, weight='bold')

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'benchmark_results.png'")
    plt.close()


def plot_task5_multilabel_results(results_multilabel, baseline_history=None):
    """Visualize Task 5a Multi-label CNN results."""
    fig = plt.figure(figsize=(16, 10))
    history = results_multilabel['history']

    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(history['out_1_accuracy']) + 1)
    ax1.plot(epochs, history['out_1_accuracy'], 'b-o', label='Digit 1', linewidth=2)
    ax1.plot(epochs, history['out_2_accuracy'], 'g-s', label='Digit 2', linewidth=2)
    ax1.plot(epochs, history['out_3_accuracy'], 'r-^', label='Digit 3', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy')
    ax1.set_title('Per-Digit Training Accuracy', weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['val_out_1_accuracy'], 'b-o', label='Digit 1', linewidth=2)
    ax2.plot(epochs, history['val_out_2_accuracy'], 'g-s', label='Digit 2', linewidth=2)
    ax2.plot(epochs, history['val_out_3_accuracy'], 'r-^', label='Digit 3', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Per-Digit Validation Accuracy', weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['loss'], 'b-o', label='Training', linewidth=2)
    ax3.plot(epochs, history['val_loss'], 'r-s', label='Validation', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Combined Loss', weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Only plot confusion matrix if y_true and y_pred are available
    ax4 = plt.subplot(2, 3, 4)
    if 'y_true' in results_multilabel and 'y_pred' in results_multilabel:
        cm = confusion_matrix(results_multilabel['y_true'], results_multilabel['y_pred'])
        im = ax4.imshow(cm, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax4, fraction=0.046)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')
        ax4.set_title('Confusion Matrix', weight='bold')
    else:
        # Show per-digit accuracy bars instead
        digits = ['Digit 1', 'Digit 2', 'Digit 3']
        accs = results_multilabel['per_digit_accuracy']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax4.bar(digits, accs, color=colors, edgecolor='black')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Per-Digit Test Accuracy', weight='bold')
        ax4.set_ylim([0, 1])
        for bar in bars:
            ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{bar.get_height():.4f}', ha='center', fontsize=10, weight='bold')
        ax4.grid(axis='y', alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')

    # Get per-digit accuracy safely
    per_digit = results_multilabel.get('per_digit_accuracy', [0, 0, 0])
    if isinstance(per_digit, dict):
        d1, d2, d3 = per_digit.get('digit_1', 0), per_digit.get('digit_2', 0), per_digit.get('digit_3', 0)
    else:
        d1, d2, d3 = per_digit[0], per_digit[1], per_digit[2]

    summary = f"""
    MULTI-LABEL CNN RESULTS

    Test Accuracy: {results_multilabel['accuracy']:.4f}
    Test F1 Score: {results_multilabel['f1_score']:.4f}
    Training Time: {results_multilabel['training_time']:.1f}s

    Per-Digit Test Accuracy:
      Digit 1: {d1:.4f}
      Digit 2: {d2:.4f}
      Digit 3: {d3:.4f}

    Final Val Accuracy:
      Digit 1: {history['val_out_1_accuracy'][-1]:.4f}
      Digit 2: {history['val_out_2_accuracy'][-1]:.4f}
      Digit 3: {history['val_out_3_accuracy'][-1]:.4f}
    """
    ax5.text(0.1, 0.5, summary, fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Add a 6th subplot with training info
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    info = f"""
    TRAINING CONFIGURATION

    Epochs completed: {len(history['loss'])}
    Final train loss: {history['loss'][-1]:.4f}
    Final val loss:   {history['val_loss'][-1]:.4f}

    Best validation accuracy achieved
    across all three digit outputs.
    """
    ax6.text(0.1, 0.5, info, fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.savefig('task5_multilabel_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'task5_multilabel_results.png'")
    plt.close()