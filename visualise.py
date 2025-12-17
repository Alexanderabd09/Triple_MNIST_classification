import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_benchmark_results(df, cnn_history=None, y_true=None, y_pred=None):
    # Expanded figure size for the extra row
    fig = plt.figure(figsize=(16, 15))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    models = df.index

    # --- 1. Grouped Bar Chart ---
    ax1 = plt.subplot(3, 3, 1)
    metrics = ['accuracy', 'f1_score', 'training_time', 'prediction_time']
    x = np.arange(len(metrics))
    width = 0.25
    for i, model_name in enumerate(models):
        vals = [df.loc[model_name, m] for m in metrics]
        ax1.bar(x + (i * width - width / 2), vals, width, label=model_name, alpha=0.8)
    ax1.set_title('Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()

    # --- 2. Accuracy Comparison ---
    ax2 = plt.subplot(3, 3, 2)
    ax2.barh(models, df['accuracy'], color=colors[:len(models)])
    ax2.set_title('Accuracy')
    ax2.set_xlim(0, 1)

    # --- 3. F1 Score ---
    ax3 = plt.subplot(3, 3, 3)
    ax3.barh(models, df['f1_score'], color=colors[:len(models)])
    ax3.set_title('F1 Score')
    ax3.set_xlim(0, 1)

    # --- 4. Training Time ---
    ax4 = plt.subplot(3, 3, 4)
    ax4.bar(models, df['training_time'], color=colors[:len(models)])
    ax4.set_title('Training Time (s)')

    # --- 5. Prediction Time ---
    ax5 = plt.subplot(3, 3, 5)
    ax5.bar(models, df['prediction_time'], color=colors[:len(models)])
    ax5.set_title('Prediction Time (s)')

    # --- 6. CNN Training History ---
    ax6 = plt.subplot(3, 3, 6)
    if cnn_history:
        ax6.plot(cnn_history['accuracy'], label='Train Acc', color='#4ECDC4', linewidth=2)
        if 'val_accuracy' in cnn_history:
            ax6.plot(cnn_history['val_accuracy'], label='Val Acc', color='#FF6B6B', linestyle='--')
        ax6.set_title('Training Progress')
        ax6.set_xlabel('Epochs')
        ax6.legend()

    # --- 7. Confusion Matrix (Span across the bottom row) ---
    if y_true is not None and y_pred is not None:
        ax7 = plt.subplot(3, 1, 3)  # Span the whole bottom row
        cm = confusion_matrix(y_true, y_pred)

        # Normalize the matrix to show percentages (easier to read)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = ax7.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        ax7.set_title('Confusion Matrix (Digit Level)', fontsize=14)
        fig.colorbar(im, ax=ax7)

        tick_marks = np.arange(10)
        ax7.set_xticks(tick_marks)
        ax7.set_yticks(tick_marks)
        ax7.set_xticklabels(range(10))
        ax7.set_yticklabels(range(10))

        ax7.set_ylabel('True Digit')
        ax7.set_xlabel('Predicted Digit')

        # Add text annotations to the matrix
        thresh = cm_norm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax7.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > thresh else "black",
                         fontsize=10)

    plt.tight_layout()
    plt.savefig('comparison_with_cm.png', dpi=300)
    plt.show()