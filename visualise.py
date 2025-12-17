import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_results(df, cnn_history=None):
    fig = plt.figure(figsize=(16, 10))
    colors = ['#FF6B6B', '#4ECDC4']
    models = df.index

    # 1. Grouped Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    metrics = ['accuracy', 'f1_score', 'training_time', 'prediction_time']
    x = np.arange(len(metrics))
    width = 0.35
    for i, model_name in enumerate(models):
        vals = [df.loc[model_name, m] for m in metrics]
        ax1.bar(x + (i*width - width/2), vals, width, label=model_name, alpha=0.8)
    ax1.set_title('All Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()

    # 2. Accuracy Comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.barh(models, df['accuracy'], color=colors)
    ax2.set_title('Accuracy')
    ax2.set_xlim(0, 1)

    # 3. F1 Score
    ax3 = plt.subplot(2, 3, 3)
    ax3.barh(models, df['f1_score'], color=colors)
    ax3.set_title('F1 Score')
    ax3.set_xlim(0, 1)

    # 4. Training Time
    ax4 = plt.subplot(2, 3, 4)
    ax4.bar(models, df['training_time'], color=colors)
    ax4.set_title('Training Time (s)')

    # 5. Prediction Time
    ax5 = plt.subplot(2, 3, 5)
    ax5.bar(models, df['prediction_time'], color=colors)
    ax5.set_title('Prediction Time (s)')

    # 6. CNN Training History
    ax6 = plt.subplot(2, 3, 6)
    if cnn_history:
        ax6.plot(cnn_history['accuracy'], label='Train Acc')
        if 'val_accuracy' in cnn_history:
            ax6.plot(cnn_history['val_accuracy'], label='Val Acc')
        ax6.set_title('CNN Training Progress')
        ax6.legend()
    else:
        ax6.scatter(df['training_time'], df['accuracy'], s=200, c=colors)
        ax6.set_title('Accuracy vs Time')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()