import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_benchmark_results(df, cnn_history=None, lr_tuning_history=None, y_true=None, y_pred=None):
    """
    Creates a comprehensive visualization comparing Logistic Regression and CNN.
    Used for TASK 2.

    Args:
        df: DataFrame containing model comparison metrics
        cnn_history: Training history from CNN model (must include 'loss' and 'val_loss')
        lr_tuning_history: List of dicts with hyperparameter tuning results
        y_true: True labels for confusion matrix (1000 classes)
        y_pred: Predicted labels for confusion matrix (1000 classes)
    """
    fig = plt.figure(figsize=(20, 12))

    # 1. Loss Curves (CNN only - LR doesn't have epoch-wise loss)
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

    # 2. Accuracy Comparison
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

    # 3. F1 Score Comparison
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

    # 4. CNN Training Progress (Accuracy)
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

    # 5. Full 1000x1000 Confusion Matrix Heatmap
    if y_true is not None and y_pred is not None:
        ax5 = plt.subplot(2, 3, 5)
        cm = confusion_matrix(y_true, y_pred)

        # Use a more appropriate colormap for large matrices
        im = ax5.imshow(cm, cmap='Blues', interpolation='nearest', aspect='auto')
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

        ax5.set_xlabel('Predicted Label', fontsize=11)
        ax5.set_ylabel('True Label', fontsize=11)
        ax5.set_title('Confusion Matrix (1000x1000)', fontsize=13, weight='bold')

        # Add tick marks at intervals for readability
        tick_interval = 100
        tick_positions = np.arange(0, 1000, tick_interval)
        ax5.set_xticks(tick_positions)
        ax5.set_yticks(tick_positions)
        ax5.set_xticklabels(tick_positions, fontsize=8)
        ax5.set_yticklabels(tick_positions, fontsize=8)

    # 6. Results Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Select only numeric columns for the table
    numeric_cols = ['accuracy', 'f1_score', 'training_time']
    df_display = df[numeric_cols].copy()

    # Format the data for display
    table_data = []
    for idx in df_display.index:
        row_data = []
        for col in df_display.columns:
            val = df_display.loc[idx, col]
            if col == 'training_time':
                row_data.append(f'{val:.2f}s')
            else:
                row_data.append(f'{val:.4f}')
        table_data.append(row_data)

    table = ax6.table(
        cellText=table_data,
        colLabels=df_display.columns,
        rowLabels=df_display.index,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(len(df_display.index)):
        table[(i + 1, -1)].set_facecolor('#E8E8E8')
        table[(i + 1, -1)].set_text_props(weight='bold')

    ax6.set_title('Detailed Model Comparison', pad=20, fontsize=13, weight='bold')

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'benchmark_results.png'")
    plt.show()


def plot_split_model_comparison(results_split, res_baseline, baseline_history, y_true_split, y_pred_split,
                                y_true_baseline, y_pred_baseline):
    """
    Creates comprehensive comparison visualizations between baseline and split models.
    Used for TASK 4 vs TASK 2 comparison.

    Args:
        results_split: Results dictionary from split model
        res_baseline: Results dictionary from baseline model
        baseline_history: Training history from baseline model
        y_true_split: True labels from split model
        y_pred_split: Predicted labels from split model
        y_true_baseline: True labels from baseline model
        y_pred_baseline: Predicted labels from baseline model
    """
    fig = plt.figure(figsize=(20, 12))

    # 1. Loss Curves Comparison
    ax1 = plt.subplot(2, 3, 1)

    if baseline_history and 'loss' in baseline_history:
        epochs_baseline = range(1, len(baseline_history['loss']) + 1)
        ax1.plot(epochs_baseline, baseline_history['loss'], 'b-o', label='Baseline Train Loss', linewidth=2)
        ax1.plot(epochs_baseline, baseline_history['val_loss'], 'b--s', label='Baseline Val Loss', linewidth=2,
                 alpha=0.7)

    if 'history' in results_split and 'loss' in results_split['history']:
        # Split model has combined loss
        history_split = results_split['history']
        epochs_split = range(1, len(history_split['loss']) + 1)
        ax1.plot(epochs_split, history_split['loss'], 'r-o', label='Split Train Loss', linewidth=2)
        ax1.plot(epochs_split, history_split['val_loss'], 'r--s', label='Split Val Loss', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Curves Comparison', fontsize=13, weight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy Comparison
    ax2 = plt.subplot(2, 3, 2)
    models = ['Baseline CNN', 'Split CNN']
    accuracies = [res_baseline['accuracy'], results_split['accuracy']]
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

    # 3. F1 Score Comparison
    ax3 = plt.subplot(2, 3, 3)
    f1_scores = [res_baseline['f1_score'], results_split['f1_score']]
    bars = ax3.bar(models, f1_scores, color=colors, edgecolor='darkred', alpha=0.7)
    ax3.set_ylabel('F1 Score', fontsize=11)
    ax3.set_title('Model F1 Score Comparison', fontsize=13, weight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # 4. Baseline Model Confusion Matrix
    ax4 = plt.subplot(2, 3, 4)
    cm_baseline = confusion_matrix(y_true_baseline, y_pred_baseline)
    im4 = ax4.imshow(cm_baseline, cmap='Blues', interpolation='nearest', aspect='auto')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_xlabel('Predicted Label', fontsize=11)
    ax4.set_ylabel('True Label', fontsize=11)
    ax4.set_title('Baseline CNN Confusion Matrix (1000x1000)', fontsize=13, weight='bold')
    tick_interval = 100
    tick_positions = np.arange(0, 1000, tick_interval)
    ax4.set_xticks(tick_positions)
    ax4.set_yticks(tick_positions)
    ax4.set_xticklabels(tick_positions, fontsize=8)
    ax4.set_yticklabels(tick_positions, fontsize=8)

    # 5. Split Model Confusion Matrix
    ax5 = plt.subplot(2, 3, 5)
    cm_split = confusion_matrix(y_true_split, y_pred_split)
    im5 = ax5.imshow(cm_split, cmap='Reds', interpolation='nearest', aspect='auto')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.set_xlabel('Predicted Label', fontsize=11)
    ax5.set_ylabel('True Label', fontsize=11)
    ax5.set_title('Split CNN Confusion Matrix (1000x1000)', fontsize=13, weight='bold')
    ax5.set_xticks(tick_positions)
    ax5.set_yticks(tick_positions)
    ax5.set_xticklabels(tick_positions, fontsize=8)
    ax5.set_yticklabels(tick_positions, fontsize=8)

    # 6. Split Model Per-Digit Accuracy
    if 'history' in results_split:
        ax6 = plt.subplot(2, 3, 6)
        history = results_split['history']
        epochs = range(1, len(history['out_1_accuracy']) + 1)

        ax6.plot(epochs, history['out_1_accuracy'], 'b-o', label='Digit 1 (Train)', linewidth=2)
        ax6.plot(epochs, history['out_2_accuracy'], 'g-s', label='Digit 2 (Train)', linewidth=2)
        ax6.plot(epochs, history['out_3_accuracy'], 'r-^', label='Digit 3 (Train)', linewidth=2)

        if 'val_out_1_accuracy' in history:
            ax6.plot(epochs, history['val_out_1_accuracy'], 'b--', label='Digit 1 (Val)', linewidth=2, alpha=0.6)
            ax6.plot(epochs, history['val_out_2_accuracy'], 'g--', label='Digit 2 (Val)', linewidth=2, alpha=0.6)
            ax6.plot(epochs, history['val_out_3_accuracy'], 'r--', label='Digit 3 (Val)', linewidth=2, alpha=0.6)

        ax6.set_xlabel('Epoch', fontsize=11)
        ax6.set_ylabel('Accuracy', fontsize=11)
        ax6.set_title('Split Model: Per-Digit Accuracy', fontsize=13, weight='bold')
        ax6.legend(loc='lower right', fontsize=8, ncol=2)
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('split_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'split_model_comparison.png'")
    plt.show()


def plot_task5_multilabel_results(results_multilabel, baseline_history=None):
    """
    Visualize Task 5a Multi-label CNN results.

    Args:
        results_multilabel: Results from multi-label CNN training
        baseline_history: Optional baseline history for comparison
    """
    fig = plt.figure(figsize=(16, 10))

    history = results_multilabel['history']

    # 1. Per-digit training accuracy
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(history['out_1_accuracy']) + 1)
    ax1.plot(epochs, history['out_1_accuracy'], 'b-o', label='Digit 1', linewidth=2)
    ax1.plot(epochs, history['out_2_accuracy'], 'g-s', label='Digit 2', linewidth=2)
    ax1.plot(epochs, history['out_3_accuracy'], 'r-^', label='Digit 3', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Accuracy', fontsize=11)
    ax1.set_title('Per-Digit Training Accuracy', fontsize=13, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Per-digit validation accuracy
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['val_out_1_accuracy'], 'b-o', label='Digit 1', linewidth=2)
    ax2.plot(epochs, history['val_out_2_accuracy'], 'g-s', label='Digit 2', linewidth=2)
    ax2.plot(epochs, history['val_out_3_accuracy'], 'r-^', label='Digit 3', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Accuracy', fontsize=11)
    ax2.set_title('Per-Digit Validation Accuracy', fontsize=13, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Loss curves
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['loss'], 'b-o', label='Training Loss', linewidth=2)
    ax3.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Combined Loss Curves', fontsize=13, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Confusion matrix
    ax4 = plt.subplot(2, 3, 4)
    cm = confusion_matrix(results_multilabel['y_true'], results_multilabel['y_pred'])
    im = ax4.imshow(cm, cmap='Blues', interpolation='nearest', aspect='auto')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_xlabel('Predicted Label', fontsize=11)
    ax4.set_ylabel('True Label', fontsize=11)
    ax4.set_title('Confusion Matrix (1000x1000)', fontsize=13, weight='bold')
    tick_interval = 100
    tick_positions = np.arange(0, 1000, tick_interval)
    ax4.set_xticks(tick_positions)
    ax4.set_yticks(tick_positions)
    ax4.set_xticklabels(tick_positions, fontsize=8)
    ax4.set_yticklabels(tick_positions, fontsize=8)

    # 5. Performance summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    summary_text = f"""
    MULTI-LABEL CNN RESULTS

    Test Accuracy: {results_multilabel['accuracy']:.4f}
    Test F1 Score: {results_multilabel['f1_score']:.4f}
    Training Time: {results_multilabel['training_time']:.1f}s

    Final Validation Accuracy:
      Digit 1: {history['val_out_1_accuracy'][-1]:.4f}
      Digit 2: {history['val_out_2_accuracy'][-1]:.4f}
      Digit 3: {history['val_out_3_accuracy'][-1]:.4f}
    """
    ax5.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig('task5_multilabel_results.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'task5_multilabel_results.png'")
    plt.show()

    """
    Creates a comprehensive visualization comparing Logistic Regression and CNN.

    Args:
        df: DataFrame containing model comparison metrics
        cnn_history: Training history from CNN model (must include 'loss' and 'val_loss')
        lr_tuning_history: List of dicts with hyperparameter tuning results
        y_true: True labels for confusion matrix (1000 classes)
        y_pred: Predicted labels for confusion matrix (1000 classes)
    """
    fig = plt.figure(figsize=(20, 12))

    # 1. Loss Curves (CNN only - LR doesn't have epoch-wise loss)
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

    # 2. Accuracy Comparison
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

    # 3. F1 Score Comparison
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

    # 4. CNN Training Progress (Accuracy)
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

    # 5. Full 1000x1000 Confusion Matrix Heatmap
    if y_true is not None and y_pred is not None:
        ax5 = plt.subplot(2, 3, 5)
        cm = confusion_matrix(y_true, y_pred)

        # Use a more appropriate colormap for large matrices
        im = ax5.imshow(cm, cmap='Blues', interpolation='nearest', aspect='auto')
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

        ax5.set_xlabel('Predicted Label', fontsize=11)
        ax5.set_ylabel('True Label', fontsize=11)
        ax5.set_title('Confusion Matrix (1000x1000)', fontsize=13, weight='bold')

        # Add tick marks at intervals for readability
        tick_interval = 100
        tick_positions = np.arange(0, 1000, tick_interval)
        ax5.set_xticks(tick_positions)
        ax5.set_yticks(tick_positions)
        ax5.set_xticklabels(tick_positions, fontsize=8)
        ax5.set_yticklabels(tick_positions, fontsize=8)

    # 6. Results Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Select only numeric columns for the table
    numeric_cols = ['accuracy', 'f1_score', 'training_time']
    df_display = df[numeric_cols].copy()

    # Format the data for display
    table_data = []
    for idx in df_display.index:
        row_data = []
        for col in df_display.columns:
            val = df_display.loc[idx, col]
            if col == 'training_time':
                row_data.append(f'{val:.2f}s')
            else:
                row_data.append(f'{val:.4f}')
        table_data.append(row_data)

    table = ax6.table(
        cellText=table_data,
        colLabels=df_display.columns,
        rowLabels=df_display.index,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(len(df_display.index)):
        table[(i + 1, -1)].set_facecolor('#E8E8E8')
        table[(i + 1, -1)].set_text_props(weight='bold')

    ax6.set_title('Detailed Model Comparison', pad=20, fontsize=13, weight='bold')

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'benchmark_results.png'")
    plt.show()


def plot_split_model_comparison(results_split, res_baseline, baseline_history, y_true_split, y_pred_split,
                                y_true_baseline, y_pred_baseline):
    """
    Creates comprehensive comparison visualizations between baseline and split models.

    Args:
        results_split: Results dictionary from split model
        res_baseline: Results dictionary from baseline model
        baseline_history: Training history from baseline model
        y_true_split: True labels from split model
        y_pred_split: Predicted labels from split model
        y_true_baseline: True labels from baseline model
        y_pred_baseline: Predicted labels from baseline model
    """
    fig = plt.figure(figsize=(20, 12))

    # 1. Loss Curves Comparison
    ax1 = plt.subplot(2, 3, 1)

    if baseline_history and 'loss' in baseline_history:
        epochs_baseline = range(1, len(baseline_history['loss']) + 1)
        ax1.plot(epochs_baseline, baseline_history['loss'], 'b-o', label='Baseline Train Loss', linewidth=2)
        ax1.plot(epochs_baseline, baseline_history['val_loss'], 'b--s', label='Baseline Val Loss', linewidth=2,
                 alpha=0.7)

    if 'history' in results_split and 'loss' in results_split['history']:
        # Split model has combined loss
        history_split = results_split['history']
        epochs_split = range(1, len(history_split['loss']) + 1)
        ax1.plot(epochs_split, history_split['loss'], 'r-o', label='Split Train Loss', linewidth=2)
        ax1.plot(epochs_split, history_split['val_loss'], 'r--s', label='Split Val Loss', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Curves Comparison', fontsize=13, weight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy Comparison
    ax2 = plt.subplot(2, 3, 2)
    models = ['Baseline CNN', 'Split CNN']
    accuracies = [res_baseline['accuracy'], results_split['accuracy']]
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

    # 3. F1 Score Comparison
    ax3 = plt.subplot(2, 3, 3)
    f1_scores = [res_baseline['f1_score'], results_split['f1_score']]
    bars = ax3.bar(models, f1_scores, color=colors, edgecolor='darkred', alpha=0.7)
    ax3.set_ylabel('F1 Score', fontsize=11)
    ax3.set_title('Model F1 Score Comparison', fontsize=13, weight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # 4. Baseline Model Confusion Matrix
    ax4 = plt.subplot(2, 3, 4)
    cm_baseline = confusion_matrix(y_true_baseline, y_pred_baseline)
    im4 = ax4.imshow(cm_baseline, cmap='Blues', interpolation='nearest', aspect='auto')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_xlabel('Predicted Label', fontsize=11)
    ax4.set_ylabel('True Label', fontsize=11)
    ax4.set_title('Baseline CNN Confusion Matrix (1000x1000)', fontsize=13, weight='bold')
    tick_interval = 100
    tick_positions = np.arange(0, 1000, tick_interval)
    ax4.set_xticks(tick_positions)
    ax4.set_yticks(tick_positions)
    ax4.set_xticklabels(tick_positions, fontsize=8)
    ax4.set_yticklabels(tick_positions, fontsize=8)

    # 5. Split Model Confusion Matrix
    ax5 = plt.subplot(2, 3, 5)
    cm_split = confusion_matrix(y_true_split, y_pred_split)
    im5 = ax5.imshow(cm_split, cmap='Reds', interpolation='nearest', aspect='auto')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.set_xlabel('Predicted Label', fontsize=11)
    ax5.set_ylabel('True Label', fontsize=11)
    ax5.set_title('Split CNN Confusion Matrix (1000x1000)', fontsize=13, weight='bold')
    ax5.set_xticks(tick_positions)
    ax5.set_yticks(tick_positions)
    ax5.set_xticklabels(tick_positions, fontsize=8)
    ax5.set_yticklabels(tick_positions, fontsize=8)

    # 6. Split Model Per-Digit Accuracy
    if 'history' in results_split:
        ax6 = plt.subplot(2, 3, 6)
        history = results_split['history']
        epochs = range(1, len(history['out_1_accuracy']) + 1)

        ax6.plot(epochs, history['out_1_accuracy'], 'b-o', label='Digit 1 (Train)', linewidth=2)
        ax6.plot(epochs, history['out_2_accuracy'], 'g-s', label='Digit 2 (Train)', linewidth=2)
        ax6.plot(epochs, history['out_3_accuracy'], 'r-^', label='Digit 3 (Train)', linewidth=2)

        if 'val_out_1_accuracy' in history:
            ax6.plot(epochs, history['val_out_1_accuracy'], 'b--', label='Digit 1 (Val)', linewidth=2, alpha=0.6)
            ax6.plot(epochs, history['val_out_2_accuracy'], 'g--', label='Digit 2 (Val)', linewidth=2, alpha=0.6)
            ax6.plot(epochs, history['val_out_3_accuracy'], 'r--', label='Digit 3 (Val)', linewidth=2, alpha=0.6)

        ax6.set_xlabel('Epoch', fontsize=11)
        ax6.set_ylabel('Accuracy', fontsize=11)
        ax6.set_title('Split Model: Per-Digit Accuracy', fontsize=13, weight='bold')
        ax6.legend(loc='lower right', fontsize=8, ncol=2)
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('split_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'split_model_comparison.png'")
    plt.show()