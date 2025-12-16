import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, datasets, models
import numpy as np
import CNN_CWRK_799668 as cw
from CNN_CWRK_799668 import pre_processing, split_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

print("Loading data...")
train_ds = cw.train_ds
test_ds = cw.test_ds

# Get class information
class_names =cw.class_names
num_classes =cw.num_classes
print(f"Number of classes: {num_classes}")
print(f"Sample class names: {class_names[:10]}")


# Preprocessing function
train_ds = train_ds.map(pre_processing)
test_ds = test_ds.map(pre_processing)

# Get data
print("Loading data...")

X_train, y_train = split_dataset(train_ds)
X_test, y_test = split_dataset(test_ds)


# Prepare data for each model, flatten
X_train_lr = np.array([images.flatten() for images in X_train])
X_test_lr = np.array([images.flatten() for images in X_test])

X_train_cnn = X_train
X_test_cnn = X_test

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Models
model_lr = LogisticRegression(max_iter=500, multi_class='multinomial', verbose=1)

model_cnn = models.Sequential()
model_cnn.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(*cw.img_size, 1)))
model_cnn.add(layers.MaxPooling2D(2,2))
model_cnn.add(layers.Conv2D(64, (3,3), activation='relu'))
model_cnn.add(layers.MaxPooling2D(2,2))
model_cnn.add(layers.Conv2D(128, (3,3), activation='relu'))
model_cnn.add(layers.MaxPooling2D(2,2))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(512, activation='relu'))
model_cnn.add(layers.Dropout(0.5))
model_cnn.add(layers.Dense(num_classes, activation='softmax'))
model_cnn.compile(
    loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy']
)

# Benchmarking functions
def benchmark_lr(model, X_train, X_test, y_train, y_test):
    results = {}

    print("Training Logistic Regression...")
    start = time.time()
    model.fit(X_train, y_train)
    results["training_time"] = time.time() - start

    print("Making predictions...")
    start = time.time()
    predictions = model.predict(X_test)
    results["prediction_time"] = time.time() - start

    results["accuracy"] = accuracy_score(y_test, predictions)
    results["f1_score"] = f1_score(y_test, predictions, average='macro')

    return results


def benchmark_cnn(model, X_train, X_test, y_train, y_test, epochs, batch_size):
    results = {}

    print(f"Training CNN for {epochs} epochs...")
    start = time.time()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size ,
        validation_split=0.1,
        verbose=1
    )
    results["training_time"] = time.time() - start

    print("Making predictions...")
    start = time.time()
    predictions = model.predict(X_test, verbose=0).argmax(axis=1)
    results["prediction_time"] = time.time() - start

    results["accuracy"] = accuracy_score(y_test, predictions)
    results["f1_score"] = f1_score(y_test, predictions, average='macro')

    # Store training history
    results["history"] = history.history

    return results


# Run benchmarks
print("\n" + "=" * 50)
print("BENCHMARKING LOGISTIC REGRESSION")
print("=" * 50)
results_lr = benchmark_lr(model_lr, X_train_lr, X_test_lr, y_train, y_test)

print("\n" + "=" * 50)
print("BENCHMARKING CNN")
print("=" * 50)
results_cnn = benchmark_cnn(model_cnn, X_train_cnn, X_test_cnn, y_train, y_test, epochs=10, batch_size=64)

# Create results DataFrame
df = pd.DataFrame(
    [results_lr, results_cnn],
    index=['Logistic Regression', 'Convolutional Neural Network']
)

print("\n" + "=" * 50)
print("BENCHMARK RESULTS")
print("=" * 50)
print(df)

# Visualization
fig = plt.figure(figsize=(16, 10))

# 1. Grouped Bar Chart - All Metrics
ax1 = plt.subplot(2, 3, 1)
metrics = ['accuracy', 'f1_score', 'training_time', 'prediction_time']
x = np.arange(len(metrics))
width = 0.35

lr_values = [df.loc['Logistic Regression', m] for m in metrics]
cnn_values = [df.loc['Convolutional Neural Network', m] for m in metrics]

bars1 = ax1.bar(x - width / 2, lr_values, width, label='Logistic Regression', alpha=0.8)
bars2 = ax1.bar(x + width / 2, cnn_values, width, label='CNN', alpha=0.8)

ax1.set_xlabel('Metrics')
ax1.set_ylabel('Values')
ax1.set_title('All Metrics Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 2. Accuracy Comparison
ax2 = plt.subplot(2, 3, 2)
models = df.index
accuracy_vals = df['accuracy'].values
colors = ['#FF6B6B', '#4ECDC4']
bars = ax2.barh(models, accuracy_vals, color=colors, alpha=0.8)
ax2.set_xlabel('Accuracy')
ax2.set_title('Accuracy Comparison')
ax2.set_xlim([0, 1])
for i, (bar, val) in enumerate(zip(bars, accuracy_vals)):
    ax2.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. F1 Score Comparison
ax3 = plt.subplot(2, 3, 3)
f1_vals = df['f1_score'].values
bars = ax3.barh(models, f1_vals, color=colors, alpha=0.8)
ax3.set_xlabel('F1 Score (Macro)')
ax3.set_title('F1 Score Comparison')
ax3.set_xlim([0, 1])
for i, (bar, val) in enumerate(zip(bars, f1_vals)):
    ax3.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Training Time Comparison
ax4 = plt.subplot(2, 3, 4)
train_time_vals = df['training_time'].values
bars = ax4.bar(models, train_time_vals, color=colors, alpha=0.8)
ax4.set_ylabel('Time (seconds)')
ax4.set_title('Training Time Comparison')
ax4.set_xticklabels(models, rotation=45, ha='right')
for bar, val in zip(bars, train_time_vals):
    ax4.text(bar.get_x() + bar.get_width() / 2., val + max(train_time_vals) * 0.02,
             f'{val:.2f}s', ha='center', fontsize=10, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Prediction Time Comparison
ax5 = plt.subplot(2, 3, 5)
pred_time_vals = df['prediction_time'].values
bars = ax5.bar(models, pred_time_vals, color=colors, alpha=0.8)
ax5.set_ylabel('Time (seconds)')
ax5.set_title('Prediction Time Comparison')
ax5.set_xticklabels(models, rotation=45, ha='right')
for bar, val in zip(bars, pred_time_vals):
    ax5.text(bar.get_x() + bar.get_width() / 2., val + max(pred_time_vals) * 0.02,
             f'{val:.3f}s', ha='center', fontsize=10, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 6. CNN Training History (if available)
ax6 = plt.subplot(2, 3, 6)
if 'history' in results_cnn:
    history = results_cnn['history']
    epochs_range = range(1, len(history['accuracy']) + 1)
    ax6.plot(epochs_range, history['accuracy'], label='Training Accuracy', marker='o')
    if 'val_accuracy' in history:
        ax6.plot(epochs_range, history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Accuracy')
    ax6.set_title('CNN Training Progress')
    ax6.legend()
    ax6.grid(alpha=0.3)
else:
    # Performance vs Time Trade-off
    ax6.scatter(df['training_time'], df['accuracy'], s=300, alpha=0.6, c=colors)
    for i, model in enumerate(models):
        ax6.annotate(model, (df.loc[model, 'training_time'], df.loc[model, 'accuracy']),
                     xytext=(10, 10), textcoords='offset points', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))
    ax6.set_xlabel('Training Time (seconds)')
    ax6.set_ylabel('Accuracy')
    ax6.set_title('Accuracy vs Training Time Trade-off')
    ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'model_comparison.png'")