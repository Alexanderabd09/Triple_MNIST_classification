import os
import time
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
import pandas as pd
import Benchmark_models as bm
import pre_processing as pp
import visualise as vs
import tensorflow as tf
import main_task2 as t2

# Configuration
IMG_SIZE = t2.img_size
BATCH_SIZE = t2.batch_Size
EPOCHS = t2.epochs
# --- STEP 1: Baseline Processing (Load or Train Standard CNN) ---
print("\n--- Processing Standard CNN (Task 2 Baseline) ---")

if (t2.best_model_id == 1):
    model_path = "best_model.keras"
elif (t2.best_model_id == 2):
    model_path = "best_model.pkl"

test_ds_std = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=False)
X_test, y_test = pp.split_dataset(test_ds_std)

if os.path.exists(model_path):
    print(f"Found existing best model at '{model_path}'. Loading for evaluation...")
    std_cnn = tf.keras.models.load_model(model_path)

    # Evaluate the loaded model to populate results for comparison
    start_time = time.time()
    y_pred_probs = std_cnn.predict(X_test, verbose=0)
    eval_time = time.time() - start_time
    y_pred = np.argmax(y_pred_probs, axis=-1)

    results_std = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'training_time': 0.0,  # 0 because we didn't train it now
        'prediction_time': eval_time,
        'history': None  # No history for a loaded model
    }
    print(f"Loaded Model Accuracy: {results_std['accuracy']:.4f}")
else:
    print("No saved model found. Training Standard CNN from scratch...")
    train_ds_std = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=False)
    val_ds_std = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=False)
    X_train, y_train = pp.split_dataset(train_ds_std)
    X_val, y_val = pp.split_dataset(val_ds_std)

    std_cnn = bm.create_cnn_model(IMG_SIZE, NUM_CLASSES)
    results_std = bm.run_benchmark_cnn(std_cnn, X_train, X_test, y_train, y_test, X_val, y_val, EPOCHS, BATCH_SIZE)
# --- STEP 2: Task 4 Training (Split Siamese CNN) ---
print("\n--- Training Split CNN (Task 4) ---")
# Load as 3-way split pieces (split=True)
train_ds_split = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=True)
val_ds_split = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=True)
test_ds_split = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=True)

split_model = pp.build_split_cnn(num_classes=10) # 10 classes per head (0-9)
results_split, y_true, y_pred = bm.run_benchmark_split_cnn(split_model, train_ds_split, val_ds_split, test_ds_split, EPOCHS)

# --- STEP 3: Comparison and Visuals ---
df_comparison = pd.DataFrame(
    [results_std, results_split],
    index=['Standard CNN', 'Split CNN (Siamese)']
)

print("\n" + "="*40)
print("BENCHMARK COMPARISON")
print("="*40)
print(df_comparison[['accuracy', 'f1_score', 'training_time']])

vs.plot_benchmark_results(
    df_comparison,
    cnn_history=results_split['history'],
    y_true=y_true,
    y_pred=y_pred
)