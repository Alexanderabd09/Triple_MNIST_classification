import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
import Benchmark_models as bm
import pre_processing as pp
import visualise as vs
import main_task2 as t2

# Configuration
IMG_SIZE = t2.img_size
BATCH_SIZE = t2.batch_Size
EPOCHS = t2.epochs
NUM_CLASSES = t2.num_classes

# --- STEP 1: Baseline Processing (Load or Train Standard CNN/LR) ---
print("\n--- Processing Baseline (Task 2) ---")

# Determine model path based on best_model_id from task 2
model_path = "best_model.keras" if getattr(t2, 'best_model_id', 1) == 1 else "best_model.pkl"

test_ds_std = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=False)
X_test, y_test = pp.split_dataset(test_ds_std)

results_std = {}
best_baseline_model = None

if os.path.exists(model_path):
    print(f"Found existing best model at '{model_path}'. Loading for evaluation...")
    start_time = time.time()

    if model_path.endswith('.keras'):
        best_baseline_model = tf.keras.models.load_model(model_path)
        y_pred_probs = best_baseline_model.predict(X_test, verbose=0)
        y_pred_std = np.argmax(y_pred_probs, axis=-1)
    else:
        # Load Logistic Regression using joblib
        best_baseline_model = joblib.load(model_path)
        y_pred_std = best_baseline_model.predict(X_test.reshape(len(X_test), -1))

    eval_time = time.time() - start_time
    results_std = {
        'accuracy': accuracy_score(y_test, y_pred_std),
        'f1_score': f1_score(y_test, y_pred_std, average='macro'),
        'training_time': 0.0,
        'prediction_time': eval_time,
        'history': None
    }
    print(f"Loaded Baseline Accuracy: {results_std['accuracy']:.4f}")
else:
    print("No saved model found. Training Standard CNN from scratch...")
    train_ds_std = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=False)
    val_ds_std = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=False)
    X_train, y_train = pp.split_dataset(train_ds_std)
    X_val, y_val = pp.split_dataset(val_ds_std)

    best_baseline_model = bm.create_cnn_model(IMG_SIZE, NUM_CLASSES)
    results_std = bm.run_benchmark_cnn(best_baseline_model, X_train, X_test, y_train, y_test, X_val, y_val, EPOCHS,
                                       BATCH_SIZE)

# --- STEP 2: Task 4 Training (Split Siamese CNN) ---
print("\n--- Training Split CNN (Task 4) ---")
train_ds_split = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=True)
val_ds_split = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=True)
test_ds_split = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=True)

# Note: Ensure you have build_split_cnn in your pre_processing.py
split_model = pp.build_split_cnn(num_classes=10)
results_split, y_true_total, y_pred_total = bm.run_benchmark_split_cnn(split_model, train_ds_split, val_ds_split,
                                                                       test_ds_split, EPOCHS)

# --- STEP 3: Comparison ---
df_comparison = pd.DataFrame(
    [results_std, results_split],
    index=['Baseline', 'Split CNN']
)

print("\n" + "=" * 40)
print("Final Benchmark Comparison")
print("=" * 40)
print(df_comparison[['accuracy', 'f1_score', 'training_time']])

# --- STEP 4: SAVE THE ABSOLUTE WINNER ---
best_overall_name = df_comparison['accuracy'].idxmax()
print(f"\nOverall Best: {best_overall_name}")

if best_overall_name == 'Baseline':
    if model_path.endswith('.keras'):
        best_baseline_model.save('best_final_model.keras')
    else:
        joblib.dump(best_baseline_model, 'best_final_model.pkl')
    print("Saved Baseline as 'best_final_model'")
else:
    split_model.save('best_final_model.keras')
    print("Saved Split CNN as 'best_final_model.keras'")

#  Visuals
vs.plot_benchmark_results(
    df_comparison,
    cnn_history=results_split['history'],
    y_true=y_true_total,
    y_pred=y_pred_total
)