import pandas as pd
import Benchmark_models as bm
import pre_processing as pp
import visualise as vs
import tensorflow as tf

# Configuration
IMG_SIZE = (84, 84)
BATCH_SIZE = 64
EPOCHS = 10

# --- STEP 1: Baseline Training (Standard CNN) ---
print("\n--- Training Standard CNN (Task 2 Baseline) ---")
# Load as full images (split=False)
train_ds_std = pp.prepare_dataset("triple_mnist/train/", IMG_SIZE, BATCH_SIZE, split=False)
val_ds_std = pp.prepare_dataset("triple_mnist/val/", IMG_SIZE, BATCH_SIZE, split=False)
test_ds_std = pp.prepare_dataset("triple_mnist/test/", IMG_SIZE, BATCH_SIZE, split=False)

X_train, y_train = pp.split_dataset(train_ds_std)
X_val, y_val = pp.split_dataset(val_ds_std)
X_test, y_test = pp.split_dataset(test_ds_std)

std_cnn = bm.create_cnn_model(IMG_SIZE, num_classes=1000) # 1000 classes for 000-999
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