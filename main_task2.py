import pandas as pd
import Benchmark_models as bm
import pre_processing as pp
import visualise as vs
import tensorflow as tf


img_size = (84, 84,)
batch_Size = 64
epochs = 10

#getting the datasets
print("\n Getting datasets ...")
train_ds = pp.prepare_dataset("triple_mnist/train/", img_size, batch_Size)
test_ds = pp.prepare_dataset("triple_mnist/test/", img_size, batch_Size)
val_ds = pp.prepare_dataset("triple_mnist/val/", img_size, batch_Size)

raw_ds = tf.keras.utils.image_dataset_from_directory(
    "triple_mnist/train/",
    image_size=img_size,
    batch_size=batch_Size,
)
class_names = raw_ds.class_names
num_classes = len(class_names)

print(f"Number of classes detected: {num_classes}")
pp.show_samples(train_ds, class_names)

print("\nConverting datasets to arrays for benchmarking...")
X_train, y_train = pp.split_dataset(train_ds)
X_val, y_val = pp.split_dataset(val_ds)
X_test, y_test = pp.split_dataset(test_ds)

print("\n" + "="*30)
print("RUNNING LOGISTIC REGRESSION BENCHMARK")
print("="*30)
# Flattening is handled inside run_benchmark_lr
results_lr = bm.run_benchmark_lr(X_train, X_test, y_train, y_test)

print("\n" + "="*30)
print("RUNNING CNN BENCHMARK")
print("="*30)
# Initialize the standard CNN model architecture
cnn_model = bm.create_cnn_model(img_size, num_classes)
results_cnn = bm.run_benchmark_cnn(
    cnn_model, X_train, X_test, y_train, y_test,
    X_val, y_val, epochs=epochs, batch_size=batch_Size
)

df_results = pd.DataFrame(
    [results_lr, results_cnn],
    index=['Logistic Regression', 'Standard CNN']
)

print("\n--- Final Benchmark Results ---")
print(df_results[['accuracy', 'f1_score', 'training_time']])

# 6. Visualization
print("\nGenerating comparison plots...")
vs.plot_benchmark_results(df_results, cnn_history=results_cnn.get('history'))
