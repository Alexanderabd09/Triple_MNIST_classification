import pandas as pd
import Benchmark_models as bm
import pre_processing as pp
import visualise as vs
import tensorflow as tf
import joblib
import os


img_size = (84, 84,)
batch_Size = 64
epochs = 10

#getting the datasets
print("\n Getting datasets ...")
train_ds = pp.prepare_dataset("triple_mnist/train/", img_size, batch_Size, split = False)
test_ds = pp.prepare_dataset("triple_mnist/test/", img_size, batch_Size, split = False)
val_ds = pp.prepare_dataset("triple_mnist/val/", img_size, batch_Size, split = False)

raw_ds = tf.keras.utils.image_dataset_from_directory(
    "triple_mnist/train/",
    image_size=img_size,
    batch_size=batch_Size,
)
class_names = raw_ds.class_names
num_classes = len(class_names)

print(f"classes detected: {num_classes}")
pp.show_samples(train_ds, class_names)

print("\nConverting datasets to arrays for benchmarking...")
X_train, y_train = pp.split_dataset(train_ds)
X_val, y_val = pp.split_dataset(val_ds)
X_test, y_test = pp.split_dataset(test_ds)

print("\n" + "_"*30)
print("Running Logistic regression Benchmark...")
print("_"*30)
# Flattening is handled inside run_benchmark_lr
results_lr, model_lr = bm.run_benchmark_lr(X_train, X_test, y_train, y_test)


print("\n" + "_"*30)
print("Running CNN benchmark...")
print("_"*30)
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

print("\nFinal Benchmark Results... ")
print(df_results[['accuracy', 'f1_score', 'training_time']])

best_model_id = 0;
best_model_name = df_results['accuracy'].idxmax()
print(f"\n>>> Best performing model: {best_model_name} (Acc: {df_results.loc[best_model_name, 'accuracy']:.4f})")

if best_model_name == 'Standard CNN':
    cnn_model.save('best_model.keras')
    print("Successfully saved CNN to 'best_model.keras'")
    best_model_id = 1;

elif best_model_name == 'Logistic Regression':

    joblib.dump(model_lr, 'best_model.pkl')
    print("Successfully saved Logistic Regression to 'best_model.pkl'")
    best_model_id = 2;

# 6. Visualization
print("\nGenerating comparison plots...")
vs.plot_benchmark_results(df_results, cnn_history=results_cnn.get('history'))
