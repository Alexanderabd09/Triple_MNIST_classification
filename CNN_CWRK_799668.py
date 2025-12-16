import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



img_size = (32, 32,)
batch_Size = 64

#getting the datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "triple_mnist/train/",
    image_size= img_size,
    batch_size=batch_Size,
    color_mode='grayscale',
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "triple_mnist/test/",
    image_size = img_size,
    batch_size = batch_Size,
    color_mode='grayscale',
    shuffle = False
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "triple_mnist/val/",
    image_size = img_size,
    batch_size = batch_Size,
    color_mode='grayscale',
    shuffle = False
)
#getting classes names and the number of classes
class_names = train_ds.class_names
num_classes = 1000
print(num_classes)

#normalisation class, we've got three datasets so class will be quite useful
def pre_processing(img, label):
    img = tf.cast(img, tf.float32)/255.0

    return img, label

train_ds = train_ds.map(pre_processing)
test_ds = test_ds.map(pre_processing)
val_ds = val_ds.map(pre_processing)


# class for displaying samples from the datasets
def show_samples(dataset):
    for images, labels in dataset.take(1):
        plt.figure(figsize=(12, 8))

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        img = images[i]

        if img.shape[-1] == 1:
            plt.imshow(img[..., 0], cmap='gray')
        else:
            plt.imshow(img)

        label_idx = labels[i].numpy()
        plt.title(class_names[label_idx], fontsize=8)
        plt.axis("off")

    plt.show()


show_samples(train_ds)

#flatten dataset
def split_dataset(dataset):
    X_batches = []
    Y_batches = []
    for images, labels in dataset:
        X_batches.append(images.numpy())
        Y_batches.append(labels.numpy())

    X = np.concatenate(X_batches, axis=0)
    y = np.concatenate(Y_batches, axis=0)

    return X, y

X, y = split_dataset(train_ds)
X_flat = np.array([images.flatten() for images in X])

df = pd.DataFrame(X_flat)
df.insert(0, 'target', y) # Insert labels as the first column
df.to_csv("flattened.csv", index=False)
