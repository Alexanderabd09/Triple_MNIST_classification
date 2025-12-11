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
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "triple_mnist/test/",
    image_size = img_size,
    batch_size = batch_Size,
    shuffle = False
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "triple_mnist/val/",
    image_size = img_size,
    batch_size = batch_Size,
    shuffle = False
)
#getting classes names and the number of classes
class_names = train_ds.class_names
num_classes = len(class_names)

#normalisation class, we've got three datasets so class will be quite useful

def pre_processing(img, label):
    #couldn't use the dataset straight without unpacking for normalising because its not a tuple
    img = tf.cast(img, tf.float32)/255.0
    img = tf.image.rgb_to_grayscale(img)

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

        label_idx = np.argmax(labels[i].numpy())
        plt.title(class_names[label_idx], fontsize=8)
        plt.axis("off")

    plt.show()


print(show_samples(train_ds))

#flatten dataset
x = train_ds.map(img)