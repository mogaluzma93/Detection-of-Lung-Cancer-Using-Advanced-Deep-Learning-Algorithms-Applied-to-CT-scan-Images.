import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras import *
import os
import PIL
import cv2
import pathlib
import pandas as pd
import matplotlib.image as mpimg
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.ops.numpy_ops import np_utils
from sklearn.metrics import *
import seaborn as sns
BATCH_SIZE = 62
IMAGE_SIZE = 256
EPOCHS=40
CHANNELS=3 
dataset = tf.keras.preprocessing.image_dataset_from_directory("/kaggle/input/all-images/train",
 seed=123,
 shuffle=True,
 image_size=(IMAGE_SIZE,IMAGE_SIZE),
 batch_size=BATCH_SIZE
) 
class_names = dataset.class_names 
len(dataset) 
for image_batch, label_batch in dataset.take(1):
 print(image_batch.shape)
 print(image_batch[1])
 print(label_batch.numpy())
class_names
plt.figure(figsize=(15, 15))
for image_batch, labels_batch in dataset.take(1):
 for i in range(BATCH_SIZE):
 ax = plt.subplot(8, 8, i + 1)
 plt.imshow(image_batch[i].numpy().astype("uint8"))
 plt.title(class_names[labels_batch[i]])
 plt.axis("off")
 def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, 
shuffle_size=10000):
 assert (train_split + test_split + val_split) == 1
 ds_size = len(ds)
 if shuffle:
 ds = ds.shuffle(shuffle_size, seed=12)
 train_size = int(train_split * ds_size)
 val_size = int(val_split * ds_size)
 train_ds = ds.take(train_size) 
 val_ds = ds.skip(train_size).take(val_size)
 test_ds = ds.skip(train_size).skip(val_size)
 # Autotune all the 3 datasets 
 train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
 val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
 test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
return train_ds, val_ds, test_ds
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
resize_and_rescale = tf.keras.Sequential([
 tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
 tf.keras.layers.Rescaling(1./255),
])
data_augmentation = tf.keras.Sequential([
 tf.keras.layers.RandomFlip("horizontal_and_vertical"),
 tf.keras.layers.RandomRotation(0.2),
])
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 9
cnn_model = models.Sequential([
 resize_and_rescale,
 # data_augmentation,
 layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(64, (3, 3), activation='relu'),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(64, (3, 3), activation='relu'),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(64, (3, 3), activation='relu'),
 layers.MaxPooling2D((2, 2)),
 layers.Flatten(),
 layers.Dense(64, activation='relu'),
 layers.Dense(n_classes, activation='softmax'),
])
cnn_model.build(input_shape=input_shape)
cnn_model.compile(
 optimizer='adam',
 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
 metrics=['accuracy']
)
cnn_model.summary()
cnn_history = cnn_model.fit(
 train_ds,
 batch_size=BATCH_SIZE,
 validation_data=val_ds,
 verbose=1,
 epochs=EPOCHS,
)
cnn_model.evaluate(test_ds)
cnn_acc = cnn_history.history['accuracy']
cnn_loss = cnn_history.history['loss']
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), cnn_acc, label=' Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), cnn_loss, label=' Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()
image_path = "/kaggle/input/all-images/train/adenocarcinoma/000000 (6).png"
image = preprocessing.image.load_img(image_path)
image_array = preprocessing.image.img_to_array(image)
scaled_img = np.expand_dims(image_array, axis=0)
Image pred = cnn_model.predict(scaled_img)
output = class_names[np.argmax(pred)]
print(output)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
resnet_model = ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
 include_top=False,
 weights='imagenet')
resnet_model.trainable = False
resnet_model_full = models.Sequential([
 resnet_model, # Base ResNet50 model
 layers.Flatten(), 
 layers.Dense(128, activation='relu'), 
 layers.Dropout(0.5), 
 layers.Dense(n_classes, activation='softmax') 
])
resnet_model_full.compile(
 optimizer=Adam(),
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
)
resnet_model_full.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
resnet_history = resnet_model_full.fit(
 train_ds,
 validation_data=val_ds,
 epochs=EPOCHS,
 callbacks=[early_stopping]
)
resnet_test_loss, resnet_test_accuracy = resnet_model_full.evaluate(test_ds)
print(f"Test Accuracy of ResNet Model: {resnet_test_accuracy * 100:.2f}%")
resnet_acc = resnet_history.history['accuracy']
resnet_loss = resnet_history.history['loss']
epochs_range = range(len(resnet_acc))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, resnet_acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy (ResNet)')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, resnet_loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss (ResNet)')
from sklearn.metrics import confusion_matrix
resnet_predictions = resnet_model_full.predict(test_ds)
resnet_pred_classes = np.argmax(resnet_predictions, axis=1)
true_labels = np.concatenate([y for x, y in test_ds], axis=0)
class_report = classification_report(true_labels, resnet_pred_classes, target_names=class_names)
#print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
zfnet_model = models.Sequential([
 resize_and_rescale, 
 layers.Conv2D(96, kernel_size=(7, 7), strides=(2, 2), activation='relu', input_shape=input_shape),
 layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
 layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='same'),
 layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
 layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
 layers.Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
 layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
 layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
 layers.Flatten(),
 layers.Dense(4096, activation='relu'),
 layers.Dropout(0.5),
 layers.Dense(4096, activation='relu'),
 layers.Dropout(0.5),
 layers.Dense(n_classes, activation='softmax')
])
zfnet_model.compile(
 optimizer='adam',
 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
 metrics=['accuracy']
)
zfnet_history = zfnet_model.fit(
 train_ds,
 batch_size=BATCH_SIZE,
 validation_data=val_ds,
 verbose=1,
 epochs=EPOCHS,
)
zfnet_acc = zfnet_history.history['accuracy']
zfnet_loss = zfnet_history.history['loss']
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(range(EPOCHS), cnn_acc, label='CNN Accuracy', color='blue')
plt.plot(range(EPOCHS), zfnet_acc, label='ZFNet Accuracy', color='orange')
plt.legend(loc='lower right')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.subplot(2, 1, 2)
plt.plot(range(EPOCHS), cnn_loss, label='CNN Loss', color='blue')
plt.plot(range(EPOCHS), zfnet_loss, label='ZFNet Loss', color='orange')
plt.legend(loc='upper right')
plt.title('Model Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()
predictions = np.argmax(zfnet_model.predict(test_ds), axis=1)
true_labels = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
print("ZFNet Classification Report:")
print(classification_report(true_labels, predictions, target_names=class_names))
conf_matrix = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
yticklabels=class_names)
plt.title('ZFNet Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
