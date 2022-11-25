import os
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

x_train = np.load('data/train/image_array_train.npz')['arr_0']
x_val = np.load('data/validation/image_array_val.npz')['arr_0']

y_train = np.load('labels_array_train.npz', allow_pickle=True)['arr_0']
y_val = np.load('labels_array_val.npz', allow_pickle=True)['arr_0']

x_train = x_train.astype(int)
y_train = y_train.astype(int)
x_val = x_val.astype(int)
y_val = y_val.astype(int)

def samplewise_intensity_normalization(images):
    for i in range(images.shape[3]):
        img = images[:,:,:,i]
        maxim = np.max(img)
        minim = np.min(img)
        if maxim == 0 and minim == 0:
            images[:,:,:,i] = img
        else:
            images[:,:,:,i] = (img - minim) / (maxim - minim)
    return images

x_train = samplewise_intensity_normalization(x_train)
x_val = samplewise_intensity_normalization(x_val)

def get_model(width=128, height=128, depth=128):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=5, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=128)
model.summary()

x_train = np.transpose(x_train, (3, 0, 1, 2))
x_val = np.transpose(x_val, (3, 0, 1, 2))
print(x_train.shape)
print(len(x_train))
print(y_train.shape)
print(len(y_train))
print(x_val.shape)
print(len(x_val))
print(y_val.shape)
print(len(y_val))

train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 4

train_dataset = (
    train_loader.shuffle(len(x_train))
    .batch(batch_size)
    .prefetch(2)
)

validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .batch(batch_size)
    .prefetch(2)
)

# Compile model.
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
#early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 2
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb],
)
