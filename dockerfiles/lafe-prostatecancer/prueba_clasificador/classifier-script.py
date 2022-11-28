import os
import zipfile

import argparse
import time

import numpy as np
import tensorflow as tf
from datetime import datetime
from functools import wraps
from tensorflow import keras
from tensorflow.keras import layers

def print_timing(func):
    '''
    create a timing decorator function
    use
    @print_timing
    just above the function you want to time
    '''
    @wraps(func)  # improves debugging
    def wrapper(*arg):
        date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'[{date_time}]')
        start = time.perf_counter()  # needs python3.3 or higher
        result = func(*arg)
        end = time.perf_counter()
        fs = '{} took {:.3f} seconds'
        print(fs.format(func.__name__, (end - start)))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return result
    return wrapper

def parse_arguments():
    """parse_arguments

    Read arguments and return appropriate values.
    """
    parser = argparse.ArgumentParser(description='Train the Prostate Cancer model')
    #parser.add_argument('--model_summary', help='Shows model summary')
    parser.add_argument('--ckpt_dir', help='Directory to save checkpoints')
    #parser.add_argument('--ckpt', action='store_true', help='Enable checkpointing')
    parser.add_argument('--ES', action='store_true', help='Enable EarlyStopping')
    parser.add_argument('--TB', action='store_true', help='Enable TensorBoard')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size (default: 1)')
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of training epochs (default: 1)')
    parser.add_argument('--learning_rate', default=1e-2, type=float,
                        help='Learning rate')
    parser.add_argument("-s", "--save", type=str, default="my_model.h5",
                        help="path to save the model in HD5 format")

    print("Parsing arguments...")
    args = parser.parse_args()

    batch_size = args.batch_size
    print("Batch size set to: " + str(batch_size))

    epochs = args.epochs
    print("Num. epochs set to: " + str(epochs))

    learning_rate = args.learning_rate
    print("Learning rate set to: " + str(learning_rate))

    path_model = args.save
    print("Path to save the model: " + path_model)

    checkpoint_directory = None
    last_epoch = None
    if args.ckpt_dir:
        if os.path.isdir(args.ckpt_dir):
            checkpoint_directory = args.ckpt_dir
            print("Checkpoint directory set to: " + checkpoint_directory)
            last_epoch = get_last_epoch(checkpoint_directory)
            if (last_epoch):
                print("Last epoch: "+  str(last_epoch))
        else:
            print("Checkpoint directory doesn't exist! Continuing without checkpointing...")

    return batch_size, epochs, learning_rate, path_model, checkpoint_directory, last_epoch, args.ES, args.TB

def get_last_epoch(checkpoint_directory):
    """get_last_epoch
    
    Get last epoch that has been executed and checkpointed"""
    epochs = []
    if checkpoint_directory:
        for i in os.listdir(checkpoint_directory):
            try:
                epochs.append(int(i[:-5].split('-')[1]))
            except:
                pass
    return max(epochs) if (epochs) else None

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

@print_timing
def fit(model, train_dataset, validation_dataset, epochs, callbacks):
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=callbacks
    )
    
def run():
    batch_size, epochs, learning_rate, path_model, checkpoint_directory, last_epoch, early_stopping, tensorboard = parse_arguments()
    
    x_train = np.load('data/train/image_array_train.npz')['arr_0']
    x_val = np.load('data/validation/image_array_val.npz')['arr_0']

    y_train = np.load('labels_array_train.npz', allow_pickle=True)['arr_0']
    y_val = np.load('labels_array_val.npz', allow_pickle=True)['arr_0']

    x_train = x_train.astype(int)
    y_train = y_train.astype(int)
    x_val = x_val.astype(int)
    y_val = y_val.astype(int)

    x_train = samplewise_intensity_normalization(x_train)
    x_val = samplewise_intensity_normalization(x_val)

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
    callbacks = []
    
    if tensorboard:
        log_dir = "/tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    
    if checkpoint_directory:
        model_checkpoint_epoch_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_directory+"/ckpt-{epoch:03d}.hdf5",
            save_weights_only=False,
            monitor='val_accuracy',
            mode='auto',
            save_best_only=False,
            save_freq="epoch")
        callbacks.append(model_checkpoint_epoch_callback)
        model_checkpoint_best_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_directory+"/best.hdf5",
            save_weights_only=False,
            monitor='val_accuracy',
            mode='auto',
            save_best_only=True,
            save_freq="epoch")
        callbacks.append(model_checkpoint_best_callback)

        if last_epoch:
            model.load_weights(checkpoint_directory+f"/ckpt-{last_epoch:03d}.hdf5")
        else:
            last_epoch = 0
    else:
        last_epoch = 0

    if early_stopping:
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
        callbacks.append(early_stopping_cb)

    # Train the model, doing validation at the end of each epoch
    fit(model,train_dataset, validation_dataset, epochs, callbacks)
    
if __name__ == "__main__":
    run()
