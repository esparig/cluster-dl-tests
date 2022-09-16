"""training.py

Python script to train the cancernet model.
"""

import argparse
import time
import os
from datetime import datetime
from functools import wraps

import tensorflow as tf

from pyimagesearch.cancernet import CancerNet
from pyimagesearch import config
from pyimagesearch import paths


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
    parser = argparse.ArgumentParser(description='Train the Cancernet model')
    #parser.add_argument('--model_summary', help='Shows model summary')
    parser.add_argument('--ckpt', action='store_true', help='Enable checkpointing')
    parser.add_argument('--ES', action='store_true', help='Enable EarlyStopping')
    parser.add_argument('--TB', action='store_true', help='Enable TensorBoard')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size (default: 128)')
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of training epochs (default: 10)')
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
    if args.ckpt:
        checkpoint_directory = config.CKPY_PATH
        if os.path.isdir(checkpoint_directory):
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

def get_train_labels():
    """get_train_labels

    Get categorical train labels from paths"""

    train_paths = list(paths.list_images(config.TRAIN_PATH))
    train_labels = tf.keras.utils.to_categorical([int(p.split(os.path.sep)[-2]) for p in train_paths])

    return train_labels, len(train_paths)

def get_class_weight(train_labels):
    """get_class_weight

    Get class weight from training data.
    """
    class_totals = train_labels.sum(axis=0)
    class_max = class_totals.max()
    class_weight = {k : class_max/class_totals[k] for k in range(len(class_totals))}

    return class_weight

@print_timing
def fit_cancernet(model, training_gen, batch_size, val_gen, callbacks, class_weight, epochs, last_epoch):
    """fit_cancernet
    
    Train the model."""
    
    history = model.fit(
        x=training_gen,
        batch_size = batch_size,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight,
        epochs=epochs,
        initial_epoch=last_epoch)
    
    return history


def run():
    """run

    Train model according to parsed arguments."""

    # construct the argument parser and parse the arguments
    batch_size, epochs, learning_rate, path_model, checkpoint_directory, last_epoch, earlystopping, tensorboard = parse_arguments()

    # Determine the total number of image paths in validation directories
    total_val = len(list(paths.list_images(config.VAL_PATH)))

    # calculate the total number of training images in each class
    train_labels, total_train = get_train_labels()

    class_weight = get_class_weight(train_labels)

    # Initialize the training data augmentation object
    training_data_augmentator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest")

    # initialize the validation data augmentation object
    val_data_augmentator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)

    # initialize the training generator
    training_gen = training_data_augmentator.flow_from_directory(
        config.TRAIN_PATH,
        class_mode="categorical",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size)

    # initialize the validation generator
    val_gen = val_data_augmentator.flow_from_directory(
        config.VAL_PATH,
        class_mode="categorical",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size)

    # initialize the test generator
    test_gen = val_data_augmentator.flow_from_directory(
        config.TEST_PATH,
        class_mode="categorical",
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size)

    # initialize our CancerNet model and compile it
    model = CancerNet.build(width=48, height=48, depth=3, classes=2)
    model.summary()

    model.save(path_model)

    callbacks =  []
    
    if tensorboard:
        log_dir = config.TENSORBOARD_PATH +"/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    
    if earlystopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=3,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=False)
        )
    
    opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, decay=learning_rate/epochs) if epochs > 0 else "RMSprop"

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    if checkpoint_directory:
        model_checkpoint_epoch_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_directory+"/ckpt-{epoch:03d}.hdf5",
            save_weights_only=False,
            monitor='val_accuracy',
            mode='auto',
            save_best_only=False,
            save_freq="epoch")
        model_checkpoint_best_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_directory+"/best.hdf5",
            save_weights_only=False,
            monitor='val_accuracy',
            mode='auto',
            save_best_only=True,
            save_freq="epoch")
        callbacks = [model_checkpoint_epoch_callback, model_checkpoint_best_callback]

        if last_epoch:
            model.load_weights(checkpoint_directory+f"/ckpt-{last_epoch:03d}.hdf5")
        else:
            last_epoch = 0
    else:
        last_epoch = 0
        
    # fit the model
    history = fit_cancernet(model, training_gen, batch_size, val_gen, callbacks, class_weight, epochs, last_epoch)

    results = model.evaluate(x=test_gen)
    print("Evaluation of the model (test loss, test acc): ", results)


if __name__ == "__main__":
    run()
