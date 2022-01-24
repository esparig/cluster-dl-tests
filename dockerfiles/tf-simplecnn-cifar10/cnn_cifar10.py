import multiprocessing
import os
import argparse
import time
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models, callbacks
from datetime import datetime
from functools import wraps

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

def virtualize_gpus(mem_limit):
    if mem_limit is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Create 2 virtual GPUs with mem_limit memory each
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit),
                     tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
    return logical_gpus

def limit_gpu_memory(mem_limit):

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate mem_limit(MB) of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)])
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

def get_last_epoch(checkpoint_path: str):
    epochs = []
    if checkpoint_path:
        for i in os.listdir(checkpoint_path):
            try:
                epochs.append(int(i[:-5].split('-')[1]))
            except:
                pass
    return max(epochs) if (epochs) else None

def parse_arguments():            
    parser = argparse.ArgumentParser(description='Train a Simple CNN for CIFAR-10 classification')
    parser.add_argument('--model_summary', help='Shows model summary')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size (default: 128)')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of training epochs (default: 10)')
    parser.add_argument('--ckpt_dir', 
                        help='Directory to save checkpoints')
    parser.add_argument('--mem_limit', type=int,
                        help='Memory allocation in GPU')
    parser.add_argument('--multi_gpu', help='Create two virtual GPUs')

    print("Parsing arguments...")
    args = parser.parse_args()

    batch_size = args.batch_size
    print("Batch size set to: " + str(batch_size))

    epochs = args.epochs
    print("Num. epochs set to: " + str(epochs))

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

    mem_limit = None
    if args.mem_limit:
        mem_limit = args.mem_limit
        print(f"Memory limit set to: {mem_limit}")

    return args.model_summary, batch_size, epochs, checkpoint_directory, last_epoch, mem_limit, args.multi_gpu

def cnn_cifar10_model(summary: bool):
    # Creating a sequential model and adding layers to it

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))    # num_classes = 10

    # Checking the model summary
    if summary:
        model.summary()
    return model

def cnn_cifar10_basic_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()
    return model

@print_timing
def cnn_cifar10_fit(model, train_images, train_labels, batch_size, epochs, last_epoch, test_images, test_labels, callbacks, verbose):
    history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, initial_epoch=last_epoch,
                    validation_data=(test_images, test_labels),
                    callbacks=callbacks,
                    verbose=verbose)
    return history

def cnn_cifar10(batch_size: int, epochs: int, checkpoint_directory: str, last_epoch: int, verbose: int):

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    callbacks = []

    model = cnn_cifar10_model(False)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

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
        
    history = cnn_cifar10_fit(model, train_images, train_labels, batch_size, epochs, last_epoch, test_images, test_labels, callbacks, verbose)
    #
    # history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, initial_epoch=last_epoch,
    #                    validation_data=(test_images, test_labels),
    #                    callbacks=callbacks,
    #                    verbose=verbose)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=verbose)

    print(test_acc)

def parallel_run(batch_size: int, epochs: int, device: str, verbose: int):

    logical_gpus = tf.config.list_logical_devices('GPU')
    
    start = time.perf_counter()
    
    p = []
    
    for d in logical_gpus:
        try:
            with tf.device(d):
                print(f'Executing on device {d}')
                p.append(multiprocessing.Process(target=cnn_cifar10), args=(batch_size, epochs, None, None, verbose,))
                p[-1].start()
                p[-1].join()
        except:
            print('Error while executing on device {d}.')
            time.sleep(2)

    finish = time.perf_counter()

    print(f'Parallel run finished in {round(finish - start, 2)} second(s).')
    

def run():
    model_summary, batch_size, epochs, checkpoint_directory, last_epoch, mem_limit, multi_gpu = parse_arguments()
    if model_summary:
        cnn_cifar10_model(True)
    if multi_gpu:
        virtualize_gpus(mem_limit)
        parallel_run()
    else:
        if mem_limit:
            limit_gpu_memory(mem_limit)
        cnn_cifar10(batch_size, epochs, checkpoint_directory, last_epoch, 2)

if __name__ == '__main__':
    run()
