import os
import argparse
import tensorflow as tf

from tensorflow.keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt

def get_last_epoch(checkpoint_path):
    epochs = []
    for i in os.listdir(checkpoint_path):
        try:
            epochs.append(int(i[:-5].split('-')[1]))
        except:
            pass
    return max(epochs) if (epochs) else None
            
parser = argparse.ArgumentParser(description='Train a Simple CNN for CIFAR-10 classification')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size (default: 128)')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of training epochs (default: 10)')
parser.add_argument('--ckpt_dir', 
                    help='Directory to save checkpoints')
args = parser.parse_args()

batch_size= args.batch_size
epochs = args.epochs
print("----------------------------------------------")
print("Parsing arguments...")
print("Batch size set to: " + str(batch_size))
print("Num. epochs set to: " + str(epochs))
if args.ckpt_dir:
    checkpoint_directory = args.ckpt_dir
    print("Checkpoint directory set to: " + checkpoint_directory)
    last_epoch = get_last_epoch(checkpoint_directory)
    if (last_epoch):
        print("Last epoch: "+  str(last_epoch))
print("----------------------------------------------")

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
callbacks = []

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

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

if args.ckpt_dir:
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
      
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, initial_epoch=last_epoch,
                    validation_data=(test_images, test_labels),
                    callbacks=callbacks)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
