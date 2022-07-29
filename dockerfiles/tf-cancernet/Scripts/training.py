from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from pyimagesearch.cancernet import CancerNet
from pyimagesearch import config
from pyimagesearch import paths
import argparse
import os

def parse_arguments():            
    parser = argparse.ArgumentParser(description='Train the Cancernet model')
    #parser.add_argument('--model_summary', help='Shows model summary')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size (default: 128)')
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of training epochs (default: 10)')
    # parser.add_argument('--ckpt_dir', 
    #                     help='Directory to save checkpoints')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='Learning rate')
    parser.add_argument("-s", "--save", type=str, default="my_model.h5",
	help="path to save the model in HD5 format")

    print("Parsing arguments...")
    args = parser.parse_args()

    batch_size = args.batch_size
    print("Batch size set to: " + str(batch_size))

    epochs = args.epochs
    print("Num. epochs set to: " + str(epochs))

    lr = args.lr
    print("Learning rate set to: " + str(lr))

    path_model = args.save
    print("Path to save the model: " + path_model)

    # checkpoint_directory = None
    # last_epoch = None
    # if args.ckpt_dir:
    #     if os.path.isdir(args.ckpt_dir):
    #         checkpoint_directory = args.ckpt_dir
    #         print("Checkpoint directory set to: " + checkpoint_directory)
    #         last_epoch = get_last_epoch(checkpoint_directory)
    #         if (last_epoch):
    #             print("Last epoch: "+  str(last_epoch))
    #     else:
    #         print("Checkpoint directory doesn't exist! Continuing without checkpointing...")

    return batch_size, epochs, lr, path_model

# construct the argument parser and parse the arguments

batch_size, epochs, lr, path_model = parse_arguments()
# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = epochs
INIT_LR = lr
BS = batch_size

# Determine the total number of image paths in training and validation directories
trainPaths = list(paths.list_images(config.TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(config.VAL_PATH)))

# calculate the total number of training images in each class and initialize a dictionary to store the class weights
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = dict()

# Loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]

# Initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

# initialize the validation data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize our CancerNet model and compile it
model = CancerNet.build(width=48, height=48, depth=3,
	classes=2)

opt = Adagrad(learning_rate=INIT_LR, decay=INIT_LR / NUM_EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# fit the model
H = model.fit(
	x=trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	class_weight=classWeight,
	epochs=NUM_EPOCHS)

model.summary()

model.save(path_model)