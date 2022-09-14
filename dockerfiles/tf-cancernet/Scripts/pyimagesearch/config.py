# import the necessary packages
import os
# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "/data/breast-histopathology/datasets/orig"
#ORIG_INPUT_DATASET = "/home/esparig/projects/data/cancernet"
# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "/data/breast-histopathology/datasets/idc"
#BASE_PATH = "/home/esparig/projects/data/cancernet/idc"
# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
# define the amount of data that will be used training
TRAIN_SPLIT = 0.8
# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.1
CKPY_PATH = "/data/breast-histopathology/ckpt"
#CKPY_PATH = "/tmp/ckpt"