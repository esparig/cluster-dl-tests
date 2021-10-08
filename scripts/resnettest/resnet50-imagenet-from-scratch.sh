# Dataset preparation

## Download Imagenet2012: https://image-net.org/

export IMAGENET_HOME=<my_imagenet_folder>

## Setup folders
mkdir -p $IMAGENET_HOME/validation
mkdir -p $IMAGENET_HOME/train

## Extract validation and training
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_HOME/validation
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_HOME/train

cd $IMAGENET_HOME/train

for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
  rm $f
done

## Download validation labels file
wget -O $IMAGENET_HOME/synset_labels.txt \
https://raw.githubusercontent.com/tensorflow/models/c7df5a3dde886509fbd1c7b317f76fb876f23506/research/inception/inception/data/imagenet_2012_validation_synset_labels.txt

## Download imagenet_to_gcs script
wget -O $IMAGENET_HOME/imagenet_to_gcs.py \
https://github.com/tensorflow/tpu/blob/8cca0ff35e1d8c6fcd1dfac98978495ff2cadb84/tools/datasets/imagenet_to_gcs.py

## Edit imagenet_to_gcs.py bug: lines 350-351 to regex to catch all files.

## Inside the Tensorflow docker container
### sudo docker run -v /disk2/data:/data --gpus all -it --rm tensorflow/tensorflow:latest-gpu /bin/bash

python3 -m pip install --upgrade pip &&\
pip install tf-models-official &&\
pip install gcloud google-cloud-storage

python3 imagenet_to_gcs.py \ 
--raw_data_dir=$IMAGENET_HOME \ 
--local_scratch_dir=$IMAGENET_HOME/tfrecord \ 
--nogcs_upload

## Move processed files so the final content is as follows:
'''
${DATA_DIR}/train-00000-of-01024
${DATA_DIR}/train-00001-of-01024
 ...
${DATA_DIR}/train-01023-of-01024

${DATA_DIR}/validation-00000-of-00128
S{DATA_DIR}/validation-00001-of-00128
 ...
${DATA_DIR}/validation-00127-of-00128
'''

## Alternative to prepare dataset using tfds
## https://github.com/tensorflow/models/issues/10258#issuecomment-938193707


# Run training: Resnet50 over Imagenet from scratch

## Download experiment yaml
wget -O $EXPERIMENT/imagenet_resnet50_gpu_custom.yaml \
https://raw.githubusercontent.com/tensorflow/models/1fa648a753b877f18ca3a1de9bb921c3f024c11d/official/vision/beta/configs/experiments/image_classification/imagenet_resnet50_gpu.yaml

## Edit experiment file: batch size to 256, folders location...

## Inside the Tensorflow docker container

python3 -m pip install --upgrade pip &&\
pip install tf-models-official

export DATA_DIR=/data/imagenet/imagenet2012/5.1.0/train
export MODEL_DIR=/data/model_checkpoints
export IMAGE_CLASSIFICATION=/usr/local/lib/python3.6/dist-packages/official/vision/beta
export EXPERIMENT=/data/imagenet/experiment

## Check experiment configuration:
cat $EXPERIMENT/imagenet_resnet50_gpu_custom.yaml

python3 $IMAGE_CLASSIFICATION/train.py --experiment=resnet_imagenet --config_file=$EXPERIMENT/imagenet_resnet50_gpu_custom.yaml --mode=train_and_eval --model_dir=$MODEL_DIR --params_override='runtime.num_gpus=1'