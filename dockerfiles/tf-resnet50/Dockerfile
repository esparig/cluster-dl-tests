# Dockerfile to create a container for testing Tensorflow Models: Resnet 50 using Imagenet
#
# 20211006 - Estibaliz Parcero

# https://hub.docker.com/r/tensorflow/tensorflow
FROM tensorflow/tensorflow:2.6.0-gpu

RUN python3 -m pip install --upgrade pip

# https://github.com/tensorflow/models/tree/master/official
RUN pip install tf-models-official==2.6.0

ENV DATA_DIR=/data/imagenet/imagenet2012/5.1.0/train
ENV MODEL_DIR=/data/model_checkpoints
ENV IMAGE_CLASSIFICATION=/usr/local/lib/python3.6/dist-packages/official/vision/beta
ENV EXPERIMENT=/data/imagenet/experiment

COPY custom.yaml /data/imagenet/experiment/imagenet_resnet50_gpu_custom.yaml

CMD ["python3", "${IMAGE_CLASSIFICATION}/train.py --experiment=resnet_imagenet --config_file=${EXPERIMENT}/imagenet_resnet50_gpu_custom.yaml --mode=train_and_eval --model_dir=${MODEL_DIR} --params_override='runtime.num_gpus=1'"]
