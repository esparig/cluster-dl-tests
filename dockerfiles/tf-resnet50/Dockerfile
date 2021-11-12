# Dockerfile to create a container for testing Tensorflow Models: Resnet 50 using Imagenet
#
# 20211006 - Estibaliz Parcero

# https://hub.docker.com/r/tensorflow/tensorflow
FROM tensorflow/tensorflow:2.6.0-gpu

RUN python3 -m pip install --upgrade pip

# https://github.com/tensorflow/models/tree/master/official
RUN pip install tf-models-official==2.6.0

COPY custom.yaml /data/imagenet/experiment/imagenet_resnet50_gpu_custom.yaml

ENTRYPOINT ["python3", "/usr/local/lib/python3.6/dist-packages/official/vision/beta/train.py", \
"--experiment=resnet_imagenet", "--config_file=/data/imagenet/experiment/imagenet_resnet50_gpu_custom.yaml", \
"--mode=train_and_eval", "--model_dir=/data/model_checkpoints", "--params_override='runtime.num_gpus=1'"]
