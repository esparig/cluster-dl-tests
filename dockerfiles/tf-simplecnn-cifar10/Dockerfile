# Dockerfile to create a container for testing Tensorflow Models: Simple CNN to classifify CIFAR-10
#
# 20211119 - Estibaliz Parcero

# https://hub.docker.com/r/tensorflow/tensorflow
FROM tensorflow/tensorflow:2.6.0-gpu

RUN python3 -m pip install --upgrade pip

RUN pip install matplotlib

COPY cnn_simple.py cnn_simple.py 

CMD ["python3", "cnn_simple.py", "--batch_size=64", "--epochs=25", "--ckpt_dir=/data/ckpt"]
