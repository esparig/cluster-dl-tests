# Dockerfile to create a container for training a small network for MNIST
# recognition using Tensorflow
#
# 20210506 - Estibaliz Parcero

FROM ubuntu:18.04

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get -y install python3-dev python3-pip python3-venv
RUN pip3 install --upgrade pip
RUN pip3 install keras
RUN pip3 install --upgrade tensorflow

# Verify the tensorflow installation
RUN python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

COPY scripts/simple-mnist.py /opt/test-gpu-training/simple-mnist.py
