# Dockerfile to create a container for training a small network for MNIST
# recognition using Tensorflow
#
# 20210505 - Estibaliz Parcero

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install keras
RUN pip3 install --upgrade tensorflow-gpu==1.10.0

COPY ../simple-mnist.py /opt/test-gpu-training/simple-mnist.py
