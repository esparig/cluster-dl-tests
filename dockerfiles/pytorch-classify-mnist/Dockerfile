# Dockerfile to create a container for training a Classifier of MNIST
# https://medium.com/analytics-vidhya/training-deep-neural-networks-on-a-gpu-with-pytorch-2851ccfb6066
#
# 20221024 - Estibaliz Parcero

# From https://github.com/pytorch/pytorch
FROM pytorch/pytorch:latest

RUN mkdir -p data/MNIST/raw

ADD http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz data/MNIST/raw/train-images-idx3-ubyte.gz
RUN gzip -d data/MNIST/raw/train-images-idx3-ubyte.gz
ADD http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz data/MNIST/raw/train-labels-idx1-ubyte.gz
RUN gzip -d data/MNIST/raw/train-labels-idx1-ubyte.gz
ADD http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz data/MNIST/raw/t10k-images-idx3-ubyte.gz
RUN gzip -d data/MNIST/raw/t10k-images-idx3-ubyte.gz
ADD http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz data/MNIST/raw/t10k-labels-idx1-ubyte.gz
RUN gzip -d data/MNIST/raw/t10k-labels-idx1-ubyte.gz

COPY  classify-mnist.py .
