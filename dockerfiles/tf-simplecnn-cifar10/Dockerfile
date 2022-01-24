# Dockerfile to create a container for testing Tensorflow Models: Simple CNN to classifify CIFAR-10
#
# 20211119 - Estibaliz Parcero

# https://hub.docker.com/r/tensorflow/tensorflow
FROM tensorflow/tensorflow:2.6.0-gpu

RUN python3 -m pip install --upgrade pip

RUN pip install matplotlib

COPY cnn_cifar10.py cnn_cifar10.py
COPY 20220124_test.sh 20220124_test.sh
RUN chmod +x 20220124_test.sh

CMD exec ./20220124_test.sh