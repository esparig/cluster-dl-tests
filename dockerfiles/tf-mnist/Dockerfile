# Dockerfile to create a container for training a small network for MNIST
# recognition using Tensorflow
#
# 20211102 - Estibaliz Parcero

FROM tensorflow/tensorflow:latest-gpu

#COPY scripts/simple-mnist.py /opt/test-gpu-training/simple-mnist.py
#CMD [ "python", "/opt/test-gpu-training/simple-mnist.py" ]

COPY scripts/mnist-conv.py /opt/test-gpu-training/mnist-conv.py

CMD [ "python", "/opt/test-gpu-training/mnist-conv.py" ]
