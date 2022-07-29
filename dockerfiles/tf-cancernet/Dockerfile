# Dockerfile to create a container for training CancerNet
# Instructions in https://pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/
#
# 20220729 - Estibaliz Parcero

# https://hub.docker.com/r/tensorflow/tensorflow
FROM tensorflow/tensorflow:2.6.0-gpu

RUN python3 -m pip install --upgrade pip

RUN pip install imutils numpy opencv-python-headless Pillow SciPy

COPY  Scripts .

# CMD ["python", "training.py"]
