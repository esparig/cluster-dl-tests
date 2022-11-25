# Dockerfile to create a container for training Tensorflow Models
#
# 20221125 - Estibaliz Parcero

# https://hub.docker.com/r/tensorflow/tensorflow
FROM tensorflow/tensorflow:2.6.0-gpu

RUN python3 -m pip install --upgrade pip

COPY prueba_clasificador/requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY prueba_clasificador/classifier-script.py script.py