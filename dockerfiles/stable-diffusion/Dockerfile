# Dockerfile to create a container for Stable Diffusion
# Instructions in https://github.com/cmdr2/stable-diffusion-ui
# 20220907 - Estibaliz Parcero

FROM ubuntu:20.04

ADD https://github.com/cmdr2/stable-diffusion-ui/releases/download/v2.05/stable-diffusion-ui-linux.tar.xz /

RUN cd / && tar -xf stable-diffusion-ui-linux.tar.xz  

# RUN cd stable-diffusion-ui-linux && ./start.sh

# How to run this docker container: 
# docker run -it esparig/stable-diffusion ./stable-diffusion-ui-linux/start.sh