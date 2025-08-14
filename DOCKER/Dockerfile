FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

## GPU-specific configurations
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

## for apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive 
#set defaut shell to bash
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash 
ENV SHELL=/bin/bash

# Install basis C library, can be needed to build other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    wget \
    autoconf \
    tar \
    unzip \
    # gcc-multilib \
    automake \
    # nvidia-utils-535 \   
    sudo &&\
    rm -rf /var/lib/apt/lists/*

# install gdal
RUN add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update

# Install python env
RUN apt-get -y install python3 python-is-python3 python3-pip cmake unzip && \
    python3 -m pip install --upgrade pip && \
    pip install onnxoptimizer==0.3.13

FROM continuumio/miniconda3:24.7.1-0 
RUN conda install -c conda-forge gdal=3.6 rasterio=1.4.3 onnx==1.15.0 

# Install python requirements
COPY ./requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Setup user, avoid root user
RUN apt-get update && apt-get -y install gosu
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

