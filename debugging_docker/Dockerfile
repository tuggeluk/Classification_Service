FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Set environment settings for LANG and LD_LIBRARY_PATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/targets/x86_64-linux/lib/

# Install required packages
RUN apt-get update && apt-get install -y libgtk2.0.0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    wget \
    bzip2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3.6 \
    && conda update conda \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

# Then install ML stuff
RUN pip install opencv-python~=4.2.0 \
    pillow==5.3.0 \
    pandas \
    tensorflow==1.11 \
    tensorflow-gpu==1.11 \
    tqdm \

# Make workdir
RUN mkdir /workspace
WORKDIR /workspace

ENTRYPOINT /bin/bash