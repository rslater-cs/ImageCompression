# syntax=docker/dockerfile:1

# Replace with specific OS
FROM continuumio/miniconda3

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda --version

COPY ubuntu_env.yml .
# RUN conda env create -f ubuntu_env.yml

# RUN conda init ml_compression

# RUN conda activate ml_compression

RUN cat ~/.bashrc && \
    conda init bash && \
    cat ~/.bashrc && \
    conda env create -f ubuntu_env.yml && \
    conda activate ml_compression

COPY data_loading .
COPY data_processing .
COPY model_analyser .
COPY models .
COPY session.py .
COPY train.py .

ENTRYPOINT ["python", "session.py"]
# --model swin --epochs 5 --batch 128