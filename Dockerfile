# syntax=docker/dockerfile:1

# Replace with specific OS
FROM ubuntu:22.04

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

COPY ml_compression_enviroment.yml .
RUN conda env create -f ml_compression_enviroment.yml

RUN conda activate ml_compression

COPY data_loading .
COPY data_processing .
COPY model_analyser .
COPY models .
COPY session.py .
COPY train.py .

ENTRYPOINT ["python", "session.py"]
# --model swin --epochs 5 --batch 128