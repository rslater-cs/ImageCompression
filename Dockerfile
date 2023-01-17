# syntax=docker/dockerfile:1

# Replace with specific OS
FROM continuumio/miniconda3

WORKDIR /app

RUN conda --version

COPY ubuntu_env.yml .
RUN conda env create -f ubuntu_env.yml

RUN echo "CHECKPOINT"

SHELL ["conda", "run", "-n", "ml_compression", "/bin/bash", "-c"]

RUN echo "Make sure enviroment is installed:"
RUN python -c "import torch"

RUN mkdir ./saved_models

COPY data ./data

COPY data_loading ./data_loading
COPY data_processing ./data_processing
COPY model_analyser ./model_analyser
COPY models ./models
COPY session.py .
COPY train.py .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ml_compression", "python", "session.py"]

# To run do 
# docker run -it --rm --gpus all <docker_name> --model swin --epochs <epochs> --batch <batch_size>