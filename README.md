# Enviroment Setup
To setup my code on windows (note that only the training code works on windows):

**cd <project_path>**

**conda create env -f enviroment/anaconda_enviroments/ml_compression_enviroment.yml**


To setup my code on ubuntu (this works for both training and testing):

**cd <project_path>**

**conda create env -f enviroment/anaconda_enviroments/ubuntu_env.yml**

# Training
To run a training session (for one epoch):

-s is the directory to save the model

-i is the path to the imagenet dataset

-e is the number of epochs

-b is the batch size

-m is the embedding dimension

-t is the transfer dimension

-w is the window size 

-d is the depth of the network

**python session.py -s ./saved_models -i <imagenet_path> -e 1 -b 8 -m 24 -t 16 -w 2 -d 3**

# Testing
To test the compression (for one image):

-m is the model directory

-s is a random seed 

-i is the path to the imagenet dataset

-n is the number of images to be compressed

**python test_compression.py -m ./saved_models/<model_name> -i <imagenet_path> -s 55 -n 1**

To test quantisation:

-m is the model directory

-i is the path to the imagenet dataset


**python test_quantisation.py -m ./saved_model/<model_name> -i <imagenet_path>**