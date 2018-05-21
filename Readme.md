# Image Similarity Detection using Resnet50

## Introduction
Given a batch of images, the program tries to find similarity between images using Resnet50 based feature vector extraction. 

## Usage
``python kreas_resnet50.py`` will compare all the images present in ``images`` folder with each other and provide the most similar image for every image. 

## Pre-Requisites
* Download [Anaconda](https://www.anaconda.com/download/#linux)
* Make the downloaded shell script executable and install
* ``conda -V`` to check that installation was successfull. 
* ``conda update conda`` and ``conda update anaconda``
* ``conda update scikit-learn``
* ``conda install theano``
* ``conda install -c conda-forge tensorflow``
* ``pip install keras``
* ``export MKL_THREADING_LAYER=GNU``

Note: More Descriptive Instructions On [Installation](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)
