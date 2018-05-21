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
* ```javascript
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
```
* 