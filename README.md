# RGB-D-Plant-Segmentation
The objective of this code repo is to implement "instance" segmentation on RGB-Depth images taken from 1080p stereo images, such as Stereo Labs products. 

Under the directions of Dr. Zhaodan Kong & Dr. Mohsen Mesgaran, of UC Davis Department of Mechanical & Aerospace Engineering and Plant Sciences respectively.

## Synopsis
The code being implemented in this library is attemtping to perform Image Segmentation on the Kaggle "Carvana" Data set from scratch. The rational behind attemtping image segmentation on such a well studied dataset has several motivations. 

1. Comparision against known solutions, enables baseline performance to be evaluated before the UNet Model is adapted RGB-D performance 
2. Troubleshooting problems with code implementations of NN models, training loops, or evalutation of metrics..etc is easier in a 1-to-1 comparison between existing code which I can reference for assistance.

## Usage & Dependencies 
Scripts were created on a system utilizing Python 3.7 with CUDA 10.2 on GTX 1080Ti Nvidia GPU. 

* [Pytorch](https://pytorch.org/) 
* [Pytorch Lighting](https://pytorch-lightning.readthedocs.io/en/latest/)

## Code Structure 

**Pytorch** 
Pytorch offered a great framework within which to learn the nitty-gritty details of machine learning, while providing an intuitive and comfortable Python environment which directly enables the use of tools native to Python such as debugging, unlike Tensor Flow. With its built in code library for building custom Nerual Networks, using Pytorch has been a very educational experience in learnging how to model and train machine learning code from the ground up. It supports learning about ML and the processes required to make it functional and successful, without abstracting away so much detail from the user.

**Pytorch Lightning**
While Pytorch is very hands on, the necessity to code tedious looping, logging, and status operations, all manually, becomes an cumbersome especially when tuning hyper-parameters or adjusting methods (such as loss or optimization functions) to obtain the best performance. Pytorch Lightning which is a lightweight but incredibly useful libray (built completely from Pytorch code base) whose objective is to simplify process of creating Pytorch models by abstracting boilerplate (training loops, validation, metrics, logging...etc) simply by refactoring ones code into predetermined functions defined in the Pytorch Lightning Library. This not only has the effect of making training Pytorch models more "Keras-Like" it also has the benefit of being 100% compatible with Pytorch, since Lightning is written ontop of Pytorch and is merely adding structuring Pytorch coding. The end result is a lightweight enjoyable coding using the main concepts of Pytorch but removing the headaches of managing the large number of details previously required to be performed manually my Pytorch. 



Inline-style: 
![alt text](https://github.com/JonnyD1117/RGB-D-Plant-Segmentation/blob/master/ZED.jpg "Logo Title Text 1")

## Training: 

## Validation: 

## Metrics: 

## Performance:

