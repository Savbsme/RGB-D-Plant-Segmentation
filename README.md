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

As a newcomer to ML, Pytorch offered a great framework within which to learn the nitty-gritty details of machine learning, while providing an intuitive and comfortable Python environment which directly enables the use of tools native to Python such as debugging, unlike Tensor Flow. In addition to learning and implementing the entire training loop manually (as shown in standard Pytorch examples/docs), I came across "Pytorch Lightning" which is a libray (built completely from Pytorch code base) which simplifies the process of implementing models via "Keras-like" and significantly reduces the amount of time required to train and tune models with a simple refactoring of default Pytorch code. 





## Code Structure 

Inline-style: 
![alt text](https://github.com/JonnyD1117/RGB-D-Plant-Segmentation/blob/master/ZED.jpg "Logo Title Text 1")

## Training: 

## Validation: 

## Metrics: 

## Performance:

