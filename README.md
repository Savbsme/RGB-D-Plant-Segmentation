# RGB-D-Plant-Segmentation
The objective of this code repo is to implement "instance" segmentation on RGB-Depth images taken from 1080p stereo images, such as Stereo Labs products. 

Under the directions of *Dr. Zhaodan Kong* & *Dr. Mohsen Mesgaran*, of UC Davis Department of Mechanical & Aerospace Engineering and Plant Sciences respectively.

## Synopsis
The code being implemented in this library is attemtping to perform Image Segmentation on the Kaggle "Carvana" Data set from scratch. The rational behind attemtping image segmentation on such a well studied dataset has several motivations. 

1. Comparision against known solutions, enables baseline performance to be evaluated before the UNet Model is adapted RGB-D performance 
2. Troubleshooting problems with code implementations of NN models, training loops, or evalutation of metrics..etc is easier in a 1-to-1 comparison between existing code which I can reference for assistance.

## Usage & Dependencies 
Scripts were created on a system utilizing Python 3.7 with CUDA 10.2 on GTX 1080Ti Nvidia GPU. 

* [Pytorch](https://pytorch.org/) 
* [Pytorch Lighting](https://pytorch-lightning.readthedocs.io/en/latest/)

## Code Structure 

**Pytorch** <br>
Pytorch offered a great framework within which to learn the nitty-gritty details of machine learning, while providing an intuitive and comfortable Python environment which directly enables the use of tools native to Python such as debugging, unlike Tensor Flow. With its built in code library for building custom Nerual Networks, using Pytorch has been a very educational experience in learnging how to model and train machine learning code from the ground up. It supports learning about ML and the processes required to make it functional and successful, without abstracting away so much detail from the user.

**Pytorch Lightning**<br>
While Pytorch is very hands on, the necessity to code tedious looping, logging, and status operations, all manually, becomes an cumbersome especially when tuning hyper-parameters or adjusting methods (such as loss or optimization functions) to obtain the best performance. Pytorch Lightning which is a lightweight but incredibly useful libray (built completely from Pytorch code base) whose objective is to simplify process of creating Pytorch models by abstracting boilerplate (training loops, validation, metrics, logging...etc) simply by refactoring ones code into predetermined functions defined in the Pytorch Lightning Library. This not only has the effect of making training Pytorch models more "Keras-Like" it also has the benefit of being 100% compatible with Pytorch, since Lightning is written ontop of Pytorch and is merely adding structuring Pytorch coding. The end result is a lightweight enjoyable coding using the main concepts of Pytorch but removing the headaches of managing the large number of details previously required to be performed manually my Pytorch. 


## UNet Segmentation Model 

Unet is a type of convolutional neural network which gained popularity as the basis for many image-based machine learning tasks. The main premise of any UNet model is that unlike, a straight forward cascade of of CNNs, skip connections are added across layers of the same shape. These connections presever data fidelity in the later layers of the model, which has the benefit of preventing vanishing gradient on "deep" neural networks. The original work can be see in the following link.

[Unet Paper](https://arxiv.org/abs/1505.04597)

From this initial paper, the popularity of Unet has demonstrated how effect the Unet architecture is for image segmentation problems. As this is model is nearly universally known in the ML community, the objective of this research being to perform image segmentation on 4 channel (RGB-Depth) data instead of 3 channel (RGB) data seemed like a great opportunity to have a consistent model architecture whose features can be adjusted to accommidate any number of input channels required for the problem. 

Below is the schematic of how data flows through the Unet structure. 
![Unet Schematic](https://github.com/JonnyD1117/RGB-D-Plant-Segmentation/blob/master/unet.png)


## Dataset & Pre-Processing 
The dataset used for this trial project is the "Kaggle Carvana" data set. This data set contains over 5000 image/ground truth pairs for segmentation training. The data set was manually split into training and validation subsets for training and validation loops respectively. Since the Carvana dataset contains 16 images per car, being veiwed from 360 degrees, the images/mask pairs used for validation were selected as a contiguous grouping, from the end of the dataset. This is done so as to prevent an artificially high validation accuracy which could have otherwise been caused by the model making predictions on images of cars which it previously trained on but merely view from a slightly different perspective. 

Both the training data and validation are constructed using the Pytorch "Dataset" structure and written in seperate files, which are later imported into the main code. These two files are denoted as follows. 

* [CarvanaDS](https://github.com/JonnyD1117/RGB-D-Plant-Segmentation/blob/master/CarvanaDS.py) 
* [ValDS](https://github.com/JonnyD1117/RGB-D-Plant-Segmentation/blob/master/ValDS.py)

Once imported into the main code, these dataset objects are loaded into the training and validations loops using Pytorch Data Loaders, which assist the developer in how data is passed into the model for training, and performs useful tasks such as abstracting away concepts likes batch size, and the number of CPU threads used to preprocess the data before being sent to the model for training. 






Inline-style: 
![alt text](https://github.com/JonnyD1117/RGB-D-Plant-Segmentation/blob/master/ZED.jpg "Logo Title Text 1")

## Training: 
### Loss Function 
### Validation: 
### Segmentation Performance Metrics: 


