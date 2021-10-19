# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./pictures/network_image.png "Model Visualization"
[image2]: ./pictures/project_result.gif "Model Visualization"


![alt text][image2]


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with variable filter sizes between 8x8 and 2x2 and depths between 32 and 128 (model.py lines 65-71) 

The model includes RELU layers to introduce nonlinearity (e.g. code line 66), and the data is normalized in the model using a Keras lambda layer (code line 63), image is cropped by Keras cropping layer (code line 64). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 74 and 77). 

The model was trained and validated on the data set which is already prepared by Udacity because of the connection issues that I have, I could not collect new data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen as already provided by the Udacity due to internet connection issues that I have at the moment, I could not collect any additional data. Because I could not drive the car manually in simulator. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture that achieves practically good steering commands.

My first step was to use a convolution neural network model similar to the Nvidia Model I thought this model might be appropriate because Nvidia used this network for vehicle steering purposes.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training and validation sets. 

The final step was to run the simulator to see how well the car was driving around track one. Driving behavior was good and vehicle was managed to keep on the track.
The vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture

![alt text][image1]

#### 3. Training Process

I used training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
