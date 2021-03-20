# **Behavioral Cloning** 


<div align="center">
    <img src="https://github.com/Ventu012/P4_BehavioralCloning/blob/main/report_images/video_gif.gif" width="500" />
</div>

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
* model.py (or train.ipynb) containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model/model.h5 containing a trained convolution neural network 
* report.md or writeup_report.pdf summarizing the results
* video_output_autonomous/video.mp4 - A video recording of your vehicle driving autonomously at least one lap around the track.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py (or train.ipynb) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the convolution neural network model used in the NVIDIA's End to End Learning for Self-Driving Cars paper <br>(Reference: https://arxiv.org/pdf/1604.07316v1.pdf). <br>
<div align="center">
    <img src="https://github.com/Ventu012/P4_BehavioralCloning/blob/main/report_images/cnn.png" width="500" />
</div>
<br>
The model consists of a convolution neural network with 3 layer with kernel size of 5x5 and 2 with kernel size of 3x3, the depths varies between 24 and 64 (model.py lines 70-80). <br>

The input images are first cropped to remove irrelevant parts of the images, the hood and the sky (code line 64), and then normalized in the model using a Keras lambda layer (code line 67). <br>

The model contains dropout layers in order to reduce overfitting (model.py lines 75). <br>

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap in each direction of track one using center lane driving. <br>
Here is an example image of center lane driving: <br>
<div align="center">
    <img src="https://github.com/Ventu012/P4_BehavioralCloning/blob/main/report_images/center.jpg" width="500" />
</div>
<br>
I then recorded short rides of places with different patterns and short rides of the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when approaching the edge of the road. <br>
These images show what a recovery looks like:
<div align="center">
    <img src="https://github.com/Ventu012/P4_BehavioralCloning/blob/main/report_images/recover_1.jpg" width="500" />
</div>
<div align="center">
    <img src="https://github.com/Ventu012/P4_BehavioralCloning/blob/main/report_images/recover_2.jpg" width="500" />
</div>
<div align="center">
    <img src="https://github.com/Ventu012/P4_BehavioralCloning/blob/main/report_images/recover_3.jpg" width="500" />
</div>
<br>
To augment the dataset, I also flipped images and angles. <br>

I used this data for training (0.8) and validation (0.2). The validation set helped determine if the model was over or under fitting and the correct amount of Epochs for the training process. <br>

#### 3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually (model.py line 84). <br>
I trained the model several times to choose the best number of Epochs (5-10-15 were tested). At the end 10 Epochs was the best number of Epoch, after 10 Epoch the loss of validation of the model increases.  <br>

In the end I run the simulator to see how well the car was driving around track one. There are a few spots where the vehicle deviate from the center of the track but quickly returns to the center line of the track. <br>

The result is that the vehicle is able to drive autonomously around the track without leaving the road. <br>