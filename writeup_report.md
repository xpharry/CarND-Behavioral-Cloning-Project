# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/model_architecture/lenet.png "LeNet-5 Model Visualization"
[image2]: ./writeup/center_lane/center_2017_08_03_11_27_02_625.jpg "Centel Raw"
[image3]: ./writeup/recovery/center_2017_08_04_18_06_51_725.jpg "Recovery 1"
[image4]: ./writeup/recovery/center_2017_08_04_18_06_54_359.jpg "Recovery 2"
[image5]: ./writeup/recovery/center_2017_08_04_18_06_57_272.jpg "Recovery 3"
[image6]: ./writeup/flip/original.jpg "Normal Image"
[image7]: ./writeup/flip/flipped.jpg "Flipped Image"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 a video recording a complete autonomous driving on track 1
* run2.mp4 a video recording a complete autonomous driving on track 2

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My first working model is simply a LeNet-5 model which consists of 2 convolution neural network with 5x5 filter sizes and depths between 6 and 16 (model.py lines 82 and 84).

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 80). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 88 and 90). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 124). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 97).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Pipeline

#### 1. Initial Model Design

The overall strategy is to generalize the association of the images captured by the cameras mounted in the self-driving car and the steering angles.

I quickly came up with LeNet-5 as my initial model due to the successful application on the previous project. I thought this model might be appropriate because it had shown a considerable performance in image classification tasks. I could hopefully handle well on generalizing the images captured here.

### 2. Data Collecting

To capture good driving behavior, I first recorded three laps on track one using center lane driving, two clockwise and one counterclockwise. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right side of the road back to center so that the vehicle would learn to recover from sticking to the side. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Also, I recorded a lap focusing on driving smoothly around curves where the car were shown more chance to go off the track.

Then I repeated this process on track two in order to get more data points.

### 3. Data Preprocessing

After the collection process, I had 57,207 number of data points. I then preprocessed this data by Gaussian Blurring and converting color format from BGR to RGB.

OpenCV reads images as BGR format, so I need to convert the loaded images into RGB format to be properly used in drive.py.

### 4. Data Augmentation

To augment the data set, I flipped images and angles thinking that this would avoid driving to one side. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

### 5. Testing in Simulator

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.

To improve the driving behavior in these cases, I add in more data focusing on moving around the curves and increase the number of the epochs.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 5. Final Model Architecture

The final model architecture consisted of two convolution layers and three fully connected layers with the following layers and layer sizes.

Here is a visualization of the architecture.

![alt text][image1]

#### 6. Training Process

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

To combat the potential overfitting, I inserted two dropout layers with keep ratio 0.5.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 12 as evidenced by practice. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Result Analysis

### Model Assessment

|       Model        |                Setting                  |   1st Track   |         2nd Track       |
|--------------------|-----------------------------------------|---------------|-------------------------|
| ./model/model01.h5 | center camera only, LetNet-5, 5 epochs  |    Perfect    | Go off the track soon.  |
| ./model/model02.h5 | center camera only, LetNet-5, 20 epochs |    Perfect    | Good. Go off the track one time in a lap. |
| ./model/model03.h5 | three cameras, LetNet-5, 12 epochs      |    Perfect    | Perfect for several laps. Possibly go off the track when it runs in a high speed |

### Bug Fixing

I got relative good performance with LeNet-5 and a few data and I believed it was close to success. But finally I had to spend one week for almost nothing. Because I made a mistake in preprocessing the images and also ran into a wrong direction when debugging.

I happened to see someone mention BGR and RGB only when I noticed that I mistakenly converted the image color format from BGR to YUV. The Nvidia paper actually mentioned YUV format but it is not consistent with what was used in drive.py. That is why I got even worse performance with more data.
