# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

[//]: # (Image References)

[image1]: ./images/results_Step1_imageClasses.png "Image Classes"
[image2]: ./images/results_Step1_classHisto.png "Image Histogram"
[image3]: ./images/results_Step1_imgTransform.png "Image Transformations"
[image4]: ./images/results_Step1_classHistoNew.png "Image New Histogram"
[image5]: ./images/results_Step2_imgPreprocess.png "Image Preprocess"
[image6]: ./images/figure_model_LeNet.png "LeNet Architecture"
[image7]: ./images/figure_modle_LeNet2.png "Modified LeNet Architecture"
[image8]: ./images/inputs_Step3_imgSamples.png "Test Images"
[image9]: ./images/results_Step3_Softmax.png "Test Images Softmax"

#### Sources
"LeNet Architecture": ResearchGate (https://www.researchgate.net/figure/282997080_fig10_Figure-10-Architecture-of-LeNet-5-one-of-the-first-initial-architectures-of-CNN)

---

## Writeup / README

This is the writeup for my Udacity Self-Driving Car Nanodegree Term 1 [Project 2 submission](https://github.com/liangk7/CarND-Term1-Project2/blob/master/Traffic_Sign_Classifier.ipynb) in accordance with the [rubric guidelines](https://review.udacity.com/#!/rubrics/481/view)

---

### Dataset Exploration

#### Dataset Summary
Data is provided for *training* and *testing*. Although *validation* data is also provided, it is not usable for this project. Therefore we must partition the *training* set into both *training* and *validation* data sets.
**Step 1:** Load the data
- `pickle.load` is used to import the data
**Step 2:** Summarize the dataset shapes and their classifications
- `numpy.ndarray.shape` is used to determine data properties
	* Number of training examples = 34799
	* Number of testing examples = 12630
	* Image data shape = (32, 32, 3)
	* Number of classes = 43

#### Exploratory Visualization
With an understanding of the data shapes and classifications, we can visualize the distribution of images per class.
**Step 1:** Show a sample image from each of the classes
- `numpy.genfromtxt` is used to get the sign class names from the provided *signnames.csv* file
- `matplotlib.subplots` is used to visualize grouped subplots of the class image samples
![alt text][image1]
**Step 2:** Plot a histogram to depict the distribution of the images classes
- `matplotlib.hist` is used to generate a histogram of the training dataset
![alt text][image2]

#### Data Generation
From the visualization of the data, we can determine whether or not our dataset contains enough samples to properly train the classification model. In the case that the distribution of our classes is skewed, we can replicate our existing data pool and generate new data using various image processing techniques. 
*NOTE:* one may desire to partition the data into *training* and *validation* sets prior to data generation in order to prevent data spilling (training the model to images similar to validation data).

**Step 1:** Create image adjustment functions to help generate fake data
- `img_affine` is used to pivot an image with respect to 3 points and their parallel relationship to 3 new points
- `img_bright` is used to augment the color values of an image to produce a visually brighter result
- `img_scale` is used to produce a zoomed in/out perspective of an image
- `img_translate` is used to shift an image in a single direction
- `img_TRANSFORM` utilizes all of the above techniques to produce a more distinct image sample
	![alt text][image3]

**Step 2:** For each underrepresented image class, generate new images
- `dat_generate` takes an existing dataset; compares it to the desired threshold; then will generate new data (as needed) using `img_TRANSFORM`
- `dat_partition` will take an existing dataset; separate the dataset per class; partition it into *training* and *validation* sets; then generate data using `dat_generate` according to the `threshold` input
	(for `ratio`=0.2 and `threshold`=1000)
	```
	Shape of X_train: (34799, 32, 32, 3)
	---
	Verifying and partitioning dataset...
	---
	Class 0 data: Training Samples = 144, Validation Samples = 36
	Class 1 data: Training Samples = 1584, Validation Samples = 396
	Class 2 data: Training Samples = 1608, Validation Samples = 402
	...
	---
	TOTAL SAMPLES GENERATED: 187161 Training, 36040 Validation
	---
	Shape of genXtrain: (215000, 32, 32, 3)
	Shape of genXvalid: (43000, 32, 32, 3)
	```

**Step 3 (optional):** Visualize data from the new datasets
- `matplotlib.hist` is used again to show the new distribution of class images
	![alt text][image4]

#### Sources
Transformations: OpenCV-Python Tutorials (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html)

---

### Design and Test a Model Architecture

#### Preprocessing
This step is important for multiple reasons. Firstly, it serves as a method to filter out unnecessary noise in the images (hue, saturation, lighting, damage, etc.). Secondly, it reduces the number or parameters that the model would otherwise need to retain. Thirdly, a consistent processing method allows more flexibility in the type and quality of data that is gathered.
Preprocessing steps of the images include:
- `equalization` is used to spread out the exposure of an image more evenly, occupying 0 to 255 (min to max) values
- `grayscale` is used to convert a color image into grayscale
- `normalization` is used to adjust the values ((x - mu)/stdev) to prevent the propagation of error in the model calculations
- 'preProcess' uses the above functions to process an array of images for model training and prediction evaluation
	![alt text][image5]
Note: I moved this to preclude image transform to reduce the storage cost.

#### Model Architecture
For an image classifier neural network, practice has demonstrated convolutional filters to be paramount in producing accurate results. Convolutional filters generally enhance important features in an image. Accompanying these convolutional filters are rectifier (ReLU) and pooling (max pool) operations. By nulling unimportant image features, this set of operations (used in repeated succession) is commonly employed to produce more distinguishable feature sets. However, this level of accuracy comes with the compromise of memory and processing speed. 
- First Implementation: LeNet
	![alt text][image6]
	
	|  Layer			|  Description					|
	|:-----------------:|:-----------------------------:|
	|  Input			|  32x32x31RGB image			|
	|  Convolution 5x5	|  1x1 stride, VALID padding,	|
	|					|	outputs 28x28x6				|
	|  ReLU 			|								|
	|  Max pooling		|  outputs 14x14x6				|
	|  Convolution 5x5	|  1x1 stride, VALID padding,	|
	|					|	outputs 10x10x16			|
	|  ReLU 			|								|
	|  Max pooling		|  outputs 5x5x16				|
	|  Flatten			|  5x5x16 -> 400				|
	|  Fully Connected	|  outputs 120					|
	|  ReLU				|								|
	|  Fully Connected	|  outputs 84					|
	|  ReLU				|								|
	|  Fully Connected	|  outputs 43					|

- Second Implementation: Dropout LeNet
	![alt text][image7]
	
	|  Layer			|  Description					|
	|:-----------------:|:-----------------------------:|
	|  Input			|  32x32x1 RGB image			|
	|  Convolution 5x5	|  1x1 stride, VALID padding,	|
	|					|	outputs 28x28x6				|
	|  ReLU 			|								|
	|  Max pooling		|  outputs 14x14x6				|
	|  Convolution 5x5	|  1x1 stride, VALID padding,	|
	|					|	outputs 10x10x16			|
	|  ReLU 			|								|
	|  Max pooling		|  outputs 5x5x16				|
	|  Convolution 5x5	|  1x1 stride, VALID padding,	|
	|					|	outputs 1x1x400				|
	|  ReLU 			|								|
	|  Flatten			|  5x5x16 -> 400				|
	|  Flatten			|  1x1x400 -> 400				|
	|  Concatenate		|  400 + 400 -> 800				|
	|  Dropout			|  Keep Probability = 0.5		|
	|  Fully Connected	|  outputs 	43					|

#### Model Training
- Choosing global parameters: `EPOCH`, `batch_size`, `learning rate`
	* the increase of the `EPOCH` parameter generally correlates with the increase in model accuracy (assuming reasonable learning rate), but requires much more computational power (and thus, time) to fully develop. I settled on an `EPOCH` value of `30` based on previous runs of the model
	* `batch_size` was at `128` by default, but I chose a lower value of `100` since it yielded a higher accuracy
	* `learning rate` is a very big factor in test accuracy and computation speed, which ultimately led me to choose a value of `0.001` in order to retain a consistent accuracy, but not have to spend days running the algorithm
- Choosing hyperparameters: `mu`, `sigma`
	* `mu` was left as 0
	* `sigma` was left as 0.1
	* parameter initialization wasn't an issue for this algorithm, so I felt no need to alter these parameters
- Choosing `convNN` features: number of filters, shape of filters
	* in lecture it was mentioned that multiple convolutional layers generally yields more refined feature identification, which would imply that using more convolution layers would improve accuracy. So layers (based on LeNet filter size) were added until there were no more spatial convolutions to be made
	* number of filters: 2
	* shape of filters (5,5,?,?)
- Other techniques: ReLU, Pooling, Flatten, Dropout, Fully Connected
	* ReLU is used after every convolution layer to null out any noise
	* Pooling is used after the first two covolution layers to decrease the spatial size (decreasing computational demand)
	* Flattening is used after completing the final convolution layer. And to improve upon LeNet, the resulting features of the third convolution layer are flattened and concatenated to the flattened second convolution to provide additional information for the Fully Connected (linear transformation) layer
	* Dropout is included after the Flatten layer to refine the training of model parameters
	* logits are then derived by fully connecting the flattened layer into vectors of length equivalent to the number of image classes

---

### Test a Model on New Images

#### Acquiring New Images
Samples images were acquired from the internet. The sample pool is meant to reflect a variety of sign shapes, angles, colors, and content:
	![alt text][image8]
- sample for class 3 - Speed limit (60km/h)
- sample for class 11 - Right-of-way at the next intersection
- sample for class 17 - No entry
- sample for class 25 - Road Work
- sample for class 36 - Go straight or right

#### Performance on New Images
The overall performance of the model yielded an accuracy of `0.8` for this set of images. The hiccup occurred with the image for class 3, which may have been due to the poor image arrangement.
Upon this discovery, I decided to generate a cropped version of the same image and check if results would improve. However, they did not. 

#### Model Certainty - Softmax Probabilities
Running the samples images from the internet yielded the following Softmax Probabilities. With the sample image for class 3 showing strange results, it may serve as a useful example to postulate potential flaws in the model.
	![alt text][image9]

---

### Future Implementation

#### Preprocessing
Based on the nature of our model, it seems that there is much to be gained from enhancing the image preprocessing techniques. Some improvements may include:
- image centering (offset and perspective)
- shape recognition and refining
- character and symbol recognition and enhancing
While these preprocessing techniques may increase the overhead cost of the model, it may serve to provide a more reliable means of homogenizing the images - especially with the prevalence sign wear or defacement.

#### Modeling
Due to the fact that computation costs play a major role in determining the workflow timeshare split of projects, it is pertinent that the architecture of a model is developed using inferences made from higher level model design. For example, by using the softmax probability visualization we can determine certain conditions (or patterns) where the model has low confidence. Another example would be to use the feature map visualizations to understand how to tweak convolution layer parameters.

---