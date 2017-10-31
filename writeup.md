# Semantic Segmentation Project

## Introduction

In this project, we aim to label the pixels of a road in images using a Fully Convolutional Network (FCN). This classification technique will help other systems in an autonomous driving car to determine where the free space is. With necessary modifications, this technique can be used to classify more classes like road, vehicle, bicycle, and pedestrian.

## Data

* The [Kitti Road dataset](http://kitti.is.tue.mpg.de/kitti/data_road.zip) is used in training and testing. *Extract the dataset in the data folder. It will create the folder data_road with all the training and test images.*

   The orginal image size of Kitti Road dataset is 1242x375. But we will resize them to 576x160. The FCN-8 pipeline is based on the new image size. Refer to the function `gen_batch_function` of **helper.py** for its implementation.

```
def gen_batch_function(data_folder, image_shape):
    def get_batches_fn(batch_size):
        ...
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
        ...
```

* The [VGG16 model](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip) pretrained on ImageNet for classification is used in encoder of FCN-8. *To save time, we suggest one downloads VGG16 model in advance and extracts it to the folder data.*

## FCN-8 Architecture

We’ll basically use the [FCN-8 architecture](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) developed at Berkeley. The encoder for FCN-8 is the VGG16 model pretrained on ImageNet for classification. The fully-connected layers are replaced by 1-by-1 convolutions. It follows with the decoder part, whose structure will be explained in detail in below.


Here’s an example of going from a fully-connected layer to a 1-by-1 convolution in TensorFlow:


## Model Performance

### Data Augmentation

### Model

```
num_epochs
batch_size
tf.truncated_normal_initializer(stddev=0.01)
tf.contrib.layers.l2_regularizer(1e-3)
learning_rate
tf.train.AdamOptimizer()
```



First, an automotic video is split up into individual camera frames. Then by taking the advantage of pretrained VGG16 model, we will design, implement and train a Fully Convolutional Network (FCN-8). Next using it to classify each pixel of a frame as either road or not road.  
