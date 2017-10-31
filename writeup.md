# Semantic Segmentation Project

## Introduction

In this project, we aim to label the pixels of a road in images using a Fully Convolutional Network (FCN). This classification technique will help other systems in an autonomous driving car to determine where the free space is. With necessary modifications, this technique can be used to classify more classes like road, vehicle, bicycle, and pedestrian.

## Training and Testing Data

* The [Kitti Road dataset](http://kitti.is.tue.mpg.de/kitti/data_road.zip) will be used as training and testing datasets. *Extract the dataset to the 'data' folder. It will create a folder 'data_road' with all the training and test images.*

   It's worth to mention that the orginal image size of Kitti Road dataset is 1242x375. But they will be resized to 576x160 and then used as the input of our model. Please refer to the function `gen_batch_function` of **helper.py** for the implementation.

```
def gen_batch_function(data_folder, image_shape):
    def get_batches_fn(batch_size):
        ...
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
        ...
```

* The [VGG16 model](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip) pretrained on ImageNet for classification will be used as the encoder part of FCN-8. *It will be downloaded in advance and extracted to the folder 'data'.*

## FCN-8 Architecture

In this project, we develop the famous [FCN-8 architecture](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) and use them during the model. It includes two parts: encoder and decoder. The idea is as follows. 

First, use a VGG16 model as the encoder part of the FCN. To preserve certain spatial information, we replace the fully connected layers (last layer of VGG16) by $1x1$ convolutions. 

![alt text](https://github.com/fangchun007/CarND-Semantic-Segmentation/blob/master/vgg16.png "VGG16")

The decoder part is consist of three convolutional layers. They progressively upsample to the original image size. As we known, some global information, such as frames of road, are normally extracted during earily part of VGG16. They will be very useful during labelling pixels of an object. In order to do this, three skip connections are added between encoder and decoder. Please refer to the following figure.

![alt_text](https://github.com/fangchun007/CarND-Semantic-Segmentation/blob/master/FCN8.jpg)

adding skip connections to the model. In order to do this we’ll combine the output of two layers. The first output is the output of the current layer. The second output is the output of a layer further back in the network, typically a pooling layer. In the following example we combine the result of the previous layer with the result of the 4th pooling layer through elementwise addition (tf.add).


encoder for FCN-8 is the VGG16 model pretrained on ImageNet for classification. The fully-connected layers are replaced by 1-by-1 convolutions. It follows with the decoder part, whose structure will be explained in detail in below.


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
