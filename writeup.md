# Semantic Segmentation Project

### Introduction

In this project, we aim to label the pixels of a road in images using a Fully Convolutional Network (FCN). This classification technique will help other systems in an autonomous driving car to determine where the free space is. With necessary modifications, this technique can be used to classify more classes like road, vehicle, bicycle, and pedestrian.

### Training and Testing Data

* The [Kitti Road dataset](http://kitti.is.tue.mpg.de/kitti/data_road.zip) will be used as training and testing datasets. *Download and Extract the dataset to the 'data' folder. It will create a folder 'data_road' with all the training and test images.*

   It's worth to mention that the orginal image size for Kitti Road dataset is 1242x375. They will be resized to 576x160 and then used as input data of our model. Please refer to the function `gen_batch_function` of **helper.py** for the implementation.

```
def gen_batch_function(data_folder, image_shape):
    def get_batches_fn(batch_size):
        ...
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
        ...
```

* The [VGG16 model](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip) pretrained on ImageNet for classification is used as the encoder part of FCN. *Download and extract to the folder 'data'.*

### FCN Architecture

In this project, we develop the famous [FCN-8 architecture](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) and use it as our training model. It includes two parts: encoder and decoder.

First, use VGG16 as the encoder of the model. To preserve certain spatial information, we replace the fully connected layers, i.e., last layer of VGG16, by 1x1 convolutions. 

![alt text](https://github.com/fangchun007/CarND-Semantic-Segmentation/blob/master/vgg16.png "VGG16")

The decoder part is consist of three convolutional layers. They progressively upsample to the original image size. The whole process of decoder is linear operation, meaning all nonlinear actions are all located in encoder part. 

It is known that some global information, such as road boundaries and frames, are normally extracted during earily part of VGG16. They will be very useful during labelling the pixels of an object. In order to do this, three skip connections are added between encoder and decoder. Please refer to next figure for the whole architecture of FCN-8.

![alt_text](https://github.com/fangchun007/CarND-Semantic-Segmentation/blob/master/FCN8.jpg)

Implementation: See the *layers* function of [main.py](https://github.com/fangchun007/CarND-Semantic-Segmentation/blob/master/main.py)

### Performance

#### Data Augmentation

In the Kitty Road Dataset, there are only 290 images used in training. With the same architecture and similar hyperparameters, which should be adjusted in order to obtain a better result, the testing results show less clear roads' boundaries. Sometimes, they miss labelling part of road information. Data augmentation can be helpful in this situation. In this project, we tried to combine the following four methods in the pipeline of data augmentation.

* randomly flip
* randomly change contrast
* randomly change brightness
* randomly blur (kernel size 1=no change, 3, 5, or 7)

In the experiments, we observed that the first three methods are good for obtain a better result. But for the blur method, it seems brings an adverse effect, which cause the *loss* fluctuating at around 0.04, even if we decrease the learning rate to 1e-6.

I also tried to **normalization** on the image data by 

```
    image = image/255
```
and by 
```
    image = (image-np.ones*128)/256
```

In this case, the loss will drop very fast from around 1.2 to 0.3. But it then fluctuate at around 0.25, even though I decrease the learning rate to 1e-6. The testing result is naturally unacceptable.

Implementation: See the *data_augmentation* function of [main.py](https://github.com/fangchun007/CarND-Semantic-Segmentation/blob/master/main.py)

### Hyperparameters

The hyperparameters of this version are set as follows. 

```
num_epochs = 50 # 40 should be OK
batch_size = 8
learning_rate = 0.00008
keep_prob = 0.5
```

### Regularization

Truncated normal initializer `tf.truncated_normal_initializer(stddev=0.01)`, kernel regularizer `tf.contrib.layers.l2_regularizer(1e-3)`, dropout and Adam optimizer `tf.train.AdamOptimizer()` are used here to overcome the overfitting.

We also point out that the 'filters' of several convolutional (transpose) layers of the decode part are adjusted and they are not always equal to `num_classes`, in order to get a lower loss in 50 epochs of training. Of course, this purpose can also be achived by increase the keep probability to 0.8.

### Result

![alt text](https://github.com/fangchun007/CarND-Semantic-Segmentation/blob/master/data_augment_sample.png "data augmentation sample")




### Appendix: data augmentation

[!alt_text]()
