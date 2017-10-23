import numpy as np
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name      = 'image_input:0'
    vgg_keep_prob_tensor_name  = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = sess.graph
    input_image     = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob       = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out  = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out  = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out  = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # FCN-8 implementation
    # We follow the architecture developed at Berkeley
    # https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    # Refer to Figure 3 of above paper.
    
    # For convenience, we use the following short notations for
    # Kernel Initializer and Kernel Regularizer.
    tn = tf.truncated_normal_initializer(stddev=0.01)
    l2 = tf.contrib.layers.l2_regularizer(1e-3)
    # 1x1 convolution from vgg_layer7_out
    layer7_fcn = tf.layers.conv2d(vgg_layer7_out,
                                  num_classes,
                                  kernel_size=(1,1),
                                  strides=(1,1),
                                  padding='same',
                                  kernel_initializer=tn,
                                  kernel_regularizer=l2)
    # followed with upsampling
    layer7_upsample = tf.layers.conv2d_transpose(layer7_fcn,
                                                 num_classes,
                                                 kernel_size=(4,4),
                                                 strides=(2,2),
                                                 padding='same',
                                                 kernel_initializer=tn,
                                                 kernel_regularizer=l2)
    # 1-by-1 convolution from vgg_layer4_out. The output will be acted as a  
    # skip layer and connect with layer7_upsample
    layer4_fcn = tf.layers.conv2d(vgg_layer4_out,
                                  num_classes,
                                  kernel_size=(1,1),
                                  strides=(1,1),
                                  padding='same',
                                  kernel_initializer=tn,
                                  kernel_regularizer=l2)
    layer4_skip = tf.add(layer4_fcn, layer7_upsample)
    # continue with upsampling, whose output will have same shape with
    # vgg_layer3_out
    layer4_upsample = tf.layers.conv2d_transpose(layer4_skip,
                                                 num_classes,
                                                 kernel_size=(4,4),
                                                 strides=(2,2),
                                                 padding='same',
                                                 kernel_initializer=tn,
                                                 kernel_regularizer=l2)
    # 1x1 convolution from vgg_layer3_out. The output will be acted as a
    # skip layer and connect with layer4_upsample
    layer3_fcn = tf.layers.conv2d(vgg_layer3_out,
                                  num_classes,
                                  kernel_size=(1,1),
                                  strides=(1,1),
                                  padding='same',
                                  kernel_initializer=tn,
                                  kernel_regularizer=l2)
    layer3_skip = tf.add(layer3_fcn,layer4_upsample)
    # continue with upsmapling, whose output will have same shape with the
    # input_image layer
    output_layer = tf.layers.conv2d_transpose(layer3_skip,
                                              num_classes,
                                              kernel_size=(16,16),
                                              strides=(8,8),
                                              padding='same',
                                              kernel_initializer=tn,
                                              kernel_regularizer=l2)
    return output_layer
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    labels = tf.reshape(correct_label, [-1, num_classes])
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    # We use Adam optimizer
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def data_augmentation(images, gt_images):
    """
    Data augmentation: flip, add noise
    :param images: batch of input images
    :param gt_images: batch of ground truth images
    :return: augmented images and corresponding ground truth images
    """
    shape = images.shape
    #print(shape)
    # flip horizontally
    images_h    = images[:, :, ::-1, :]
    gt_images_h = gt_images[:, :, ::-1, :]
    # flip vertically
    images_v    = images[:, ::-1, :, :]
    gt_images_v = gt_images[:, ::-1, :, :]
    # add noises
    noises = np.random.randint(0, 256, shape)
    images_n = (images + noises) % 256
    gt_images_n = gt_images
    # concatenate all augmented images and ground truth images
    augmented_images    = np.concatenate([images, images_h, images_v, images_n],
                                      axis=0)
    augmented_gt_images = np.concatenate([gt_images, gt_images_h, gt_images_v, gt_images_n],
                                         axis=0)
    
    return augmented_images, augmented_gt_images

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for epoch in range(epochs):
        for batch_num, (X,y) in enumerate(get_batches_fn(batch_size)):
            if X.ndim == 4:
                X, y = data_augmentation(X,y)
            feed_dict = {
                input_image: X,
                correct_label: y,
                keep_prob: 0.3,
                learning_rate: 0.001
            }
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed_dict)
            if batch_num % 5 == 0:
                print("Epoch: {} | Batch: {} | Loss: {}".format(epoch, batch_num, loss))
tests.test_train_nn(train_nn)

def run():
    num_epochs = 50
    batch_size = 5
    
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    # https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        # https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        output_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        # placeholder for labels and learning rate
        correct_label = tf.placeholder(dtype=tf.float32, shape=[None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(dtype=tf.float32)
        logits, train_op, loss = optimize(output_layer, correct_label, learning_rate, num_classes)

        saver = tf.train.Saver()
        # load pretrained model from disk
        checkpoint = tf.train.get_checkpoint_state('./model')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(sess, checkpoint.model_checkpoint_path)

        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, num_epochs, batch_size, get_batches_fn, train_op, loss, input_image, correct_label,
                 keep_prob, learning_rate)

        # save model
        save_path = saver.save(sess, "./models/model.ckpt")
        print("Model saved in file: {}".format(save_path))

        saver.export_meta_graph("./models/model.meta")
        tf.train.write_graph(sess.graph.as_graph_def(), './model', 'saved_graph.pb', as_text=False)
        # Save inference data using helper.save_inference_samples
        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()









