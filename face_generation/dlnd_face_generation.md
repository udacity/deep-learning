
# Face Generation
In this project, you'll use generative adversarial networks to generate new images of faces.
### Get the Data
You'll be using two datasets in this project:
- MNIST
- CelebA

Since the celebA dataset is complex and you're doing GANs in a project for the first time, we want you to test your neural network on MNIST before CelebA.  Running the GANs on MNIST will allow you to see how well your model trains sooner.

If you're using [FloydHub](https://www.floydhub.com/), set `data_dir` to "/input" and use the [FloydHub data ID](http://docs.floydhub.com/home/using_datasets/) "R5KrjnANiKVhLWAkpXhNBe".


```python
data_dir = './data'

# FloydHub - Use with data ID "R5KrjnANiKVhLWAkpXhNBe"
data_dir = '/input'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)
```

    Found mnist Data
    Found celeba Data


## Explore the Data
### MNIST
As you're aware, the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains images of handwritten digits. You can view the first number of examples by changing `show_n_images`. 


```python
show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
%matplotlib inline
import os
from glob import glob
from matplotlib import pyplot

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7fa913657d30>




![png](dlnd_face_generation_files/dlnd_face_generation_3_1.png)


### CelebA
The [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations.  Since you're going to be generating faces, you won't need the annotations.  You can view the first number of examples by changing `show_n_images`.


```python
show_n_images = 25

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))
```




    <matplotlib.image.AxesImage at 0x7fa91354ee10>




![png](dlnd_face_generation_files/dlnd_face_generation_5_1.png)


## Preprocess the Data
Since the project's main focus is on building the GANs, we'll preprocess the data for you.  The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images.  The CelebA images will be cropped to remove parts of the image that don't include a face, then resized down to 28x28.

The MNIST images are black and white images with a single [color channel](https://en.wikipedia.org/wiki/Channel_(digital_image%29) while the CelebA images have [3 color channels (RGB color channel)](https://en.wikipedia.org/wiki/Channel_(digital_image%29#RGB_Images).
## Build the Neural Network
You'll build the components necessary to build a GANs by implementing the following functions below:
- `model_inputs`
- `discriminator`
- `generator`
- `model_loss`
- `model_opt`
- `train`

### Check the Version of TensorFlow and Access to GPU
This will check to make sure you have the correct version of TensorFlow and access to a GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    Default GPU Device: /gpu:0


### Input
Implement the `model_inputs` function to create TF Placeholders for the Neural Network. It should create the following placeholders:
- Real input images placeholder with rank 4 using `image_width`, `image_height`, and `image_channels`.
- Z input placeholder with rank 2 using `z_dim`.
- Learning rate placeholder with rank 0.

Return the placeholders in the following the tuple (tensor of real input images, tensor of z data)


```python
import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    # TODO: Implement Function
    inputs_real = tf.placeholder(tf.float32, shape=(None, image_width, image_height, image_channels), name='input_real') 
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    return inputs_real, inputs_z, learning_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed


### Discriminator
Implement `discriminator` to create a discriminator neural network that discriminates on `images`.  This function should be able to reuse the variabes in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "discriminator" to allow the variables to be reused.  The function should return a tuple of (tensor output of the discriminator, tensor logits of the discriminator).


```python
def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param image: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    alpha = 0.2
    
    with tf.variable_scope('discriminator', reuse=reuse):
        # using 4 layer network as in DCGAN Paper
        
        # Conv 1
        conv1 = tf.layers.conv2d(images, 64, 5, 2, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        lrelu1 = tf.maximum(alpha * conv1, conv1)
        
        # Conv 2
        conv2 = tf.layers.conv2d(lrelu1, 128, 5, 2, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        batch_norm2 = tf.layers.batch_normalization(conv2, training=True)
        lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)
        drop2 = tf.nn.dropout(lrelu2, keep_prob=0.5)        
        
        # Conv 3
        conv3 = tf.layers.conv2d(drop2, 256, 5, 1, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        batch_norm3 = tf.layers.batch_normalization(conv3, training=True)
        lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)
        drop3 = tf.nn.dropout(lrelu3, keep_prob=0.5)        
        
        # Conv 4
        conv4 = tf.layers.conv2d(drop3, 512, 5, 1, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        batch_norm4 = tf.layers.batch_normalization(conv4, training=True)
        lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)
        drop4 = tf.nn.dropout(lrelu4, keep_prob=0.5)
       
        # Flatten
        flat = tf.reshape(drop4, (-1, 7*7*512))
        
        # Logits
        logits = tf.layers.dense(flat, 1)
        
        # Output
        out = tf.sigmoid(logits)
        
        return out, logits

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(discriminator, tf)
```

    Tests Passed


### Generator
Implement `generator` to generate an image using `z`. This function should be able to reuse the variabes in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "generator" to allow the variables to be reused. The function should return the generated 28 x 28 x `out_channel_dim` images.


```python
def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    alpha = 0.2
    
    with tf.variable_scope('generator', reuse=False if is_train==True else True):
        # Fully connected
        fc1 = tf.layers.dense(z, 7*7*512)
        fc1 = tf.reshape(fc1, (-1, 7, 7, 512))
        fc1 = tf.maximum(alpha*fc1, fc1)
        
        # Starting Conv Transpose Stack
        deconv2 = tf.layers.conv2d_transpose(fc1, 256, 3, 1, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        batch_norm2 = tf.layers.batch_normalization(deconv2, training=is_train)
        lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)
        drop2 = tf.nn.dropout(lrelu2, keep_prob=0.5)
        
        deconv3 = tf.layers.conv2d_transpose(drop2, 128, 3, 1, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        batch_norm3 = tf.layers.batch_normalization(deconv3, training=is_train)
        lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)
        drop3 = tf.nn.dropout(lrelu3, keep_prob=0.5)
        
        deconv4 = tf.layers.conv2d_transpose(drop3, 64, 3, 2, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        batch_norm4 = tf.layers.batch_normalization(deconv4, training=is_train)
        lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)
        drop4 = tf.nn.dropout(lrelu4, keep_prob=0.5)
        
        # Logits
        logits = tf.layers.conv2d_transpose(drop4, out_channel_dim, 3, 2, 'SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        # Output
        out = tf.tanh(logits)
        
        return out



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(generator, tf)
```

    Tests Passed


### Loss
Implement `model_loss` to build the GANs for training and calculate the loss.  The function should return a tuple of (discriminator loss, generator loss).  Use the following functions you implemented:
- `discriminator(images, reuse=False)`
- `generator(z, out_channel_dim, is_train=True)`


```python
def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    g_model = generator(input_z, out_channel_dim)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)
    
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real) * 0.9)
    )
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake))
    )
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake))
    )
    
    return d_loss, g_loss




"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_loss(model_loss)
```

    Tests Passed


### Optimization
Implement `model_opt` to create the optimization operations for the GANs. Use [`tf.trainable_variables`](https://www.tensorflow.org/api_docs/python/tf/trainable_variables) to get all the trainable variables.  Filter the variables with names that are in the discriminator and generator scope names.  The function should return a tuple of (discriminator training operation, generator training operation).


```python
def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
        g_train_opt = tf.train.AdamOptimizer(learning_rate = learning_rate,beta1 = beta1).minimize(g_loss, var_list = g_vars)

    return d_train_opt, g_train_opt

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_opt(model_opt, tf)
```

    Tests Passed


## Neural Network Training
### Show Output
Use this function to show the current output of the generator during training. It will help you determine how well the GANs is training.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()
```

### Train
Implement `train` to build and train the GANs.  Use the following functions you implemented:
- `model_inputs(image_width, image_height, image_channels, z_dim)`
- `model_loss(input_real, input_z, out_channel_dim)`
- `model_opt(d_loss, g_loss, learning_rate, beta1)`

Use the `show_generator_output` to show `generator` output while you train. Running `show_generator_output` for every batch will drastically increase training time and increase the size of the notebook.  It's recommended to print the `generator` output every 100 batches.


```python
def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    tf.reset_default_graph()
    input_real, input_z, _ = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    
    steps = 0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                batch_images = batch_images * 2
                steps += 1
            
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_opt, feed_dict={input_z: batch_z})
                
                if steps % 100 == 0:
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})

                    print("Epoch {}/{}...".format(epoch_i+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    
                    _ = show_generator_output(sess, 1, input_z, data_shape[3], data_image_mode)

                
                
```

### MNIST
Test your GANs architecture on MNIST.  After 2 epochs, the GANs should be able to generate images that look like handwritten digits.  Make sure the loss of the generator is lower than the loss of the discriminator or close to 0.


```python
batch_size = 32
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)
```

    Epoch 1/2... Discriminator Loss: 2.2619... Generator Loss: 0.6202



![png](dlnd_face_generation_files/dlnd_face_generation_23_1.png)


    Epoch 1/2... Discriminator Loss: 1.4664... Generator Loss: 1.0111



![png](dlnd_face_generation_files/dlnd_face_generation_23_3.png)


    Epoch 1/2... Discriminator Loss: 1.8397... Generator Loss: 2.6817



![png](dlnd_face_generation_files/dlnd_face_generation_23_5.png)


    Epoch 1/2... Discriminator Loss: 2.8058... Generator Loss: 0.1577



![png](dlnd_face_generation_files/dlnd_face_generation_23_7.png)


    Epoch 1/2... Discriminator Loss: 2.5614... Generator Loss: 0.1357



![png](dlnd_face_generation_files/dlnd_face_generation_23_9.png)


    Epoch 1/2... Discriminator Loss: 1.1280... Generator Loss: 2.5137



![png](dlnd_face_generation_files/dlnd_face_generation_23_11.png)


    Epoch 1/2... Discriminator Loss: 2.5035... Generator Loss: 0.1815



![png](dlnd_face_generation_files/dlnd_face_generation_23_13.png)


    Epoch 1/2... Discriminator Loss: 2.3468... Generator Loss: 0.1857



![png](dlnd_face_generation_files/dlnd_face_generation_23_15.png)


    Epoch 1/2... Discriminator Loss: 0.8233... Generator Loss: 1.8711



![png](dlnd_face_generation_files/dlnd_face_generation_23_17.png)


    Epoch 1/2... Discriminator Loss: 3.1373... Generator Loss: 0.1207



![png](dlnd_face_generation_files/dlnd_face_generation_23_19.png)


    Epoch 1/2... Discriminator Loss: 1.4474... Generator Loss: 2.4362



![png](dlnd_face_generation_files/dlnd_face_generation_23_21.png)


    Epoch 1/2... Discriminator Loss: 2.7549... Generator Loss: 0.1637



![png](dlnd_face_generation_files/dlnd_face_generation_23_23.png)


    Epoch 1/2... Discriminator Loss: 1.3976... Generator Loss: 0.7433



![png](dlnd_face_generation_files/dlnd_face_generation_23_25.png)


    Epoch 1/2... Discriminator Loss: 3.7434... Generator Loss: 0.0523



![png](dlnd_face_generation_files/dlnd_face_generation_23_27.png)


    Epoch 1/2... Discriminator Loss: 1.2509... Generator Loss: 0.8387



![png](dlnd_face_generation_files/dlnd_face_generation_23_29.png)


    Epoch 1/2... Discriminator Loss: 1.5232... Generator Loss: 0.7338



![png](dlnd_face_generation_files/dlnd_face_generation_23_31.png)


    Epoch 1/2... Discriminator Loss: 0.7061... Generator Loss: 1.4835



![png](dlnd_face_generation_files/dlnd_face_generation_23_33.png)


    Epoch 1/2... Discriminator Loss: 1.2265... Generator Loss: 0.7313



![png](dlnd_face_generation_files/dlnd_face_generation_23_35.png)


    Epoch 1/2... Discriminator Loss: 0.7040... Generator Loss: 2.0532



![png](dlnd_face_generation_files/dlnd_face_generation_23_37.png)


    Epoch 1/2... Discriminator Loss: 2.0675... Generator Loss: 0.3187



![png](dlnd_face_generation_files/dlnd_face_generation_23_39.png)


    Epoch 1/2... Discriminator Loss: 1.1110... Generator Loss: 1.8096



![png](dlnd_face_generation_files/dlnd_face_generation_23_41.png)


    Epoch 1/2... Discriminator Loss: 1.1479... Generator Loss: 0.9486



![png](dlnd_face_generation_files/dlnd_face_generation_23_43.png)


    Epoch 1/2... Discriminator Loss: 0.5944... Generator Loss: 2.2339



![png](dlnd_face_generation_files/dlnd_face_generation_23_45.png)


    Epoch 1/2... Discriminator Loss: 0.9548... Generator Loss: 1.2912



![png](dlnd_face_generation_files/dlnd_face_generation_23_47.png)


    Epoch 1/2... Discriminator Loss: 1.7245... Generator Loss: 0.4146



![png](dlnd_face_generation_files/dlnd_face_generation_23_49.png)


    Epoch 1/2... Discriminator Loss: 0.6424... Generator Loss: 2.4182



![png](dlnd_face_generation_files/dlnd_face_generation_23_51.png)


    Epoch 1/2... Discriminator Loss: 1.6380... Generator Loss: 0.5339



![png](dlnd_face_generation_files/dlnd_face_generation_23_53.png)


    Epoch 1/2... Discriminator Loss: 0.5883... Generator Loss: 1.6673



![png](dlnd_face_generation_files/dlnd_face_generation_23_55.png)


    Epoch 1/2... Discriminator Loss: 0.9067... Generator Loss: 1.8976



![png](dlnd_face_generation_files/dlnd_face_generation_23_57.png)


    Epoch 1/2... Discriminator Loss: 2.1298... Generator Loss: 0.2611



![png](dlnd_face_generation_files/dlnd_face_generation_23_59.png)


    Epoch 1/2... Discriminator Loss: 1.9435... Generator Loss: 0.4812



![png](dlnd_face_generation_files/dlnd_face_generation_23_61.png)


    Epoch 1/2... Discriminator Loss: 0.9500... Generator Loss: 0.9799



![png](dlnd_face_generation_files/dlnd_face_generation_23_63.png)


    Epoch 1/2... Discriminator Loss: 0.6482... Generator Loss: 1.6101



![png](dlnd_face_generation_files/dlnd_face_generation_23_65.png)


    Epoch 1/2... Discriminator Loss: 0.7970... Generator Loss: 1.4174



![png](dlnd_face_generation_files/dlnd_face_generation_23_67.png)


    Epoch 1/2... Discriminator Loss: 0.9533... Generator Loss: 2.1404



![png](dlnd_face_generation_files/dlnd_face_generation_23_69.png)


    Epoch 1/2... Discriminator Loss: 1.0686... Generator Loss: 0.9664



![png](dlnd_face_generation_files/dlnd_face_generation_23_71.png)


    Epoch 1/2... Discriminator Loss: 1.1878... Generator Loss: 0.8873



![png](dlnd_face_generation_files/dlnd_face_generation_23_73.png)


    Epoch 1/2... Discriminator Loss: 1.4163... Generator Loss: 0.5699



![png](dlnd_face_generation_files/dlnd_face_generation_23_75.png)


    Epoch 1/2... Discriminator Loss: 0.6724... Generator Loss: 1.6672



![png](dlnd_face_generation_files/dlnd_face_generation_23_77.png)


    Epoch 1/2... Discriminator Loss: 1.7985... Generator Loss: 0.4420



![png](dlnd_face_generation_files/dlnd_face_generation_23_79.png)


    Epoch 1/2... Discriminator Loss: 1.1635... Generator Loss: 0.8786



![png](dlnd_face_generation_files/dlnd_face_generation_23_81.png)


    Epoch 1/2... Discriminator Loss: 0.9925... Generator Loss: 1.4314



![png](dlnd_face_generation_files/dlnd_face_generation_23_83.png)


    Epoch 1/2... Discriminator Loss: 1.0148... Generator Loss: 1.9695



![png](dlnd_face_generation_files/dlnd_face_generation_23_85.png)


    Epoch 1/2... Discriminator Loss: 0.8669... Generator Loss: 1.3644



![png](dlnd_face_generation_files/dlnd_face_generation_23_87.png)


    Epoch 1/2... Discriminator Loss: 1.6663... Generator Loss: 0.4707



![png](dlnd_face_generation_files/dlnd_face_generation_23_89.png)


    Epoch 1/2... Discriminator Loss: 1.4116... Generator Loss: 0.5953



![png](dlnd_face_generation_files/dlnd_face_generation_23_91.png)


    Epoch 1/2... Discriminator Loss: 0.9415... Generator Loss: 1.1345



![png](dlnd_face_generation_files/dlnd_face_generation_23_93.png)


    Epoch 1/2... Discriminator Loss: 1.7062... Generator Loss: 0.3555



![png](dlnd_face_generation_files/dlnd_face_generation_23_95.png)


    Epoch 1/2... Discriminator Loss: 0.6146... Generator Loss: 2.0326



![png](dlnd_face_generation_files/dlnd_face_generation_23_97.png)


    Epoch 1/2... Discriminator Loss: 1.5595... Generator Loss: 0.6007



![png](dlnd_face_generation_files/dlnd_face_generation_23_99.png)


    Epoch 1/2... Discriminator Loss: 1.0279... Generator Loss: 0.8501



![png](dlnd_face_generation_files/dlnd_face_generation_23_101.png)


    Epoch 1/2... Discriminator Loss: 2.6635... Generator Loss: 0.1264



![png](dlnd_face_generation_files/dlnd_face_generation_23_103.png)


    Epoch 1/2... Discriminator Loss: 2.0149... Generator Loss: 0.3093



![png](dlnd_face_generation_files/dlnd_face_generation_23_105.png)


    Epoch 1/2... Discriminator Loss: 0.5226... Generator Loss: 3.0728



![png](dlnd_face_generation_files/dlnd_face_generation_23_107.png)


    Epoch 1/2... Discriminator Loss: 1.1839... Generator Loss: 1.0181



![png](dlnd_face_generation_files/dlnd_face_generation_23_109.png)


    Epoch 1/2... Discriminator Loss: 1.3848... Generator Loss: 0.6558



![png](dlnd_face_generation_files/dlnd_face_generation_23_111.png)


    Epoch 1/2... Discriminator Loss: 2.8023... Generator Loss: 4.1509



![png](dlnd_face_generation_files/dlnd_face_generation_23_113.png)


    Epoch 1/2... Discriminator Loss: 0.6699... Generator Loss: 2.4823



![png](dlnd_face_generation_files/dlnd_face_generation_23_115.png)


    Epoch 1/2... Discriminator Loss: 1.2144... Generator Loss: 1.0774



![png](dlnd_face_generation_files/dlnd_face_generation_23_117.png)


    Epoch 1/2... Discriminator Loss: 0.6696... Generator Loss: 1.5526



![png](dlnd_face_generation_files/dlnd_face_generation_23_119.png)


    Epoch 2/2... Discriminator Loss: 0.9967... Generator Loss: 0.9785



![png](dlnd_face_generation_files/dlnd_face_generation_23_121.png)


    Epoch 2/2... Discriminator Loss: 1.1415... Generator Loss: 0.9937



![png](dlnd_face_generation_files/dlnd_face_generation_23_123.png)


    Epoch 2/2... Discriminator Loss: 2.6582... Generator Loss: 0.3289



![png](dlnd_face_generation_files/dlnd_face_generation_23_125.png)


    Epoch 2/2... Discriminator Loss: 0.8261... Generator Loss: 1.0906



![png](dlnd_face_generation_files/dlnd_face_generation_23_127.png)


    Epoch 2/2... Discriminator Loss: 1.2622... Generator Loss: 1.0366



![png](dlnd_face_generation_files/dlnd_face_generation_23_129.png)


    Epoch 2/2... Discriminator Loss: 1.0750... Generator Loss: 0.9029



![png](dlnd_face_generation_files/dlnd_face_generation_23_131.png)


    Epoch 2/2... Discriminator Loss: 1.3828... Generator Loss: 0.5317



![png](dlnd_face_generation_files/dlnd_face_generation_23_133.png)


    Epoch 2/2... Discriminator Loss: 1.0293... Generator Loss: 1.0517



![png](dlnd_face_generation_files/dlnd_face_generation_23_135.png)


    Epoch 2/2... Discriminator Loss: 0.9707... Generator Loss: 1.0313



![png](dlnd_face_generation_files/dlnd_face_generation_23_137.png)


    Epoch 2/2... Discriminator Loss: 1.0228... Generator Loss: 1.1384



![png](dlnd_face_generation_files/dlnd_face_generation_23_139.png)


    Epoch 2/2... Discriminator Loss: 0.6503... Generator Loss: 1.6707



![png](dlnd_face_generation_files/dlnd_face_generation_23_141.png)


    Epoch 2/2... Discriminator Loss: 0.5282... Generator Loss: 2.3339



![png](dlnd_face_generation_files/dlnd_face_generation_23_143.png)


    Epoch 2/2... Discriminator Loss: 1.3892... Generator Loss: 0.5346



![png](dlnd_face_generation_files/dlnd_face_generation_23_145.png)


    Epoch 2/2... Discriminator Loss: 0.9588... Generator Loss: 1.4284



![png](dlnd_face_generation_files/dlnd_face_generation_23_147.png)


    Epoch 2/2... Discriminator Loss: 0.7025... Generator Loss: 1.6237



![png](dlnd_face_generation_files/dlnd_face_generation_23_149.png)


    Epoch 2/2... Discriminator Loss: 0.5298... Generator Loss: 2.3509



![png](dlnd_face_generation_files/dlnd_face_generation_23_151.png)


    Epoch 2/2... Discriminator Loss: 0.8266... Generator Loss: 1.6712



![png](dlnd_face_generation_files/dlnd_face_generation_23_153.png)


    Epoch 2/2... Discriminator Loss: 0.7191... Generator Loss: 1.4390



![png](dlnd_face_generation_files/dlnd_face_generation_23_155.png)


    Epoch 2/2... Discriminator Loss: 0.6141... Generator Loss: 1.6631



![png](dlnd_face_generation_files/dlnd_face_generation_23_157.png)


    Epoch 2/2... Discriminator Loss: 0.9813... Generator Loss: 1.0941



![png](dlnd_face_generation_files/dlnd_face_generation_23_159.png)


    Epoch 2/2... Discriminator Loss: 0.4799... Generator Loss: 3.0742



![png](dlnd_face_generation_files/dlnd_face_generation_23_161.png)


    Epoch 2/2... Discriminator Loss: 1.2080... Generator Loss: 0.8269



![png](dlnd_face_generation_files/dlnd_face_generation_23_163.png)


    Epoch 2/2... Discriminator Loss: 0.9692... Generator Loss: 0.9870



![png](dlnd_face_generation_files/dlnd_face_generation_23_165.png)


    Epoch 2/2... Discriminator Loss: 0.6068... Generator Loss: 1.9257



![png](dlnd_face_generation_files/dlnd_face_generation_23_167.png)


    Epoch 2/2... Discriminator Loss: 1.0769... Generator Loss: 0.9509



![png](dlnd_face_generation_files/dlnd_face_generation_23_169.png)


    Epoch 2/2... Discriminator Loss: 2.3437... Generator Loss: 0.2293



![png](dlnd_face_generation_files/dlnd_face_generation_23_171.png)


    Epoch 2/2... Discriminator Loss: 0.9102... Generator Loss: 1.2042



![png](dlnd_face_generation_files/dlnd_face_generation_23_173.png)


    Epoch 2/2... Discriminator Loss: 0.6833... Generator Loss: 1.3908



![png](dlnd_face_generation_files/dlnd_face_generation_23_175.png)


    Epoch 2/2... Discriminator Loss: 0.6520... Generator Loss: 1.6765



![png](dlnd_face_generation_files/dlnd_face_generation_23_177.png)


    Epoch 2/2... Discriminator Loss: 1.1966... Generator Loss: 0.6901



![png](dlnd_face_generation_files/dlnd_face_generation_23_179.png)


    Epoch 2/2... Discriminator Loss: 0.7102... Generator Loss: 1.4153



![png](dlnd_face_generation_files/dlnd_face_generation_23_181.png)


    Epoch 2/2... Discriminator Loss: 1.3576... Generator Loss: 0.6304



![png](dlnd_face_generation_files/dlnd_face_generation_23_183.png)


    Epoch 2/2... Discriminator Loss: 0.6110... Generator Loss: 1.9147



![png](dlnd_face_generation_files/dlnd_face_generation_23_185.png)


    Epoch 2/2... Discriminator Loss: 1.1161... Generator Loss: 0.8729



![png](dlnd_face_generation_files/dlnd_face_generation_23_187.png)


    Epoch 2/2... Discriminator Loss: 0.6793... Generator Loss: 1.4487



![png](dlnd_face_generation_files/dlnd_face_generation_23_189.png)


    Epoch 2/2... Discriminator Loss: 0.6738... Generator Loss: 1.6718



![png](dlnd_face_generation_files/dlnd_face_generation_23_191.png)


    Epoch 2/2... Discriminator Loss: 0.9162... Generator Loss: 0.9070



![png](dlnd_face_generation_files/dlnd_face_generation_23_193.png)


    Epoch 2/2... Discriminator Loss: 0.6361... Generator Loss: 1.9108



![png](dlnd_face_generation_files/dlnd_face_generation_23_195.png)


    Epoch 2/2... Discriminator Loss: 0.5519... Generator Loss: 2.0668



![png](dlnd_face_generation_files/dlnd_face_generation_23_197.png)


    Epoch 2/2... Discriminator Loss: 0.6307... Generator Loss: 1.8360



![png](dlnd_face_generation_files/dlnd_face_generation_23_199.png)


    Epoch 2/2... Discriminator Loss: 0.6643... Generator Loss: 1.7487



![png](dlnd_face_generation_files/dlnd_face_generation_23_201.png)


    Epoch 2/2... Discriminator Loss: 2.2935... Generator Loss: 0.2092



![png](dlnd_face_generation_files/dlnd_face_generation_23_203.png)


    Epoch 2/2... Discriminator Loss: 1.9879... Generator Loss: 0.2661



![png](dlnd_face_generation_files/dlnd_face_generation_23_205.png)


    Epoch 2/2... Discriminator Loss: 0.6725... Generator Loss: 1.4516



![png](dlnd_face_generation_files/dlnd_face_generation_23_207.png)


    Epoch 2/2... Discriminator Loss: 1.9201... Generator Loss: 0.3181



![png](dlnd_face_generation_files/dlnd_face_generation_23_209.png)


    Epoch 2/2... Discriminator Loss: 0.9473... Generator Loss: 1.0200



![png](dlnd_face_generation_files/dlnd_face_generation_23_211.png)


    Epoch 2/2... Discriminator Loss: 0.7540... Generator Loss: 1.2531



![png](dlnd_face_generation_files/dlnd_face_generation_23_213.png)


    Epoch 2/2... Discriminator Loss: 1.3744... Generator Loss: 0.5950



![png](dlnd_face_generation_files/dlnd_face_generation_23_215.png)


    Epoch 2/2... Discriminator Loss: 0.7612... Generator Loss: 1.3399



![png](dlnd_face_generation_files/dlnd_face_generation_23_217.png)


    Epoch 2/2... Discriminator Loss: 0.7148... Generator Loss: 1.3688



![png](dlnd_face_generation_files/dlnd_face_generation_23_219.png)


    Epoch 2/2... Discriminator Loss: 1.4309... Generator Loss: 0.5351



![png](dlnd_face_generation_files/dlnd_face_generation_23_221.png)


    Epoch 2/2... Discriminator Loss: 2.3729... Generator Loss: 0.2233



![png](dlnd_face_generation_files/dlnd_face_generation_23_223.png)


    Epoch 2/2... Discriminator Loss: 0.5871... Generator Loss: 1.7392



![png](dlnd_face_generation_files/dlnd_face_generation_23_225.png)


    Epoch 2/2... Discriminator Loss: 0.8773... Generator Loss: 1.4090



![png](dlnd_face_generation_files/dlnd_face_generation_23_227.png)


    Epoch 2/2... Discriminator Loss: 0.6498... Generator Loss: 1.7226



![png](dlnd_face_generation_files/dlnd_face_generation_23_229.png)


    Epoch 2/2... Discriminator Loss: 1.0603... Generator Loss: 0.8353



![png](dlnd_face_generation_files/dlnd_face_generation_23_231.png)


    Epoch 2/2... Discriminator Loss: 1.4997... Generator Loss: 0.4198



![png](dlnd_face_generation_files/dlnd_face_generation_23_233.png)


    Epoch 2/2... Discriminator Loss: 1.4217... Generator Loss: 0.5694



![png](dlnd_face_generation_files/dlnd_face_generation_23_235.png)


    Epoch 2/2... Discriminator Loss: 1.1395... Generator Loss: 0.8307



![png](dlnd_face_generation_files/dlnd_face_generation_23_237.png)


    Epoch 2/2... Discriminator Loss: 1.0447... Generator Loss: 0.9657



![png](dlnd_face_generation_files/dlnd_face_generation_23_239.png)



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-12-e0626622894d> in <module>()
         13 with tf.Graph().as_default():
         14     train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
    ---> 15           mnist_dataset.shape, mnist_dataset.image_mode)
    

    /usr/local/lib/python3.5/contextlib.py in __exit__(self, type, value, traceback)
         64         if type is None:
         65             try:
    ---> 66                 next(self.gen)
         67             except StopIteration:
         68                 return


    /usr/local/lib/python3.5/site-packages/tensorflow/python/framework/ops.py in get_controller(self, default)
       3682     finally:
       3683       if self._enforce_nesting:
    -> 3684         if self.stack[-1] is not default:
       3685           raise AssertionError(
       3686               "Nesting violated for default stack of %s objects"


    IndexError: list index out of range


### CelebA
Run your GANs on CelebA.  It will take around 20 minutes on the average GPU to run one epoch.  You can run the whole epoch or stop when it starts to generate realistic faces.


```python
batch_size = 64
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)
```

    Epoch 1/1... Discriminator Loss: 1.1199... Generator Loss: 0.9393



![png](dlnd_face_generation_files/dlnd_face_generation_25_1.png)


    Epoch 1/1... Discriminator Loss: 1.1366... Generator Loss: 1.1290



![png](dlnd_face_generation_files/dlnd_face_generation_25_3.png)


    Epoch 1/1... Discriminator Loss: 1.1737... Generator Loss: 1.0269



![png](dlnd_face_generation_files/dlnd_face_generation_25_5.png)


    Epoch 1/1... Discriminator Loss: 1.1446... Generator Loss: 0.8075



![png](dlnd_face_generation_files/dlnd_face_generation_25_7.png)


    Epoch 1/1... Discriminator Loss: 1.3214... Generator Loss: 1.1399



![png](dlnd_face_generation_files/dlnd_face_generation_25_9.png)


    Epoch 1/1... Discriminator Loss: 1.3272... Generator Loss: 0.7571



![png](dlnd_face_generation_files/dlnd_face_generation_25_11.png)


    Epoch 1/1... Discriminator Loss: 1.2673... Generator Loss: 1.2967



![png](dlnd_face_generation_files/dlnd_face_generation_25_13.png)


    Epoch 1/1... Discriminator Loss: 1.2217... Generator Loss: 1.0707



![png](dlnd_face_generation_files/dlnd_face_generation_25_15.png)


    Epoch 1/1... Discriminator Loss: 1.2543... Generator Loss: 1.5583



![png](dlnd_face_generation_files/dlnd_face_generation_25_17.png)


    Epoch 1/1... Discriminator Loss: 2.1798... Generator Loss: 0.2265



![png](dlnd_face_generation_files/dlnd_face_generation_25_19.png)


    Epoch 1/1... Discriminator Loss: 1.1831... Generator Loss: 0.8747



![png](dlnd_face_generation_files/dlnd_face_generation_25_21.png)


    Epoch 1/1... Discriminator Loss: 2.6119... Generator Loss: 0.1361



![png](dlnd_face_generation_files/dlnd_face_generation_25_23.png)


    Epoch 1/1... Discriminator Loss: 1.2133... Generator Loss: 0.9319



![png](dlnd_face_generation_files/dlnd_face_generation_25_25.png)


    Epoch 1/1... Discriminator Loss: 1.2162... Generator Loss: 1.1988



![png](dlnd_face_generation_files/dlnd_face_generation_25_27.png)


    Epoch 1/1... Discriminator Loss: 1.1761... Generator Loss: 0.7007



![png](dlnd_face_generation_files/dlnd_face_generation_25_29.png)


    Epoch 1/1... Discriminator Loss: 1.2599... Generator Loss: 0.9386



![png](dlnd_face_generation_files/dlnd_face_generation_25_31.png)


    Epoch 1/1... Discriminator Loss: 1.2342... Generator Loss: 1.1477



![png](dlnd_face_generation_files/dlnd_face_generation_25_33.png)


    Epoch 1/1... Discriminator Loss: 1.3855... Generator Loss: 1.6220



![png](dlnd_face_generation_files/dlnd_face_generation_25_35.png)


    Epoch 1/1... Discriminator Loss: 1.3282... Generator Loss: 0.7581



![png](dlnd_face_generation_files/dlnd_face_generation_25_37.png)


    Epoch 1/1... Discriminator Loss: 1.5564... Generator Loss: 0.5030



![png](dlnd_face_generation_files/dlnd_face_generation_25_39.png)


    Epoch 1/1... Discriminator Loss: 1.4330... Generator Loss: 1.8041



![png](dlnd_face_generation_files/dlnd_face_generation_25_41.png)


    Epoch 1/1... Discriminator Loss: 1.2861... Generator Loss: 1.6193



![png](dlnd_face_generation_files/dlnd_face_generation_25_43.png)


    Epoch 1/1... Discriminator Loss: 1.4025... Generator Loss: 0.5971



![png](dlnd_face_generation_files/dlnd_face_generation_25_45.png)


    Epoch 1/1... Discriminator Loss: 1.1753... Generator Loss: 1.1494



![png](dlnd_face_generation_files/dlnd_face_generation_25_47.png)


    Epoch 1/1... Discriminator Loss: 1.3331... Generator Loss: 2.4242



![png](dlnd_face_generation_files/dlnd_face_generation_25_49.png)


    Epoch 1/1... Discriminator Loss: 1.1333... Generator Loss: 1.5083



![png](dlnd_face_generation_files/dlnd_face_generation_25_51.png)


    Epoch 1/1... Discriminator Loss: 1.3714... Generator Loss: 0.6381



![png](dlnd_face_generation_files/dlnd_face_generation_25_53.png)


    Epoch 1/1... Discriminator Loss: 1.2080... Generator Loss: 1.2453



![png](dlnd_face_generation_files/dlnd_face_generation_25_55.png)


    Epoch 1/1... Discriminator Loss: 1.0594... Generator Loss: 0.8993



![png](dlnd_face_generation_files/dlnd_face_generation_25_57.png)


    Epoch 1/1... Discriminator Loss: 1.2910... Generator Loss: 0.7798



![png](dlnd_face_generation_files/dlnd_face_generation_25_59.png)


    Epoch 1/1... Discriminator Loss: 1.0642... Generator Loss: 1.1383



![png](dlnd_face_generation_files/dlnd_face_generation_25_61.png)



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-13-97293279c58c> in <module>()
         13 with tf.Graph().as_default():
         14     train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
    ---> 15           celeba_dataset.shape, celeba_dataset.image_mode)
    

    /usr/local/lib/python3.5/contextlib.py in __exit__(self, type, value, traceback)
         64         if type is None:
         65             try:
    ---> 66                 next(self.gen)
         67             except StopIteration:
         68                 return


    /usr/local/lib/python3.5/site-packages/tensorflow/python/framework/ops.py in get_controller(self, default)
       3682     finally:
       3683       if self._enforce_nesting:
    -> 3684         if self.stack[-1] is not default:
       3685           raise AssertionError(
       3686               "Nesting violated for default stack of %s objects"


    IndexError: list index out of range


### Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.


```python
print("Done")
```


```python

```
