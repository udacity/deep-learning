# Deep Learning Nanodegree Foundation

This repository contains material related to Udacity's [Deep Learning Nanodegree Foundation](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101) program. It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks lead you through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight intialization and batch normalization.

There are also notebooks used as projects for the Nanodegree program. In the program itself, the projects are reviewed by Udacity experts, but they are available here as well.

## Table Of Contents

### Tutorials

* [Sentiment Analysis with Numpy](https://github.com/udacity/deep-learning/tree/master/sentiment-network): [Andrew Trask](http://iamtrask.github.io/) leads you through building a sentiment analysis model, predicting if some text is positive or negative.
* [Intro to TensorFlow](https://github.com/udacity/deep-learning/tree/master/intro-to-tensorflow): Starting building neural networks with Tensorflow.
* [Weight Intialization](https://github.com/udacity/deep-learning/tree/master/weight-initialization): Explore how initializing network weights affects performance.
* [Autoencoders](https://github.com/udacity/deep-learning/tree/master/autoencoder): Build models for image compression and denoising, using feed-forward and convolution networks in TensorFlow.
* [Transfer Learning (ConvNet)](https://github.com/udacity/deep-learning/tree/master/transfer-learning). In practice, most people don't train their own large networkd on huge datasets, but use pretrained networks such as VGGnet. Here you'll use VGGnet to classify images of flowers without training a network on the images themselves.
* [Intro to Recurrent Networks (Character-wise RNN)](https://github.com/udacity/deep-learning/tree/master/intro-to-rnns): Recurrent neural networks are able to use information about the sequence of data, such as the sequence of characters in text.
* [Embeddings (Word2Vec)](https://github.com/udacity/deep-learning/tree/master/embeddings): Implement the Word2Vec model to find semantic representations of words for use in natural language processing.
* [Sentiment Analysis RNN](https://github.com/udacity/deep-learning/tree/master/sentiment-rnn): Implement a recurrent neural network that can predict if a text sample is positive or negative.
* [Tensorboard](https://github.com/udacity/deep-learning/tree/master/tensorboard): Use TensorBoard to visualize the network graph, as well as how parameters change through training.
* [Reinforcement Learning (Q-Learning)](https://github.com/udacity/deep-learning/tree/master/reinforcement): Implement a deep Q-learning network to play a simple game from OpenAI Gym.
* [Sequence to sequence](https://github.com/udacity/deep-learning/tree/master/seq2seq): Implement a sequence-to-sequence recurrent network.
* [Batch normalization](https://github.com/udacity/deep-learning/tree/master/batch-norm): Learn how to improve training rates and network stability with batch normalizations.
* [Generative Adversatial Network on MNIST](https://github.com/udacity/deep-learning/tree/master/gan_mnist): Train a simple generative adversarial network on the MNIST dataset.
* [Deep Convolutional GAN (DCGAN)](https://github.com/udacity/deep-learning/tree/master/dcgan-svhn): Implement a DCGAN to generate new images based on the Street View House Numbers (SVHN) dataset.
* [Intro to TFLearn](https://github.com/udacity/deep-learning/tree/master/intro-to-tflearn): A couple introductions to a high-level library for building neural networks.

### Projects

* [Your First Neural Network](https://github.com/udacity/deep-learning/tree/master/first-neural-network): Implement a neural network in Numpy to predict bike rentals.
* [Image classification](https://github.com/udacity/deep-learning/tree/master/image-classification): Build a convolutional neural network with TensorFlow to classify CIFAR-10 images.
* [Text Generation](https://github.com/udacity/deep-learning/tree/master/tv-script-generation): Train a recurrent neural network on scripts from The Simpson's (copyright Fox) to generate new scripts.
* [Machine Translation](https://github.com/udacity/deep-learning/tree/master/language-translation): Train a sequence to sequence network for English to French translation (on a simple dataset)
* [Face Generation](https://github.com/udacity/deep-learning/tree/master/face_generation): Use a DCGAN on the CelebA dataset to generate images of novel and realistic human faces.


## Dependencies

Each directory has a `requirements.txt` describing the minimal dependencies required to run the notebooks in that directory.

### pip

To install these dependencies with pip, you can issue `pip3 install -r requirements.txt`.

### Conda Environments

You can find Conda environment files for the Deep Learning program in the `environments` folder. Note that environment files are platform dependent. Versions with `tensorflow-gpu` are labeled in the filename with "GPU".