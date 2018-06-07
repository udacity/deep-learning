from copy import deepcopy
from unittest import mock
import tensorflow as tf


def test_safe(func):
    """
    Isolate tests
    """
    def func_wrapper(*args):
        with tf.Graph().as_default():
            result = func(*args)
        print('Tests Passed')
        return result

    return func_wrapper


def _assert_tensor_shape(tensor, shape, display_name):
    assert tf.assert_rank(tensor, len(shape), message='{} has wrong rank'.format(display_name))

    tensor_shape = tensor.get_shape().as_list() if len(shape) else []

    wrong_dimension = [ten_dim for ten_dim, cor_dim in zip(tensor_shape, shape)
                       if cor_dim is not None and ten_dim != cor_dim]
    assert not wrong_dimension, \
        '{} has wrong shape.  Found {}'.format(display_name, tensor_shape)


def _check_input(tensor, shape, display_name, tf_name=None):
    assert tensor.op.type == 'Placeholder', \
        '{} is not a Placeholder.'.format(display_name)

    _assert_tensor_shape(tensor, shape, 'Real Input')

    if tf_name:
        assert tensor.name == tf_name, \
            '{} has bad name.  Found name {}'.format(display_name, tensor.name)


class TmpMock():
    """
    Mock a attribute.  Restore attribute when exiting scope.
    """
    def __init__(self, module, attrib_name):
        self.original_attrib = deepcopy(getattr(module, attrib_name))
        setattr(module, attrib_name, mock.MagicMock())
        self.module = module
        self.attrib_name = attrib_name

    def __enter__(self):
        return getattr(self.module, self.attrib_name)

    def __exit__(self, type, value, traceback):
        setattr(self.module, self.attrib_name, self.original_attrib)


@test_safe
def test_model_inputs(model_inputs):
    image_width = 28
    image_height = 28
    image_channels = 3
    z_dim = 100
    input_real, input_z, learn_rate = model_inputs(image_width, image_height, image_channels, z_dim)

    _check_input(input_real, [None, image_width, image_height, image_channels], 'Real Input')
    _check_input(input_z, [None, z_dim], 'Z Input')
    _check_input(learn_rate, [], 'Learning Rate')


@test_safe
def test_discriminator(discriminator, tf_module):
    with TmpMock(tf_module, 'variable_scope') as mock_variable_scope:
        image = tf.placeholder(tf.float32, [None, 28, 28, 3])

        output, logits = discriminator(image)
        _assert_tensor_shape(output, [None, 1], 'Discriminator Training(reuse=false) output')
        _assert_tensor_shape(logits, [None, 1], 'Discriminator Training(reuse=false) Logits')
        assert mock_variable_scope.called,\
            'tf.variable_scope not called in Discriminator Training(reuse=false)'
        assert mock_variable_scope.call_args == mock.call('discriminator', reuse=False), \
            'tf.variable_scope called with wrong arguments in Discriminator Training(reuse=false)'

        mock_variable_scope.reset_mock()

        output_reuse, logits_reuse = discriminator(image, True)
        _assert_tensor_shape(output_reuse, [None, 1], 'Discriminator Inference(reuse=True) output')
        _assert_tensor_shape(logits_reuse, [None, 1], 'Discriminator Inference(reuse=True) Logits')
        assert mock_variable_scope.called, \
            'tf.variable_scope not called in Discriminator Inference(reuse=True)'
        assert mock_variable_scope.call_args == mock.call('discriminator', reuse=True), \
            'tf.variable_scope called with wrong arguments in Discriminator Inference(reuse=True)'


@test_safe
def test_generator(generator, tf_module):
    with TmpMock(tf_module, 'variable_scope') as mock_variable_scope:
        z = tf.placeholder(tf.float32, [None, 100])
        out_channel_dim = 5

        output = generator(z, out_channel_dim)
        _assert_tensor_shape(output, [None, 28, 28, out_channel_dim], 'Generator output (is_train=True)')
        assert mock_variable_scope.called, \
            'tf.variable_scope not called in Generator Training(reuse=false)'
        assert mock_variable_scope.call_args == mock.call('generator', reuse=False), \
            'tf.variable_scope called with wrong arguments in Generator Training(reuse=false)'

        mock_variable_scope.reset_mock()
        output = generator(z, out_channel_dim, False)
        _assert_tensor_shape(output, [None, 28, 28, out_channel_dim], 'Generator output (is_train=False)')
        assert mock_variable_scope.called, \
            'tf.variable_scope not called in Generator Inference(reuse=True)'
        assert mock_variable_scope.call_args == mock.call('generator', reuse=True), \
            'tf.variable_scope called with wrong arguments in Generator Inference(reuse=True)'


@test_safe
def test_model_loss(model_loss):
    out_channel_dim = 4
    input_real = tf.placeholder(tf.float32, [None, 28, 28, out_channel_dim])
    input_z = tf.placeholder(tf.float32, [None, 100])

    d_loss, g_loss = model_loss(input_real, input_z, out_channel_dim)

    _assert_tensor_shape(d_loss, [], 'Discriminator Loss')
    _assert_tensor_shape(d_loss, [], 'Generator Loss')


@test_safe
def test_model_opt(model_opt, tf_module):
    with TmpMock(tf_module, 'trainable_variables') as mock_trainable_variables:
        with tf.variable_scope('discriminator'):
            discriminator_logits = tf.Variable(tf.zeros([3, 3]))
        with tf.variable_scope('generator'):
            generator_logits = tf.Variable(tf.zeros([3, 3]))

        mock_trainable_variables.return_value = [discriminator_logits, generator_logits]
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_logits,
            labels=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=generator_logits,
            labels=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
        learning_rate = 0.001
        beta1 = 0.9

        d_train_opt, g_train_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
        assert mock_trainable_variables.called,\
            'tf.mock_trainable_variables not called'


