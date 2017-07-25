import os
import sys
import numpy as np
import tensorflow as tf
import random
import unittest
from unittest.mock import MagicMock

import my_answers

class ImageClassificationTest(unittest.TestCase):

    def test_normalize(self):
        """normalize is correctly implemented"""
        test_shape = (np.random.choice(range(1000)), 32, 32, 3)
        test_numbers = np.random.choice(range(256), test_shape)
        normalize_out = my_answers.normalize(test_numbers)

        self.assertEqual(type(normalize_out).__module__, np.__name__,\
                         'Not Numpy Object')

        self.assertEqual(normalize_out.shape, test_shape,\
                         'Incorrect Shape. {} shape found'.format(normalize_out.shape))

        self.assertTrue(normalize_out.max() <= 1 and normalize_out.min() >= 0,\
            'Incorrect Range. {} to {} found'.format(normalize_out.min(), normalize_out.max()))

    def test_one_hot_encode(self):
        """one_hot_encode is correctly implemented"""
        test_shape = np.random.choice(range(1000))
        test_numbers = np.random.choice(range(10), test_shape)
        one_hot_out = my_answers.one_hot_encode(test_numbers)

        self.assertEqual(type(one_hot_out).__module__, np.__name__,\
                         'Not Numpy Object')

        self.assertEqual(one_hot_out.shape, (test_shape, 10),\
                         'Incorrect Shape. {} shape found'.format(one_hot_out.shape))

        n_encode_tests = 5
        test_pairs = list(zip(test_numbers, one_hot_out))
        test_indices = np.random.choice(len(test_numbers), n_encode_tests)
        labels = [test_pairs[test_i][0] for test_i in test_indices]
        enc_labels = np.array([test_pairs[test_i][1] for test_i in test_indices])
        new_enc_labels = my_answers.one_hot_encode(labels)

        self.assertTrue(np.array_equal(enc_labels, new_enc_labels),\
            'Encodings returned different results for the same numbers.\n' \
            'For the first call it returned:\n' \
            '{}\n' \
            'For the second call it returned\n' \
            '{}\n' \
            'Make sure you save the map of labels to encodings outside of the function.'.format(enc_labels, new_enc_labels))

        for one_hot in new_enc_labels:
            self.assertEqual((one_hot==1).sum(), 1,\
                             'Each one-hot-encoded value should include the number 1 exactly once.\n' \
                             'Found {}\n'.format(one_hot))
            self.assertEqual((one_hot==0).sum(), len(one_hot)-1,\
                             'Each one-hot-encoded value should include zeros in all but one position.\n' \
                             'Found {}\n'.format(one_hot))

    def test_nn_image_inputs(self):
        """nn_image_inputs is correctly implemented"""
        image_shape = (32, 32, 3)
        nn_inputs_out_x = my_answers.neural_net_image_input(image_shape)

        self.assertEqual(nn_inputs_out_x.get_shape().as_list(), [None, image_shape[0], image_shape[1], image_shape[2]],\
                    'Incorrect Image Shape.  Found {} shape'.format(nn_inputs_out_x.get_shape().as_list()))

        self.assertEquals(nn_inputs_out_x.op.type,'Placeholder',\
                          'Incorrect Image Type.  Found {} type'.format(nn_inputs_out_x.op.type))

        self.assertEqual(nn_inputs_out_x.name, 'x:0', \
                         'Incorrect Name.  Found {}'.format(nn_inputs_out_x.name))

    def test_nn_label_inputs(self):
        """test_nn_label_inputs is correctly implemented"""
        n_classes = 10
        nn_inputs_out_y = my_answers.neural_net_label_input(n_classes)

        self.assertEqual(nn_inputs_out_y.get_shape().as_list(), [None, n_classes],\
            'Incorrect Label Shape.  Found {} shape'.format(nn_inputs_out_y.get_shape().as_list()))

        self.assertEqual(nn_inputs_out_y.op.type, 'Placeholder',\
            'Incorrect Label Type.  Found {} type'.format(nn_inputs_out_y.op.type))

        self.assertEqual(nn_inputs_out_y.name, 'y:0', \
            'Incorrect Name.  Found {}'.format(nn_inputs_out_y.name))

    def test_nn_keep_prob_inputs(self):
        """test_nn_keep_prob_inputs is correctly implemented"""
        nn_inputs_out_k = my_answers.neural_net_keep_prob_input()

        self.assertTrue(nn_inputs_out_k.get_shape().ndims is None,\
                        'Too many dimensions found for keep prob.  Found {} dimensions.  It should be a scalar (0-Dimension Tensor).'.format(nn_inputs_out_k.get_shape().ndims))

        self.assertEqual(nn_inputs_out_k.op.type, 'Placeholder',\
                         'Incorrect keep prob Type.  Found {} type'.format(nn_inputs_out_k.op.type))

        self.assertEqual(nn_inputs_out_k.name, 'keep_prob:0', \
                         'Incorrect Name.  Found {}'.format(nn_inputs_out_k.name))


    def test_con_pool(self):
        """con_pool is correctly implemented"""
        test_x = tf.placeholder(tf.float32, [None, 32, 32, 5])
        test_num_outputs = 10
        test_con_k = (2, 2)
        test_con_s = (4, 4)
        test_pool_k = (2, 2)
        test_pool_s = (2, 2)

        conv2d_maxpool_out = my_answers.conv2d_maxpool(test_x, test_num_outputs, test_con_k, test_con_s, test_pool_k, test_pool_s)

        self.assertEqual(conv2d_maxpool_out.get_shape().as_list(), [None, 4, 4, 10],\
                         'Incorrect Shape.  Found {} shape'.format(conv2d_maxpool_out.get_shape().as_list()))

    def test_flatten(self):
        """flatten is correctly implemented"""
        test_x = tf.placeholder(tf.float32, [None, 10, 30, 6])
        flat_out = my_answers.flatten(test_x)

        self.assertEqual(flat_out.get_shape().as_list(), [None, 10*30*6],\
                         'Incorrect Shape.  Found {} shape'.format(flat_out.get_shape().as_list()))

    def test_fully_conn(self):
        """fully_conn is correctly implemented"""
        test_x = tf.placeholder(tf.float32, [None, 128])
        test_num_outputs = 40

        fc_out = my_answers.fully_conn(test_x, test_num_outputs)

        self.assertEqual(fc_out.get_shape().as_list(), [None, 40],\
                         'Incorrect Shape.  Found {} shape'.format(fc_out.get_shape().as_list()))

    def test_output(self):
        """output is correctly implemented"""
        test_x = tf.placeholder(tf.float32, [None, 128])
        test_num_outputs = 40

        output_out = my_answers.output(test_x, test_num_outputs)

        self.assertEqual(output_out.get_shape().as_list(), [None, 40],\
                         'Incorrect Shape.  Found {} shape'.format(output_out.get_shape().as_list()))

    def test_conv_net(self):
        """conv_net is correctly implemented"""
        test_x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        test_k = tf.placeholder(tf.float32)

        logits_out = my_answers.conv_net(test_x, test_k)

        self.assertEqual(logits_out.get_shape().as_list(), [None, 10],\
                         'Incorrect Model Output.  Found {}'.format(logits_out.get_shape().as_list()))

    def test_train_nn(self):
        """train_nn invokes a tf session"""
        mock_session = tf.Session()
        test_x = np.random.rand(128, 32, 32, 3)
        test_y = np.random.rand(128, 10)
        test_k = np.random.rand(1)
        test_optimizer = tf.train.AdamOptimizer()

        mock_session.run = MagicMock()
        my_answers.train_neural_network(mock_session, test_optimizer, test_k, test_x, test_y)

        self.assertTrue(mock_session.run.called, 'Session not used')

def run(test_name):
    unittest.TextTestRunner().run(ImageClassificationTest(test_name))

if __name__ == '__main__':
    run('test_normalize')
