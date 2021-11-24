from typing import Union
import unittest
import tensorflow as tf

from utils import Dense, Conv2D


class TestDenseLayer(unittest.TestCase):
    """
        Test cases for the dense layer.
    """

    def test_output_shape(self):
        """
            Test the shape of the dense layer's output tensor.
        """

        x_in = tf.constant(
            tf.random.normal((1,224,224,3)),
            dtype = tf.float32
        )
        layer = Dense(100, "dense_test")
        x_out = layer(x_in)

        assert x_out.shape == (1,224,224,100)


class TestConv2DLAYER(unittest.TestCase):
    """
        Test cases for the Conv2D layer.
    """

    def test_output_shape(self):
        """
            Test the shape of the conv2d layer's output tensor.
        """

        x_in = tf.constant(
            tf.random.normal((1,224,224,3)),
            dtype = tf.float32
        )

        layer = Conv2D(
            32, 3, 1, 'same', name = "conv2d_test"
        )

        x_out = layer(x_in)

        assert x_out.shape == (1,224,224,32)

    def test_valid_padding(self):
        """
            Test 'valid' padding type.
        """

        x_in = tf.constant(
            tf.random.normal((1,224,224,3)),
            dtype = tf.float32
        )

        layer = Conv2D(
            32, 3, 1, 'valid', name = "conv2d_test"
        )

        x_out = layer(x_in)

        assert x_out.shape == (1,222,222,32)

    def test_stride(self):
        """
            Test stride.
        """

        x_in = tf.constant(
            tf.random.normal((1,224,224,3)),
            dtype = tf.float32
        )

        layer = Conv2D(
            32, 3, 2, 'same', name = "conv2d_test"
        )

        x_out = layer(x_in)

        assert x_out.shape == (1,112,112,32)  
