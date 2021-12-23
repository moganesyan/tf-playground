from typing import Union
import unittest
import tensorflow as tf

from utils import Dense, Conv2D, DepthwiseConv2D


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


class TestConv2DLayer(unittest.TestCase):
    """
        Test cases for the Conv2D layer.
    """

    def test_output_shape(self):
        """
            Test the output shape.
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

    @unittest.skip
    def test_output_shape_inverted(self):
        """
            Test the shape of the conv2d layer's output tensor.
            NCHW format.
        """

        x_in = tf.constant(
            tf.random.normal((1,3,224,224)),
            dtype = tf.float32
        )

        layer = Conv2D(
            32, 3, 1, 'same',
            data_format = "channels_first",
            name = "conv2d_test"
        )

        x_out = layer(x_in)

        assert x_out.shape == (1,32,224,224)


class TestDepthwiseConv2DLayer(unittest.TestCase):
    """
        Test cases for the Depthwise Conv2D layer.
    """

    def test_output_shape(self):
        """
            Test the output shape.
        """

        x_in = tf.constant(
            tf.random.normal((1,224,224,3)),
            dtype = tf.float32
        )

        layer = DepthwiseConv2D(
            3, 1, 'same', name = "dwise_conv2d_test"
        )

        x_out = layer(x_in)

        assert x_out.shape == (1,224,224,3)

    def test_depth_multiplier(self):
        """
            Test the depth multiplier.
        """

        x_in = tf.constant(
            tf.random.normal((1,224,224,3)),
            dtype = tf.float32
        )

        layer = DepthwiseConv2D(
            3, 1, 'same',
            depth_multiplier = 2,
            name = "dwise_conv2d_test"
        )

        x_out = layer(x_in)

        assert x_out.shape == (1,224,224,6)

    def test_valid_padding(self):
        """
            Test 'valid' padding type.
        """

        x_in = tf.constant(
            tf.random.normal((1,224,224,3)),
            dtype = tf.float32
        )

        layer = DepthwiseConv2D(
            3, 1, 'valid', name = "dwise_conv2d_test"
        )

        x_out = layer(x_in)

        assert x_out.shape == (1,222,222,3)

    def test_stride(self):
        """
            Test stride.
        """

        x_in = tf.constant(
            tf.random.normal((1,224,224,3)),
            dtype = tf.float32
        )

        layer = DepthwiseConv2D(
            3, 2, 'same', name = "dwise_conv2d_test"
        )

        x_out = layer(x_in)

        assert x_out.shape == (1,112,112,3)

    @unittest.skip
    def test_output_shape_inverted(self):
        """
            Test the shape of the conv2d layer's output tensor.
            NCHW format.
        """

        x_in = tf.constant(
            tf.random.normal((1,3,224,224)),
            dtype = tf.float32
        )

        layer = DepthwiseConv2D(
            3, 1, 'same',
            data_format = "channels_first",
            name = "dwise_conv2d_test"
        )

        x_out = layer(x_in)

        assert x_out.shape == (1,3,224,224)
