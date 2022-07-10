import unittest
from unittest.case import TestCase

import numpy as np
import tensorflow as tf

from layers import Dense, Conv1D
from layers import Flatten, GlobalMaxPooling1D
from layers import BatchNormalization


class TestDenseLayer(TestCase):
    """
        Tests for the dense layer.
    """

    def test_output_shape(self):
        """
            Test output shape on a standard case.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 200)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 100)),
            tf.float32
        )

        test_tensor = Dense(100)(tensor_in)

        assert tensor_out.shape == test_tensor.shape

    def test_high_dim_output(self):
        """
            Test output shape on a high dimensional input tensor.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 50, 200)),
            tf.float32
        )    

        tensor_out = tf.constant(
            tf.random.normal((32, 50, 100)),
            tf.float32
        )

        test_tensor = Dense(100)(tensor_in)

        assert tensor_out.shape == test_tensor.shape


class TestConv1DLayer(TestCase):
    """
        Tests for the 1D convolution layer.
    """

    def test_output_shape(self):
        """
            Test output shape for a standard case.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 50, 200)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 50, 100)),
            tf.float32
        )

        test_tensor = Conv1D(100, 1, 1, 1, 'same')(tensor_in)

        assert tensor_out.shape == test_tensor.shape

    @unittest.skip("Channels first conv not supported on CPU")
    def test_output_shape_inverted(self):
        """
            Test output shape for 'channels_first' mode.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 200, 50)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 100, 50)),
            tf.float32
        )

        test_tensor = Conv1D(100, 1, 1, 1, 'same', 'channels_first')(tensor_in)

        assert tensor_out.shape == test_tensor.shape

    def test_kernel(self):
        """
            Test convolution kernel.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 50, 200)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 48, 200)),
            tf.float32
        )

        test_tensor = Conv1D(200, 3, 1, 1, 'valid')(tensor_in)

        assert tensor_out.shape == test_tensor.shape

    def test_stride(self):
        """
            Test convolution stride.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 50, 200)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 25, 200)),
            tf.float32
        )

        test_tensor = Conv1D(200, 1, 2, 1, 'same')(tensor_in)

        assert tensor_out.shape == test_tensor.shape

    def test_dilation(self):
        """
            Test convolution dilation.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 50, 200)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 48, 200)),
            tf.float32
        )

        test_tensor = Conv1D(200, 2, 1, 2, 'valid')(tensor_in)

        assert tensor_out.shape == test_tensor.shape

    def test_causal_convolution(self):
        """
            Test causal convolution behaviour.
                - Test output shape
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 50, 200)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 50, 200)),
            tf.float32
        )

        test_tensor = Conv1D(200, 3, 1, 1, 'causal')(tensor_in)

        assert tensor_out.shape == test_tensor.shape


class TestFlattenLayer(TestCase):
    """
        Tests for the flatten layer.
    """

    def test_output_shape(self):
        """
            Test output shape on a standard case.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 200, 10)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 2000)),
            tf.float32
        )

        test_tensor = Flatten()(tensor_in)

        assert tensor_out.shape == test_tensor.shape


class TestGlabalMaxPooling1DLayer(TestCase):
    """
        Tests for the global 1D max pooling layer.
    """

    def test_output_shape(self):
        """
            Test output shape on a standard case.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 200, 10)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 10)),
            tf.float32
        )

        test_tensor = GlobalMaxPooling1D()(tensor_in)

        assert tensor_out.shape == test_tensor.shape

    @unittest.skip("Channels first maxpool not supported on CPU")
    def test_output_shape_inverted(self):
        """
            Test output shape for 'channels_first' mode.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 10, 200)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 10)),
            tf.float32
        )

        test_tensor = GlobalMaxPooling1D(data_format = 'channels_first')(tensor_in)

        assert tensor_out.shape == test_tensor.shape


class TestBatchNormalization(TestCase):
    """
        Tests for the batch normalization layer.
    """

    def test_output_shape(self):
        """
            Test output shape on a standard case.
                Normalization along 2 axis.
            Test output shape of the normalization axes.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 200, 10)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 200, 10)),
            tf.float32
        )

        bn = BatchNormalization()
        test_tensor = bn(tensor_in)

        assert tensor_out.shape == test_tensor.shape
        assert bn._axes == [0,1]

    def test_output_shape_inverted(self):
        """
            Test output shape on a non-standard case.
                Normalization along 1 axis.
            Test output shape of the normalization axes.
        """

        tensor_in = tf.constant(
            tf.random.normal((32, 200, 10)),
            tf.float32
        )

        tensor_out = tf.constant(
            tf.random.normal((32, 200, 10)),
            tf.float32
        )

        bn = BatchNormalization(axis = 1)
        test_tensor = bn(tensor_in)

        assert tensor_out.shape == test_tensor.shape
        assert bn._axes == [0,2]

    def test_inference_mode(self):
        """
            Test inference mode.
            Output must be (nearly) the same as input due to no training
                taking place prior (default 0 mean, 1 variance scaling).
        """

        tensor_in = tf.constant(
            3,
            tf.float32,
            (32, 200, 10)
        )

        tensor_out = tf.constant(
            3,
            tf.float32,
            (32, 200, 10)
        )

        # apply batch normalization
        bn = BatchNormalization()
        test_tensor = bn(tensor_in)

        # assert that the values are (near unchanged)
        np.testing.assert_almost_equal(
            tensor_out.numpy(), test_tensor.numpy(), decimal = 3)
