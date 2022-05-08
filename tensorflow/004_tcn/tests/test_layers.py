from typing import Union
import unittest
from unittest.case import TestCase
import tensorflow as tf

from layers import Dense, Conv1D, Flatten, GlobalMaxPooling1D


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
