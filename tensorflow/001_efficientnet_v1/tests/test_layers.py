import unittest
import tensorflow as tf

from utils import Dense


# Test Dense Layer
class TestDenseLayer(unittest.TestCase):
    """
        Test cases for the dense layer.
    """

    def test_output_shape(self):
        """
            Test the shape of the dense layer's output tensor.
        """

        x_in = tf.constant(
            tf.random.normal((224,224,3)),
            dtype = tf.float32
        )
        layer = Dense(100, "dense_test")
        x_out = layer(x_in)

        assert x_out.shape == (224,224,100)
