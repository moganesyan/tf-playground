from typing import Optional, Union, Tuple

import tensorflow as tf


class ReLU6(tf.Module):
    """
        ReLU6 activation function.
    """

    def __init__(self, name: str = None):
        super(ReLU6, self).__init__(name)

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Calculate ReLU6 by min and max functions.
        """

        x = tf.maximum(0.0, x_in)
        return tf.minimum(6.0, x)


class HSigm(tf.Module):
    """
        Hard sigmoid activation function.
    """

    def __init__(self, name=None):
        super(HSigm, self).__init__(name=name)

        self.relu6 = ReLU6("relu6")

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Calculate the hard sigmoid by shifting along x axis and scaling
            along the y axis.
        """

        x = tf.add(x_in, 3.0)
        x = self.relu6(x)
        return tf.multiply(x, 0.166666667)


class HSwish(tf.Module):
    """
        HSwish activation function.
    """

    def __init__(self, name=None):
        super(HSwish, self).__init__(name=name)

        self.hsigm = HSigm("hsigm")

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Calculate the hard swish by multiplying the output of the 
            hard sigmoid by the input.
        """

        x = self.hsigm(x_in)
        return tf.multiply(x_in, x)
