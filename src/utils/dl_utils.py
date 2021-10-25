import tensorflow as tf
from tensorflow.keras import layers


class ReLU6(layers.Layer):
    """
        Custom ReLU6 activation layer
    """

    def __init__(self) -> None:
        super(ReLU6, self).__init__()

    def call(self, x_in: layers.Layer) -> layers.Layer:
        x_out = tf.clip_by_value(x_in, 0.0, 6.0)
        return x_out


class HSigm(layers.Layer):
    """
        Hard Sigmoid from the MobilenetV3 paper
    """

    def __init__(self) -> None:
        super(HSigm, self).__init__()

    def call(self, x_in: layers.Layer) -> layers.Layer:
        x = tf.add(x_in, 3.0)
        x = ReLU6()(x)
        x_out = tf.multiply(x, 1/6)
        return x_out


class HSwish(layers.Layer):
    """
        Hard SWISH function from the MobilenetV3 paper
    """

    def __init__(self) -> None:
        super(HSwish, self).__init__()

    def call(self, x_in: layers.Layer) -> layers.Layer:
        x = HSigm()(x_in)
        x_out = layers.Multiply()([x_in, x]) 
        return x_out