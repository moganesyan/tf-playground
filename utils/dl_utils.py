import tensorflow as tf
from tensorflow.keras import layers


class ReLU6(layers.Layer):
    """
        Custom ReLU6 activation layer
    """

    def __init__(self) -> None:
        super(ReLU6, self).__init__()

    def call(self, input) -> layers.Layer:
        return tf.clip_by_value(
            input,
            clip_value_min = 0.0,
            clip_value_max = 6.0)
            