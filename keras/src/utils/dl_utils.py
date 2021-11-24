import tensorflow as tf
from tensorflow.keras import layers


class ReLU6(layers.Layer):
    """
        Custom ReLU6 activation layer
    """

    def __init__(self) -> None:
        super(ReLU6, self).__init__()

    def call(self, x_in: layers.Layer) -> layers.Layer:
        x_out = tf.maximum(tf.minimum(x_in, 6.0), 0)
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
        x_out = tf.multiply(x, 0.1666666667)
        return x_out


class HSwish(layers.Layer):
    """
        Hard SWISH function from the MobilenetV3 paper
    """

    def __init__(self) -> None:
        super(HSwish, self).__init__()

    def call(self, x_in: layers.Layer) -> layers.Layer:
        x = HSigm()(x_in)
        x_out = tf.multiply(x_in, x)
        return x_out


class StochasticDropout(layers.Layer):
    """
        Stochastic dropout layer.
        Multiplies values of input tensor by 0 in the train time based on
        survival probability parameter.
    """

    def __init__(self, survival_prob: float) -> None:
        super(StochasticDropout, self).__init__()
        self._survival_prob: float = survival_prob

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            '_survival_prob': self._survival_prob
        })
        return config

    def call(self, x_in: layers.Layer, training = None) -> layers.Layer:
        if not training:
            return x_in
        else:
            coin_toss = tf.random.uniform(
                [], dtype=tf.float32)
            x_out = tf.cond(
                tf.greater(coin_toss > self._survival_prob,
                lambda x: tf.multiply(x_in, 0),
                lambda x: tf.identity(x_in)))
            return x_out
