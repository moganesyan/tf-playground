from typing import Optional, Union
import tensorflow as tf


class Dense(tf.Module):
    """
        Dense Layer.
    """

    def __init__(self,
                 neurons: int,
                 name: Optional[str] = None):

        super(Dense, self).__init__(name)
        self.neurons: int = neurons
        self.is_built: bool = False

        self.W: Optional[tf.Variable] = None
        self.b: Optional[tf.Variable] = None

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Build tensor on the first call.
            Calculate output by matrix multiplication.
        """

        if not self.is_built:
            self.W = tf.Variable(
                tf.random.normal([x_in.shape[-1], self.neurons]),
                trainable = True,
                dtype = tf.float32,
                name = "dense_weights"
            )
            self.b = tf.Variable(
                tf.random.normal([self.neurons]),
                trainable = True,
                dtype = tf.float32,
                name = "dense_bias"
            )
            print(self.W)
            print(self.b)
            self._is_built = True

        x_out = tf.matmul(x_in, self.W) + self.b
        return x_out
