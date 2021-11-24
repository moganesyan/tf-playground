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
        self._neurons: int = neurons
        self._is_built: bool = False

        self.W: Optional[tf.Variable] = None
        self.b: Optional[tf.Variable] = None

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
            Build tensor on the first call.
            Calculate output by matrix multiplication.
        """

        if not self._is_built:
            self.W = tf.Variable(
                tf.random.normal((x.shape[-1], self._neurons)),
                trainable = True,
                dtype = tf.float32,
                name = "dense_weights"
            )
            self.b = tf.Variable(
                tf.random.normal((1, self._neurons)),
                trainable = True,
                dtype = tf.float32,
                name = "dense_bias"
            )
            self._is_built = True

        x_out = tf.matmul(x, self.W) + self.b
        return x_out
