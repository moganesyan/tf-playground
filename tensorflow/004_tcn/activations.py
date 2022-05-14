import tensorflow as tf


class ReLU(tf.Module):
    """
        Rectified Linear Unit (ReLU) activation.
    """

    def __init__(self, name: str = None) -> None:
        """
            Apply ReLU activation to input tensor.

            args:
                name: str - Name of the layer.
            returns:
                None
        """
        super().__init__(name)

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            args:
                x_in: tf.Tensor - Input tensor.
            returns:
                x_out: tf.Tensor - Output tensor.
        """

        return tf.nn.relu(x_in, name = 'relu')