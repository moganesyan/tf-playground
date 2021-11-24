from typing import Optional, Union, Tuple
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
                tf.random.normal((self.neurons,)),
                trainable = True,
                dtype = tf.float32,
                name = "dense_bias"
            )

            self._is_built = True

        return tf.matmul(x_in, self.W) + self.b


class Conv2D(tf.Module):
    """
        2D Convolution Layer.
    """

    def __init__(self,
                 filters_out: int,
                 kernel: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: str,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 data_format: str = "channels_last",
                 use_bias: bool = True,
                 name = None) -> None:

        super(Conv2D, self).__init__(name)

        self.filters_out: int = filters_out
        self.kernel: Union[int, Tuple[int, int]] = (kernel
            if isinstance(kernel, Tuple) else (kernel, kernel))
        self.stride: Union[int, Tuple[int, int]] = (stride
            if isinstance(stride, Tuple) else (stride, stride))
        self.padding: str = padding.upper()
        self.dilation: Union[int, Tuple[int, int]] = (dilation
            if isinstance(dilation, Tuple) else (dilation, dilation))
        self.data_format: str = ("NHWC"
            if data_format == "channels_last" else "NCHW")
        self.use_bias: bool = use_bias

        self.is_built: bool = False

        self.W: Optional[tf.Variable] = None
        self.b: Optional[tf.Variable] = None

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Build tensor on the first call.
            Calculate output by 2D convolution.
        """

        if not self.is_built:
            filters_in = x_in.shape[-1]
            filter_weights = self.kernel + (filters_in, self.filters_out)

            self.W = tf.Variable(
                tf.random.normal(filter_weights),
                trainable = True,
                dtype = tf.float32,
                name = "conv2d_filters"
            )
            self.b = tf.Variable(
                tf.random.normal((self.filters_out,)),
                trainable = True,
                dtype = tf.float32,
                name = "conv2d_bias"
            )

            self.is_built = True

        if self.use_bias:
            return tf.nn.conv2d(
                x_in, self.W, self.stride, self.padding,
                self.data_format, self.dilation, name = "conv2d_conv"
            ) + self.b
        else:
            return tf.nn.conv2d(
                x_in, self.W, self.stride, self.padding,
                self.data_format, self.dilation, name = "conv2d_conv"
            )
