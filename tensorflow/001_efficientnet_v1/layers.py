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

            self.is_built = True

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
        self.kernel: Tuple[int, int] = (kernel
            if isinstance(kernel, Tuple) else (kernel, kernel))
        self.stride: Tuple[int, int] = (stride
            if isinstance(stride, Tuple) else (stride, stride))
        self.padding: str = padding.upper()
        self.dilation: Tuple[int, int] = (dilation
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
            filters_in = x_in.shape[3] if self.data_format == "NHWC" else x_in.shape[1]
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


class DepthwiseConv2D(tf.Module):
    """
        2D Depthwise Convolution Layer.
    """

    def __init__(self,
                 kernel: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int, int, int]],
                 padding: str,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 depth_multiplier: int = 1,
                 data_format: str = "channels_last",
                 use_bias: bool = True,
                 name = None) -> None:

        super(DepthwiseConv2D, self).__init__(name)

        self.kernel: Tuple[int, int] = (kernel
            if isinstance(kernel, Tuple) else (kernel, kernel))
        stride_tuple = ((1, stride, stride, 1)
            if data_format == "channels_last" else (1, 1, stride, stride))
        self.stride: Tuple[int, int, int, int ] = (stride
            if isinstance(stride, Tuple) else stride_tuple)
        self.padding: str = padding.upper()
        self.dilation: Tuple[int, int] = (dilation
            if isinstance(dilation, Tuple) else (dilation, dilation))
        self.depth_multiplier: int = depth_multiplier
        self.data_format: str = ("NHWC"
            if data_format == "channels_last" else "NCHW")
        self.use_bias: bool = use_bias

        self.is_built: bool = False

        self.W: Optional[tf.Variable] = None
        self.b: Optional[tf.Variable] = None

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Build tensor on first call.
            Calculate output by 2D Depthwise Convolutions.
        """

        if not self.is_built:
            filters_in = x_in.shape[3] if self.data_format == "NHWC" else x_in.shape[1]
            filter_weights = self.kernel + (filters_in, self.depth_multiplier)

            self.W = tf.Variable(
                tf.random.normal(filter_weights),
                trainable = True,
                dtype = tf.float32,
                name = "dwise_conv2d_filters"
            )
            self.b = tf.Variable(
                tf.random.normal((filters_in * self.depth_multiplier,)),
                trainable = True,
                dtype = tf.float32,
                name = "dwise_conv2d_bias"
            )
            self.is_built = True

        if self.use_bias:
            return tf.nn.depthwise_conv2d(
                x_in, self.W, self.stride, self.padding,
                self.data_format, self.dilation, name = "dwise_conv2d_conv"
            ) + self.b
        else:
            return tf.nn.depthwise_conv2d(
                x_in, self.W, self.stride, self.padding,
                self.data_format, self.dilation, name = "dwise_conv2d_conv"
            )


class GlobalAveragePooling2D(tf.Module):
    """
        2D Globla Average Pooling layer.
    """

    def __init__(self,
                 data_format: str = "channels_last",
                 name: str = None):

        super(GlobalAveragePooling2D, self).__init__(name)

        self.data_format: str = "NHWC" if data_format == "channels_last" else "NCHW"

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Do global average pooling across channels.
        """

        if self.data_format == "NHWC":
            (batch, height, width, channels) = x_in.shape
        else:
            (batch, channels, height, width) = x_in.shape

        kernel = (height, width)
        strides = (1, 1)
        padding = "VALID"

        x_pooled = tf.nn.avg_pool2d(
            x_in, kernel, strides,
            padding, self.data_format,
            name = "global_avg_pool_2d"
        )
        return tf.reshape(
            x_pooled, (batch, channels),
            name = "global_avg_pool_2d_reshape")


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
