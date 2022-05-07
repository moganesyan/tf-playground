from typing import Optional, Union, Tuple, List

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

    @tf.Module.with_name_scope
    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Build tensor on the first call.
            Calculate output by matrix multiplication.
        """

        if not self.is_built:
            self.W = tf.Variable(
                tf.initializers.GlorotUniform()([x_in.shape[-1], self.neurons]),
                trainable = True,
                dtype = tf.float32,
                name = "dense_weights"
            )
            self.b = tf.Variable(
                tf.initializers.GlorotUniform()((self.neurons,)),
                trainable = True,
                dtype = tf.float32,
                name = "dense_bias"
            )

            self.is_built = True

        return tf.add(
            tf.matmul(x_in, self.W, name='dense_matmul'),
            self.b,
            name = 'dense_add_bias')


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

    @tf.Module.with_name_scope
    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Build tensor on the first call.
            Calculate output by 2D convolution.
        """

        if not self.is_built:
            filters_in = x_in.shape[3] if self.data_format == "NHWC" else x_in.shape[1]
            filter_weights = self.kernel + (filters_in, self.filters_out)

            self.W = tf.Variable(
                tf.initializers.GlorotUniform()(filter_weights),
                trainable = True,
                dtype = tf.float32,
                name = "conv2d_filters"
            )
            if self.use_bias:
                self.b = tf.Variable(
                    tf.initializers.GlorotUniform()((self.filters_out,)),
                    trainable = True,
                    dtype = tf.float32,
                    name = "conv2d_bias"
                )

            self.is_built = True

        if self.use_bias:
            return tf.add(
                tf.nn.conv2d(
                    x_in, self.W, self.stride, self.padding,
                    self.data_format, self.dilation, name = 'conv2d_conv'),
                    self.b,
                    name = 'conv2d_add_bias'
                )
        else:
            return tf.nn.conv2d(
                x_in, self.W, self.stride, self.padding,
                self.data_format, self.dilation, name = 'conv2d_conv'
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

    @tf.Module.with_name_scope
    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Build tensor on first call.
            Calculate output by 2D Depthwise Convolutions.
        """

        if not self.is_built:
            filters_in = x_in.shape[3] if self.data_format == "NHWC" else x_in.shape[1]
            filter_weights = self.kernel + (filters_in, self.depth_multiplier)

            self.W = tf.Variable(
                tf.initializers.GlorotUniform()(filter_weights),
                trainable = True,
                dtype = tf.float32,
                name = "dwise_conv2d_filters"
            )
            if self.use_bias:
                self.b = tf.Variable(
                    tf.initializers.GlorotUniform()((filters_in * self.depth_multiplier,)),
                    trainable = True,
                    dtype = tf.float32,
                    name = "dwise_conv2d_bias"
                )
            self.is_built = True

        if self.use_bias:
            return tf.add(
                tf.nn.depthwise_conv2d(
                    x_in, self.W, self.stride, self.padding,
                    self.data_format, self.dilation, name = "dwise_conv2d_conv"),
                self.b,
                name = 'dwise_conv2d_add_bias'
            )
        else:
            return tf.nn.depthwise_conv2d(
                x_in, self.W, self.stride, self.padding,
                self.data_format, self.dilation, name = "dwise_conv2d_conv"
            )


class GlobalAveragePooling2D(tf.Module):
    """
        2D Global Average Pooling layer.
    """

    def __init__(self,
                 data_format: str = "channels_last",
                 name: str = None):

        super(GlobalAveragePooling2D, self).__init__(name)

        self.data_format: str = "NHWC" if data_format == "channels_last" else "NCHW"

    @tf.Module.with_name_scope
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


class BatchNormalization(tf.Module):
    """
        Batch Normalization function.
    """

    def __init__(self,
                 axis: int = -1,
                 momentum: float = 0.99,
                 name: str = None):
        super(BatchNormalization, self).__init__(name)

        self.axis: int = axis
        self.momentum: float = momentum

        self.beta_weights: Optional[tf.Tensor] = None
        self.gamma_weights: Optional[tf.Tensor] = None
        self.axes: Optional[List] = None

        self.mean_ma: Optional[tf.Tensor] = None
        self.var_ma: Optional[tf.Tensor] = None
        self.epsilon: tf.Tensor = tf.constant(
            0.0000001, dtype = tf.float32,
            name = 'epsilon')

        self.is_built: bool = False

    @tf.Module.with_name_scope
    def __call__(self, x_in: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
            Apply batch normalization procedure.
        """

        if not self.is_built:
            self.beta_weights = tf.Variable(
                tf.zeros(x_in.shape[1:], tf.float32),
                trainable = True,
                name = "bn_beta",
                dtype = tf.float32)

            self.gamma_weights = tf.Variable(
                tf.ones(x_in.shape[1:], tf.float32),
                trainable = True,
                name = "bn_gamma",
                dtype = tf.float32)

            if self.axis == -1:
                self.axes = list(range(0, len(x_in.shape) - 1))
                ma_shape = [1] * (len(x_in.shape) - 1) + [x_in.shape[-1]]
            else:                
                self.axes = list(range(0, self.axis)) + list(range(self.axis + 1, len(x_in.shape)))
                ma_shape = [1] * len(x_in.shape[:self.axis]) + [x_in.shape[self.axis]] + [1] * len(x_in.shape[(self.axis + 1):])

            self.mean_ma = tf.Variable(
                tf.zeros(ma_shape, tf.float32),
                trainable = False,
                name = "bn_mean_ma",
                dtype = tf.float32
            )

            self.var_ma = tf.Variable(
                tf.ones(ma_shape, tf.float32),
                trainable = False,
                name = "bn_var_ma",
                dtype = tf.float32
            )

            self.momentum_const = tf.constant(
                self.momentum, dtype = tf.float32,
                shape = ma_shape, name = "bn_momentum"
            )

            self.is_built = True

        if training:
            x_mean, x_var = tf.nn.moments(
                x_in, self.axes, name = "get_batch_moments")
            x_out = (self.gamma_weights * (x_in - x_mean) / tf.math.sqrt(x_var + self.epsilon)) + self.beta_weights

            self.mean_ma.assign((self.mean_ma * self.momentum_const) + (x_mean * (1 - self.momentum_const)))
            self.var_ma.assign((self.var_ma * self.momentum_const) + (x_var * (1 - self.momentum_const)))
        else:
            x_out = (self.gamma_weights * (x_in - self.mean_ma) / tf.math.sqrt(self.var_ma + self.epsilon)) + self.beta_weights

        return x_out
