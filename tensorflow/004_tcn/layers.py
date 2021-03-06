from typing import Tuple, List, Optional, Union
from pytz import NonExistentTimeError
from scipy import rand

import numpy as np
import tensorflow as tf


class Dense(tf.Module):
    """
        Fully connected dense layer.
    """

    def __init__(self, neurons: int, name: str = None) -> None:
        """
            Build the weights and bias variables on the first call.
            Calculate output tensor through matrix multiplication and add bias.

            args:
                neurons: int - Number of neurons for the fully connected layer.
                name: str - Name of the layer.
            returns:
                None
        """

        super(Dense, self).__init__(name = name)
        self._neurons = neurons

        # do not initialise weights at init
        self.W = None
        self.b = None

        # is built flag for dynamic input size inference
        self._is_built = False

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            args:
                x_in: tf.Tensor - Input tensor of dimension (None, input_size).
            returns:
                x_out: tf.Tensor - Output tensor of dimension (None, neurons).
        """

        if not self._is_built:
            weights_shape = (x_in.shape[-1], self._neurons)

            self.W = tf.Variable(
                initial_value = tf.initializers.GlorotNormal()(weights_shape),
                trainable = True,
                name = "dense_weights",
                dtype = tf.float32
            )

            bias_shape = (self._neurons,)
            self.b = tf.Variable(
                initial_value = tf.initializers.GlorotNormal()(bias_shape),
                trainable = True,
                name = "dense_bias",
                dtype = tf.float32
            )

            self._is_built = True

        x_out = tf.matmul(x_in, self.W) + self.b
        return x_out


class Conv1D(tf.Module):
    """
        1D convolutional layer.
    """

    def __init__(self,
                 n_filters: int,
                 kernel_size: int,
                 stride: int,
                 dilation_rate: int,
                 padding: str,
                 data_format: str = "channels_last",
                 use_bias: bool = True,
                 name: str = None) -> None:
        """
            Calculate convolution dimensions on the first call.
            Return feature maps after 1D convolution operation.

            args:
                n_filters: int - Number of convolutional filters.
                kernel_size: int - Convolutional filter size.
                stride: int - Convolutional stride size.
                dilation_rate: int - Convolutional dilation size.
                padding: str - Padding mode ['same', 'valid', 'causal'].
                data_format: str - Input dimension order ['channels_last', 'channels_first'].
                use_bias: bool - Toggle using bias.
                name: str - Name of the layer.
            returns:
                None
        """

        super(Conv1D, self).__init__(name)
        self._n_filters = n_filters
        self._kernel_size = kernel_size
        self._stride = stride
        self._dilation_rate = dilation_rate

        assert padding in ('same', 'valid', 'causal'), "Invalid padding type."
        self._padding = padding

        assert data_format in ('channels_last', 'channels_first'), "Invalid data format."
        self._data_format = 'NWC' if data_format == 'channels_last' else 'NCW'

        self._use_bias = use_bias

        # do not initialise weights at init
        self.W = None
        self.b = None

        # conv kernel mask for simulating dilated convolutions
        # self._conv_kernel_mask = None

        # is built flag for dynamic input size inference
        self._is_built = False

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            args:
                x_in: tf.Tensor - Input tensor of dimension (None, input_size, in_channels).
            returns:
                x_out: tf.Tensor - Output tensor of dimension (None, output_size, out_channels).
        """

        if not self._is_built:
            assert len(x_in.shape) == 3, f"Input tensor rank must be 3, given {len(x_in.shape)}"

            filters_in = x_in.shape[2] if self._data_format == "NWC" else x_in.shape[1]

            # custom dilation to avoid size to batch and batch to size conversions

            # if self._dilation_rate > 1:
            #     kernel_size_new = ((self._kernel_size - 1) * self._dilation_rate) + 1
            #     kernel_shape = (kernel_size_new, filters_in, self._n_filters)

            #     temp_mask = np.zeros((kernel_shape))
            #     for kernel_idx in range(kernel_size_new):
            #         if kernel_idx % self._dilation_rate == 0:
            #             temp_mask[kernel_idx, :, :] = 1.0

            #     self._conv_kernel_mask = np.copy(temp_mask)
            # else:
            #     kernel_shape = (self._kernel_size, filters_in, self._n_filters)

            kernel_shape = (self._kernel_size, filters_in, self._n_filters)

            self.W = tf.Variable(
                tf.initializers.GlorotNormal()(kernel_shape),
                trainable = True,
                name = "conv1d_kernel",
                dtype = tf.float32
            )

            if self._use_bias:
                bias_shape = (self._n_filters,)
                self.b = tf.Variable(
                    tf.initializers.GlorotNormal()(bias_shape),
                    trainable = True,
                    name = "conv1d_bias",
                    dtype = tf.float32
                )

            self._is_built = True

        # mask weights to simulate dilated convs

        # if self._dilation_rate > 1:
        #     conv_kernel = self.W * tf.constant(self._conv_kernel_mask, tf.float32)
        # else:
        #     conv_kernel = self.W

        conv_kernel = self.W

        if self._padding != 'causal':
            x_out = tf.nn.conv1d(
                    x_in, conv_kernel,
                    self._stride,
                    self._padding.upper(),
                    self._data_format,
                    self._dilation_rate
                )
        else:
            to_pad = self._dilation_rate * (self._kernel_size - 1)
            paddings = (tf.constant([[0, 0], [to_pad, 0], [0, 0]])
                if self._data_format == "NWC" else tf.constant([[0, 0], [0, 0], [to_pad, 0]]))
            x_out = tf.pad(
                x_in, paddings,
                'CONSTANT', 0
            )
            x_out = tf.nn.conv1d(
                    x_out, conv_kernel,
                    self._stride,
                    'VALID',
                    self._data_format,
                    self._dilation_rate
                )

        if self._use_bias:
            x_out = x_out + self.b

        return x_out


class Flatten(tf.Module):
    """
        Flattening layer.
    """

    def __init__(self, name: str = None) -> None:
        """
            Reshape input tensor into (None, -1) shape.

            args:
                name: str - Name of the layer.
            returns:
                None
        """

        super(Flatten, self).__init__(name)

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            args:
                x_in: tf.Tensor - Input tensor of shape (None, dim0, ..., dimN-1)
            returns:
                x_out: tf.Tensor - Output tensor of shape (None, -1)
        """

        x_out = tf.reshape(
            x_in, (tf.shape(x_in)[0], -1), name = 'flatten')

        return x_out


class GlobalMaxPooling1D(tf.Module):
    """
        Global 1D max pooling layer.
    """

    def __init__(self,
                 data_format: str = "channels_last",
                 name: str = None) -> None:
        """
            Calculate channel wise global max pooling.

            args:
                data_format: str - Input dimension order ['channels_last', 'channels_first'].
                name: str - Name of the layer.
            returns:
                None
        """

        super(GlobalMaxPooling1D, self).__init__(name)

        assert data_format in ('channels_first', 'channels_last'), "Invalid data format."
        self._data_format = 'NWC' if data_format == 'channels_last' else 'NCW'

        self._kernel_size = None
        self._n_channels = None

        self._is_built = False

    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            args:
                x_in: tf.Tensor - Input tensor of dimension (None, input_size, in_channels).
            returns:
                x_out: tf.Tensor - Output tensor of dimension (None, 1, in_channels).
        """

        if not self._is_built:
            assert len(x_in.shape) == 3, f"Input tensor rank must be 3, given {len(x_in.shape)}"

            self._kernel_size = x_in.shape[1] if self._data_format == "NWC" else x_in.shape[2]
            self._n_channels = x_in.shape[2] if self._data_format == "NWC" else x_in.shape[1]

            self._is_built = True

        x = tf.nn.max_pool1d(
            x_in, self._kernel_size, 1, 'VALID', self._data_format
        )
        x_out = tf.reshape(x, (tf.shape(x)[0], self._n_channels))

        return x_out


class BatchNormalization(tf.Module):
    """
        Batch normalization layer.
    """

    def __init__(self,
                 axis: int = -1,
                 momentum: float = 0.99,
                 epsilon: float = 0.000001,
                 name: str = None) -> None:
        """
            Apply batch normalization.
            At train time: Calculate batchwise mean and variance and scale input.
            At inference time: Scale using learned population mean and variance.

            args:
                axis: int - Axis across which to calculate batch normalization.
                momentum: float - Momentum for adjusting running mean and variance.
                epsilon: float - Small number for division stability.
                name: Name of the layer.
            returns:
                None
        """

        super(BatchNormalization, self).__init__(name)

        self._axis = axis
        self._momentum = momentum
        self._epsilon = epsilon

        # Do not initialize variables until first call
        self._beta = None
        self._gamma = None

        self._mean_ma = None
        self._var_ma = None

        self._axes = None

        # is built flag for dynamic input size inference
        self._is_built = False

    def _train_path(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Train time behaviour of batch normalization.
                - Calculate batch mean and variance.
                - Scale input tensor.
                - Update stored global mean and variance.

            args:
                x_in: tf.Tensor - Input tensor.
            returns:
                x_out: tf.Tensor - Output tensor.
        """

        # calculate batchwise mean and variance
        mean, var = tf.nn.moments(
            x_in, self._axes
        )

        # scale batch
        x_out = (x_in - mean) / tf.math.sqrt((var + self._epsilon))
        x_out = (self._gamma * x_out) + self._beta

        # update moving mean and variance
        self._mean_ma.assign((self._mean_ma * self._momentum) + (mean * (1 - self._momentum)))
        self._var_ma.assign((self._var_ma * self._momentum) + (var * (1 - self._momentum)))

        return x_out

    def _inference_path(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Inference time behaviour of batch normalization.
                - Scale input tensor using stored global mean and variance.

            args:
                x_in: tf.Tensor - Input tensor.
            returns:
                x_out: tf.Tensor - Output tensor.
        """

        x_out = (x_in - self._mean_ma) / tf.math.sqrt((self._var_ma + self._epsilon))
        x_out = (self._gamma * x_out) + self._beta

        return x_out

    def __call__(self,
                 x_in: tf.Tensor,
                 training: bool = False) -> tf.Tensor:
        """
            args:
                x_in: tf.Tensor - Input tensor of dimension (None, input_size, in_channels).
                training: bool - Flag to toggle train time and inference time behaviour.
            returns:
                x_out: tf.Tensor - Output tensor of dimension (None, input_size, in_channels).  
        """

        if not self._is_built:

            # calculate shape for the normalization axes vector
            if self._axis == -1:
                self._axes = list(range(0, len(x_in.shape) - 1))
            else:
                self._axes = list(range(0, self._axis)) + list(range(self._axis + 1, len(x_in.shape)))

            # initialize variables
            self._beta = tf.Variable(
                initial_value = tf.constant(0, tf.float32, x_in.shape[1:]),
                trainable = True,
                name = "bnorm_beta",
                dtype = tf.float32
            )

            self._gamma = tf.Variable(
                initial_value = tf.constant(1, tf.float32, x_in.shape[1:]),
                trainable = True,
                name = "bn_gamma",
                dtype = tf.float32
            )

            # calculate shape for the ma vectors
            ma_shape = [1] * len(x_in.shape)
            ma_shape[self._axis] = x_in.shape[self._axis]

            self._mean_ma = tf.Variable(
                initial_value = tf.constant(0, tf.float32, ma_shape),
                trainable = False,
                name = "bnorm_mean_ma",
                dtype = tf.float32
            )

            self._var_ma = tf.Variable(
                initial_value = tf.constant(1, tf.float32, ma_shape),
                trainable = False,
                dtype = tf.float32,
                name = "bnorm_var_ma"
            )

            self._is_built = True

        if training:
            x_out = self._train_path(x_in)
        else:
            x_out = self._inference_path(x_in)

        return x_out


class Dropout(tf.Module):
    """
        Dropout layer.
    """

    def __init__(self,
                 dropout_rate: float,
                 random_state: int = None,
                 name: str = None) -> None:
        """
            Randomly dropout layers at train time.

            args:
                dropout_rate: float - Ratio of neurons to drop out.
                random_state: int - Operational random seed.
                name: str - Name of the layer.
            returns:
                None
        """
        super(Dropout, self).__init__(name)

        self._dropout_rate = dropout_rate
        self._random_state = random_state

    def __call__(self,
                 x_in: tf.Tensor,
                 training: bool = False) -> tf.Tensor:
        """
            args:
                x_in: tf.Tensor - Input tensor.
                training: bool - Flag to toggle train time and inference time behaviour.
            returns:
                x_out: tf.Tensor - Output tensor.
        """

        if training:
            x_out = tf.nn.dropout(
                x_in, self._dropout_rate,
                seed = self._random_state,
            )
        else:
            x_out = tf.identity(x_in)

        return x_out
