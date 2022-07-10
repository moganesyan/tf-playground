import imp
import tensorflow as tf
from layers import Conv1D
from layers import BatchNormalization
from layers import Dropout

from activations import ReLU


class ResidualTCNBlock(tf.Module):
    """
        Residual TCN Block
    """
    def __init__(self,
                 out_filters: int,
                 kernel_size: int,
                 dilation_rate: int,
                 dropout_rate: float,
                 name: str = None) -> None:
        """
            Residual TCN block, consisting of:
                - Dilated Causal 1D Conv
                - Batch Normalization
                - Activation
                - Dropout
                - Dilated Causal 1D Conv
                - Batch Normalization
                - Activation

            args:
                out_filters: int - Number of convolutional filters.
                kernel_size: int - Size of the convolutional kernel.
                dilation_rate: int - Dilation rate for the convolution.
                dropout_rate: float - Dropout rate.
            returns:
                None
        """

        super(ResidualTCNBlock, self).__init__(name)

        # define layer settings.
        self._out_filters = out_filters
        self._kernel_size = kernel_size
        self._dilation_rate = dilation_rate
        self._dropout_rate = dropout_rate

        # pre-initialise the layers.
        self._cconv_1 = Conv1D(
            out_filters, kernel_size, 1,
            dilation_rate, 'causal',
            use_bias  = False, name = 'causal_conv_1'
        )
        self._bnorm_1 = BatchNormalization(name = 'bnorm_1')
        self._relu_1 = ReLU(name = 'relu_1')
        self._dropout_1 = Dropout(dropout_rate, name= 'dropout_1')
        #
        self._cconv_2 = Conv1D(
            out_filters, kernel_size, 1,
            dilation_rate, 'causal',
            use_bias  = False, name = 'causal_conv_2'
        )
        self._bnorm_2 = BatchNormalization(name = 'bnorm_2')
        self._relu_2 = ReLU(name = 'relu_2')
        self._dropout_2 = Dropout(dropout_rate, name= 'dropout_1')

        # is built flag for dynamic input size inference
        self._is_built = False

    def __call__(self,
                 x_in: tf.Tensor,
                 training: bool = False) -> tf.Tensor:
        """
            args:
                x_in: tf.Tensor - Input tensor of dimension (None, input_size, in_filters)
                training: bool - Flag to toggle train time and inference time behaviour.
            returns:
                x_out: tf.Tensor - Output tensor of dimension (None, output_size, out_filters)
        """

        if not self._is_built:
            if x_in.shape[-1] != self._out_filters:
                self._identity = Conv1D(
                    self._out_filters, 1,
                    1, 1, 'same',
                    use_bias = False,
                    name = 'projection_conv'
                )
            else:
                self._identity = tf.identity

            self._is_built = True

        x_residual = self._identity(x_in)

        x = self._cconv_1(x_in)
        x = self._bnorm_1(x, training = training)
        x = self._relu_1(x)
        x = self._dropout_1(x, training = training)

        x = self._cconv_2(x)
        x = self._bnorm_2(x, training = training)
        x = self._relu_2(x)
        x = self._dropout_2(x, training = training)

        x_out = tf.add(
            x, x_residual, name = 'residual_tcn_block'
        )

        return x_out
