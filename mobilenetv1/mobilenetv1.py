from math import floor
from typing import Tuple, Union, Optional

import tensorflow as tf
from tensorflow.keras import layers


class ReLU6(layers.Layer):
    """
        Custom ReLU6 activation
    """

    def __init__(self) -> None:
        super(ReLU6, self).__init__()

    def call(self, input) -> layers.Layer:
        return tf.clip_by_value(
            input,
            clip_value_min = 0.0,
            clip_value_max = 6.0)
            

class MobilenetV1:
    """
        Mobilenet V1 class
    """

    def __init__(self,
                 output_size: int,
                 input_shape: Optional[Tuple[int, int, int]] = None,
                 alpha: float = 1.0) -> None:

        self.input_shape: Optional[Tuple[int, int, int]] = input_shape
        if input_shape is None:
            self.input_shape = (224, 224, 3)

        self.output_size: int = output_size
        self.alpha: float = alpha

        self.dwise_conv_kernel: Tuple[int, int] = (3, 3)
        self.dwise_conv_multiplier: int = 1
        self.fwise_conv_kernel: Tuple[int, int] = (1,1)

    def _depthwise_block(self,
                         x_in: layers.Layer,
                         num_filters: int,
                         stride: int) -> layers.Layer:
        """
            Make depthwise separable convolution block using a combination
                of depthwise and 1x1 full convolutions
        """

        num_filters_scaled = int(floor(num_filters * self.alpha))
        x = layers.DepthwiseConv2D(
            self.dwise_conv_kernel,
            strides = (stride, stride),
            padding = 'same',
            depth_multiplier = self.dwise_conv_multiplier,
            data_format = 'channels_last')(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(
            num_filters_scaled,
            self.fwise_conv_kernel,
            strides = (1, 1),
            padding = 'same')(x)
        x = layers.BatchNormalization()(x)
        x = ReLU6()(x)
        return x

    def __call__(self) -> Union[layers.Layer, layers.Layer]:
        """
            Construct graph for MobilenetV1 using functional API
        """

        x_in = layers.Input(self.input_shape)
        #
        x = layers.Conv2D(
            int(floor(32 * self.alpha)),
            (3,3), (2,2), 'same')(x_in)
        x = layers.BatchNormalization()(x)
        x = ReLU6()(x)
        #
        x = self._depthwise_block(
            x, 64, 1)
        x = self._depthwise_block(
            x, 128, 2)
        x = self._depthwise_block(
            x, 128, 1)
        x = self._depthwise_block(
            x, 256, 2)
        x = self._depthwise_block(
            x, 256, 1)
        x = self._depthwise_block(
            x, 512, 2)
        #
        x = self._depthwise_block(
            x, 512, 1)
        x = self._depthwise_block(
            x, 512, 1)
        x = self._depthwise_block(
            x, 512, 1)
        x = self._depthwise_block(
            x, 512, 1)
        x = self._depthwise_block(
            x, 512, 1)
        #
        x = self._depthwise_block(
            x, 1024, 2)
        x = self._depthwise_block(
            x, 1024, 1)
        x = layers.GlobalAveragePooling2D()(x)
        #
        x = layers.Dense(int(floor(1024 * self.alpha)))(x)
        x = ReLU6()(x)
        x = layers.Dense(self.output_size)(x)
        x = layers.Softmax()(x)

        return x_in, x
