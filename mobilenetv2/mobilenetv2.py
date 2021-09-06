from math import floor
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers


class MobilenetV2():
    """
        MobilenetV2 class
    """

    def __init__(self, output_size, input_shape = None, alpha = 1.0) -> None:

        self.output_size: int = output_size
        self.input_shape: Optional[Tuple[int, int, int]] = (input_shape if
            input_shape is not None else (224,224,3))
        self.alpha = alpha

    def _depthwise_block(self,
                         x_in: layers.Layer,
                         num_filters_out: int,
                         expansion: int,
                         stride: int) -> layers.Layer:
        """"
            Construct inverted residual block with identity skip
        """

        num_filters_expanded = int(floor(x_in.shape[3] * expansion * self.alpha))
        num_filters_out = int(floor(num_filters_out * self.alpha))

        x = layers.Conv2D(
            num_filters_expanded, (1,1), (1, 1), padding='same')(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.DepthwiseConv2D(
            (3,3), (stride, stride), padding = 'same',
            depth_multiplier = 1, data_format='channels_last')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(
            num_filters_out, (1,1), (1,1), padding  = 'same')(x)

        if x_in.shape[3] == num_filters_out:
            identity = x_in
        else:
            identity = layers.Conv2D(num_filters_out, (1,1), (1,1), 'same')(x_in)

        if stride == 2:
            x_out = x
        else:
            x_out = layers.Add()([x, identity])
        return x_out

    def __call__(self) -> Union[layers.Layer, layers.Layer]:
        """
            Model constructor
        """

        x_in = layers.Input(shape = self.input_shape)
        #
        x = layers.Conv2D(
            int(floor(32 * self.alpha)), (3,3), (2,2), padding = 'same')(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        #
        x = self._depthwise_block(
            x, int(floor(16 * self.alpha)), 1, 1)
        #
        x = self._depthwise_block(
            x, int(floor(24 * self.alpha)), 6, 2)
        x = self._depthwise_block(
            x, int(floor(24 * self.alpha)), 6, 1)
        #
        x = self._depthwise_block(
            x, int(floor(32 * self.alpha)), 6, 2)
        x = self._depthwise_block(
            x, int(floor(32 * self.alpha)), 6, 1)
        x = self._depthwise_block(
            x, int(floor(32 * self.alpha)), 6, 1)
        #
        x = self._depthwise_block(
            x, int(floor(64 * self.alpha)), 6, 2)
        x = self._depthwise_block(
            x, int(floor(64 * self.alpha)), 6, 1)
        x = self._depthwise_block(
            x, int(floor(64 * self.alpha)), 6, 1)
        x = self._depthwise_block(
            x, int(floor(64 * self.alpha)), 6, 1)
        #
        x = self._depthwise_block(
            x, int(floor(96 * self.alpha)), 6, 1)
        x = self._depthwise_block(
            x, int(floor(96 * self.alpha)), 6, 1)
        x = self._depthwise_block(
            x, int(floor(96 * self.alpha)), 6, 1)
        #
        x = self._depthwise_block(
            x, int(floor(160 * self.alpha)), 6, 2)
        x = self._depthwise_block(
            x, int(floor(160 * self.alpha)), 6, 1)
        x = self._depthwise_block(
            x, int(floor(160 * self.alpha)), 6, 1)
        #
        x = self._depthwise_block(
            x, int(floor(320 * self.alpha)), 6, 1)
        #
        x = layers.Conv2D(
            int(floor(1280 * self.alpha)), (1,1), (1,1), padding = 'same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling2D()(x)
        #
        x = layers.Dense(int(floor(1280 * self.alpha)))(x)
        x = layers.ReLU()(x)
        x = layers.Dense(self.output_size)(x)
        x_out = layers.Softmax()(x)
        return x_in, x_out
