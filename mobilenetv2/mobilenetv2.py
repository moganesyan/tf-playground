from math import floor
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from utils.dl_utils import ReLU6


class MobileNetV2():
    """
        MobileNet V2 class
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 output_size: int,
                 alpha: float = 1.0) -> None:
        """
            Mobilenet V2 constructor
            input_shape: input image shape
            output_size: number of classes
            alpha: Width scaling coefficient
        """

        self.output_size: int = output_size
        self.input_shape: Tuple[int, int, int] = input_shape
        self.alpha: float = alpha

        self.dwise_conv_kernel: Tuple[int, int] = (3, 3)
        self.dwise_conv_multiplier: int = 1
        self.fwise_conv_kernel: Tuple[int, int] = (1,1)

    def _inverted_residual_block(self,
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
            num_filters_expanded, self.fwise_conv_kernel, (1, 1), padding='same')(x_in)
        x = layers.BatchNormalization()(x)
        x = ReLU6()(x)
        x = layers.DepthwiseConv2D(
            self.dwise_conv_kernel, (stride, stride), padding = 'same',
            depth_multiplier = self.dwise_conv_multiplier,
            data_format='channels_last')(x)
        x = layers.BatchNormalization()(x)
        x = ReLU6()(x)
        x = layers.Conv2D(
            num_filters_out, self.fwise_conv_kernel, (1,1), padding  = 'same')(x)
        x  = layers.BatchNormalization()(x)

        if stride == 2 or x_in.shape[3] != num_filters_out:
            x_out = x
        else:
            identity = x_in
            x_out = layers.Add()([x, identity])
        return x_out

    def _build_network(self) -> Union[layers.Layer, layers.Layer]:
        """
            Model constructor
        """

        x_in = layers.Input(shape = self.input_shape)
        #
        x = layers.Conv2D(
            int(floor(32 * self.alpha)), (3,3), (2,2), padding = 'same')(x_in)
        x = layers.BatchNormalization()(x)
        x = ReLU6()(x)
        #
        x = self._inverted_residual_block(
            x, int(floor(16 * self.alpha)), 1, 1)
        #
        x = self._inverted_residual_block(
            x, int(floor(24 * self.alpha)), 6, 2)
        x = self._inverted_residual_block(
            x, int(floor(24 * self.alpha)), 6, 1)
        #
        x = self._inverted_residual_block(
            x, int(floor(32 * self.alpha)), 6, 2)
        x = self._inverted_residual_block(
            x, int(floor(32 * self.alpha)), 6, 1)
        x = self._inverted_residual_block(
            x, int(floor(32 * self.alpha)), 6, 1)
        #
        x = self._inverted_residual_block(
            x, int(floor(64 * self.alpha)), 6, 2)
        x = self._inverted_residual_block(
            x, int(floor(64 * self.alpha)), 6, 1)
        x = self._inverted_residual_block(
            x, int(floor(64 * self.alpha)), 6, 1)
        x = self._inverted_residual_block(
            x, int(floor(64 * self.alpha)), 6, 1)
        #
        x = self._inverted_residual_block(
            x, int(floor(96 * self.alpha)), 6, 1)
        x = self._inverted_residual_block(
            x, int(floor(96 * self.alpha)), 6, 1)
        x = self._inverted_residual_block(
            x, int(floor(96 * self.alpha)), 6, 1)
        #
        x = self._inverted_residual_block(
            x, int(floor(160 * self.alpha)), 6, 2)
        x = self._inverted_residual_block(
            x, int(floor(160 * self.alpha)), 6, 1)
        x = self._inverted_residual_block(
            x, int(floor(160 * self.alpha)), 6, 1)
        #
        x = self._inverted_residual_block(
            x, int(floor(320 * self.alpha)), 6, 1)
        #
        x = layers.Conv2D(
            int(floor(1280 * self.alpha)), (1,1), (1,1), padding = 'same')(x)
        x = layers.BatchNormalization()(x)
        x = ReLU6()(x)
        #
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1,1,x.shape[1]))(x)
        #
        x = layers.Dropout(0.20)(x)
        x = layers.Conv2D(self.output_size, 1, 1, 'valid')(x)
        x = layers.Flatten()(x)
        x_out = layers.Softmax()(x)

        return x_in, x_out

    def __call__(self) -> Union[layers.Layer, layers.Layer]:
        """
            Build Keras model
        """

        x_in, x_out = self._build_network()
        model = models.Model(inputs = x_in, outputs = x_out)

        return model
