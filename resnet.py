from typing import List, Tuple

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


class ResNet():
    """
     ResNet Base class
    """

    def __init__(self,
                 output_size: int,
                 architecture: List[int] = [3, 4, 6, 3]) -> None:

        self.output_size: int = output_size
        self.arch: List[int] = architecture
        self.nfilters: int = 64

    def _make_stack(self,
                    x_in: layers.Layer,
                    nfilters_in: int,
                    nfilters_base: int,
                    stack_size: int,
                    stride: int):
        """
            Construct residual block stack
        """
        x = residual_block(
            x_in, nfilters_in, nfilters_base, stride = stride)

        nfilters_in = self.nfilters * 4

        for _ in range(stack_size - 1):
            x = residual_block(
                x, nfilters_in, nfilters_base, stride = 1)
        return x
            
    def __call__(self, x_in: layers.Layer) -> None:
        """
            Set up functional graph for ResNet
        """

        # conv1
        x = layers.Conv2D(self.nfilters, (7,7), (2,2), padding='same')(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D((3,3), (2,2), 'same')(x)

        # resnet stack 1 (conv2_x)
        x = self._make_stack(
            x, 64, self.nfilters,
            self.arch[0],1)
        # resnet stack 2 (conv3_x)
        self.nfilters = self.nfilters * 2
        x = self._make_stack(
            x, 256, self.nfilters,
            self.arch[1],2)
        # resnet stack 3 (conv4_x)
        self.nfilters = self.nfilters * 2
        x = self._make_stack(
            x, 512, self.nfilters,
            self.arch[2],2)
        # resnet stack 4 (conv5_x)
        self.nfilters = self.nfilters * 2
        x = self._make_stack(
            x, 1024, self.nfilters,
            self.arch[3],2)

        # fc1 layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1000)(x)
        x = layers.ReLU()(x)

        # output layer
        x = layers.Dense(self.output_size)(x)
        x_out = layers.Softmax()(x)

        return x_out


def residual_block(x_in: layers.Layer,
                   nfilters_in: int,
                   nfilters_base: int,
                   stride: int = 1,
                   upsample_factor: int = 4):
    """
        Residual block generator via functional API
    """
    identity = x_in

    x = layers.Conv2D(nfilters_base,(1,1),(stride, stride),'valid')(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(nfilters_base,(3,3),(1,1),'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(nfilters_base * upsample_factor,(1,1),(1,1),'valid')(x)
    x = layers.BatchNormalization()(x)

    #identity skip connection
    if nfilters_base * 4 != nfilters_in or stride != 1:
        identity = layers.Conv2D(
            nfilters_base * upsample_factor,
            (1,1), (stride,stride))(identity)
        identity = layers.BatchNormalization()(identity)
        print(f'fancy skip. nfilters_in: {nfilters_in}, nfilters_base: {nfilters_base}, stride: {stride}')
    else:
        print(f'normal skip. nfilters_in: {nfilters_in}, nfilters_base: {nfilters_base}, stride: {stride}')

    x = layers.Add()([x, identity])
    x = layers.ReLU()(x)
    return x


resnet = ResNet(10)
layer_in = layers.Input((250,250,3))
layer_out = resnet(layer_in)

model = tf.keras.Model(inputs = layer_in, outputs = layer_out, name = 'residual block')
# model.summary()
# tf.keras.utils.plot_model(
#     model, to_file='model.png', show_shapes=True, show_dtype=True,
#     show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96
# )

model.save('resnet50.h5')