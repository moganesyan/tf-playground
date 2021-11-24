from math import floor
from typing import Optional, Tuple, Union, List, Dict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import ReLU

from src.utils.dl_utils import ReLU6, HSwish, HSigm


MOBILENETV3_LARGE = [
    {'type': 'conv2d', 'k': 3, 'exp': None, 'nout': 16, 'se': False, 'nl': HSwish, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 3, 'exp': 16, 'nout': 16, 'se': False, 'nl': ReLU, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 64, 'nout': 24, 'se': False, 'nl': ReLU, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 3, 'exp': 72, 'nout': 24, 'se': False, 'nl': ReLU, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 72, 'nout': 40, 'se': True, 'nl': ReLU, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 5, 'exp': 120, 'nout': 40, 'se': True, 'nl': ReLU, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 120, 'nout': 40, 'se': True, 'nl': ReLU, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 240, 'nout': 80, 'se': False, 'nl': HSwish, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 3, 'exp': 200, 'nout': 80, 'se': False, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 184, 'nout': 80, 'se': False, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 184, 'nout': 80, 'se': False, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 480, 'nout': 112, 'se': True, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 672, 'nout': 160, 'se': True, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 672, 'nout': 160, 'se': True, 'nl': HSwish, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 5, 'exp': 960, 'nout': 160, 'se': True, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 960, 'nout': 160, 'se': True, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'conv2d', 'k': 1, 'exp': None, 'nout': 960, 'se': False, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'pool', 'k': 7, 'exp': None, 'nout': None, 'se': False, 'nl': None, 'bn': None, 's': 1}, #
    {'type': 'conv2d', 'k': 1, 'exp': None, 'nout': 1280, 'se': False, 'nl': HSwish, 'bn': False, 's': 1}, #
]

MOBILENETV3_SMALL = [
    {'type': 'conv2d', 'k': 3, 'exp': None, 'nout': 16, 'se': False, 'nl': HSwish, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 3, 'exp': 16, 'nout': 16, 'se': True, 'nl': ReLU, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 3, 'exp': 72, 'nout': 24, 'se': False, 'nl': ReLU, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 3, 'exp': 88, 'nout': 24, 'se': False, 'nl': ReLU, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 96, 'nout': 40, 'se': True, 'nl': HSwish, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 5, 'exp': 240, 'nout': 40, 'se': True, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 240, 'nout': 40, 'se': True, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 120, 'nout': 48, 'se': True, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 144, 'nout': 48, 'se': True, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 288, 'nout': 96, 'se': True, 'nl': HSwish, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 5, 'exp': 576, 'nout': 96, 'se': True, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 576, 'nout': 96, 'se': True, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'conv2d', 'k': 1, 'exp': None, 'nout': 576, 'se': False, 'nl': HSwish, 'bn': True, 's': 1}, #
    {'type': 'pool', 'k': 7, 'exp': None, 'nout': None, 'se': False, 'nl': None, 'bn': None, 's': 1}, #
    {'type': 'conv2d', 'k': 1, 'exp': None, 'nout': 1024, 'se': False, 'nl': HSwish, 'bn': False, 's': 1}, #
]


class MobileNetV3():
    """
        Mobilenet V3 Base class
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 output_size: int,
                 alpha: float = 1.0,
                 size: str = 'large') -> None:

        self._input_shape: Tuple[int, int, int] = input_shape
        self._output_size: int = output_size
        self._alpha: float = alpha
        #
        self._se_factor: int = 4
        self._se_type: str = 'conv'
        #
        assert size in ['small', 'large'], "use 'small' or 'large'"
        self._size: str = size
        self._architecture: List[Dict]  = MOBILENETV3_LARGE if size == 'large' else MOBILENETV3_SMALL

    def _se_block(self, x_in: layers.Layer) -> layers.Layer:
        """
            Squeeze and excitation block
        """

        # FC variant of the SE block
        if self._se_type == 'fc':
            x = x_in
            x = layers.GlobalAveragePooling2D()(x)
            dim_x = x.shape[1]
            dim_x_scaled = max(int(floor(dim_x / self._se_factor)),4)
            x = layers.Dense(dim_x_scaled)(x)
            x = ReLU()(x)
            x = layers.Dense(dim_x)(x)
            x = HSigm()(x)
            x_out = layers.Multiply()([x_in, x])
            return x_out
        # Conv variant of the SE block
        elif self._se_type == 'conv':
            x = x_in
            x = layers.GlobalAveragePooling2D()(x)
            dim_x = x.shape[1]
            dim_x_scaled = max(int(floor(dim_x / self._se_factor)),4)
            x = layers.Reshape((1,1,dim_x))(x)
            x = layers.Conv2D(dim_x_scaled, 1,1,'same')(x)
            x = ReLU()(x)
            x = layers.Conv2D(dim_x, 1,1,'same')(x)
            x = HSigm()(x)
            x_out = layers.Multiply()([x_in, x])
            return x_out

    def _inverted_residual_block_v3(self, x_in: layers.Layer, k: int,
                                    exp: int, nout: int, se: bool,
                                    nl: layers.Layer, bn: bool, s: int) -> layers.Layer:
        """
            Mobilenet V3 Inverted Residual Block with SE component
        """

        x = x_in

        # expansion. Do not apply if the expansion factor == 1
        if exp > x_in.shape[3]:
            x = layers.Conv2D(exp, 1, 1, 'same')(x_in)
            if bn:
                x = layers.BatchNormalization()(x)
            x = nl()(x)
        # depthwise
        x = layers.DepthwiseConv2D(k,s,'same',1,data_format='channels_last')(x)
        if bn:
            x = layers.BatchNormalization()(x)
        x = nl()(x)
        # optional SE block
        if se:
            x = self._se_block(x)
        # projection
        x = layers.Conv2D(nout,1,1,'same')(x)
        if bn:
            x = layers.BatchNormalization()(x)
        # residual connection and output
        if s == 2 or x_in.shape[3] != nout:
            x_out = x
        else:
            identity = x_in
            x_out = layers.Add()([x, identity])
        return x_out

    def _conv2d_block(self, x_in: layers.Layer, k: int,
                      exp: int, nout: int, se: bool,
                      nl: layers.Layer, bn: bool, s: int) -> layers.Layer:
        """
            Conv2D block constructor
        """

        x = layers.Conv2D(nout, k, s, 'same')(x_in)
        if bn:
            x = layers.BatchNormalization()(x)
        x_out = nl()(x)
        return x_out

    def _pool_block(self, x_in: layers.Layer, k: int,
                    exp: int, nout: int, se: bool,
                    nl: layers.Layer, bn: bool, s: int) -> layers.Layer:
        """
            Pool block constructor
        """

        x = layers.GlobalAveragePooling2D()(x_in)
        x_out = layers.Reshape((1,1,x_in.shape[3]))(x)
        return x_out

    def _build_network(self) -> Union[layers.Layer, layers.Layer]:
        """
            Build network from architecture
        """

        x_in = layers.Input(self._input_shape)

        x = x_in
        for _, layer_dict in enumerate(self._architecture):
            n_out_scaled = (int(floor(layer_dict['nout'] * self._alpha))
                if layer_dict['nout'] is not None else None)
            exp_scaled = (int(floor(layer_dict['exp'] * self._alpha))
                if layer_dict['exp'] is not None else None)

            if layer_dict['type'] == 'conv2d':
                x = self._conv2d_block(
                    x, layer_dict['k'], exp_scaled, n_out_scaled,
                    layer_dict['se'], layer_dict['nl'], layer_dict['bn'],
                    layer_dict['s'])
            elif layer_dict['type'] == 'bneck':
                x = self._inverted_residual_block_v3(
                    x, layer_dict['k'], exp_scaled, n_out_scaled,
                    layer_dict['se'], layer_dict['nl'], layer_dict['bn'],
                    layer_dict['s'])
            elif layer_dict['type'] == 'pool':
                x = self._pool_block(
                    x, layer_dict['k'], exp_scaled, n_out_scaled,
                    layer_dict['se'], layer_dict['nl'], layer_dict['bn'],
                    layer_dict['s'])

        x = layers.Dropout(0.20)(x)
        x = layers.Conv2D(self._output_size,1,1,'valid')(x)
        x = layers.Flatten()(x)
        x_out = layers.Softmax()(x)
        return x_in, x_out

    def __call__(self) -> models.Model:
        """
            Build Keras Model
        """

        x_in, x_out = self._build_network()
        model = models.Model(inputs = x_in, outputs = x_out)
        return model
