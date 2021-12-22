from math import floor
from typing import Optional, Tuple, Union, List, Dict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from src.utils.dl_utils import HSwish, HSigm, StochasticDropout


BASE_ARCHITECTURE: List[Dict] = [
    {'kind': 'conv2d' , 'nlayers': 1, 'k': 3, 'exp': None, 'nout': 32, 'se': None, 'nl': HSwish, 'bn': True, 's': 2},
    {'kind': 'mbconv' , 'nlayers': 1, 'k': 3, 'exp': 1, 'nout': 16, 'se': True, 'nl': HSwish, 'bn': True, 's': 1},
    {'kind': 'mbconv' , 'nlayers': 2, 'k': 3, 'exp': 6, 'nout': 24, 'se': True, 'nl': HSwish, 'bn': True, 's': 2},
    {'kind': 'mbconv' , 'nlayers': 2, 'k': 5, 'exp': 6, 'nout': 40, 'se': True, 'nl': HSwish, 'bn': True, 's': 2},
    {'kind': 'mbconv' , 'nlayers': 3, 'k': 3, 'exp': 6, 'nout': 80, 'se': True, 'nl': HSwish, 'bn': True, 's': 1},
    {'kind': 'mbconv' , 'nlayers': 3, 'k': 5, 'exp': 6, 'nout': 112, 'se': True, 'nl': HSwish, 'bn': True, 's': 2},
    {'kind': 'mbconv' , 'nlayers': 4, 'k': 5, 'exp': 6, 'nout': 192, 'se': True, 'nl': HSwish, 'bn': True, 's': 2},
    {'kind': 'mbconv' , 'nlayers': 1, 'k': 3, 'exp': 6, 'nout': 320, 'se': True, 'nl': HSwish, 'bn': True, 's': 1},
    {'kind': 'conv2d' , 'nlayers': 1, 'k': 1, 'exp': None, 'nout': 1280, 'se': None, 'nl': HSwish, 'bn': True, 's': 1},
    {'kind': 'pool' , 'nlayers': 1, 'k': None, 'exp': None, 'nout': None, 'se': None, 'nl': None, 'bn': None, 's': None},
]


class EfficientNetV1:
    """
        Base Class for the EfficientNet V1 architecture.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 output_size: int,
                 width_factor: float,
                 depth_factor: float,
                 dropout_rate: float,
                 architecture: List[Dict]) -> None:
        """
            Class Constructor.
        """

        # base network parameters
        self._input_shape: Tuple[int, int, int] = input_shape
        self._output_size: int = output_size
        self._architecture: List[Dict] = architecture

        # squeeze and excitation (SE) parameters
        self._se_kind: int = "conv"
        self._se_downsample: int = 24
        self._se_min: int = 8

        # scaling parameters
        self._width_factor: float = width_factor
        self._depth_factor: float = depth_factor
        self._sd_survival_prob: float = 1 - dropout_rate

    def _block_se(self, x_in: layers.Layer) -> layers.Layer:
        """
            Squeeze and excitation block.
            Can be convolutional or fully connected.
        """

        x = x_in
        dim = x_in.shape[3]
        dim_scaled = max(int(floor(dim / self._se_downsample)), self._se_min)
        x = layers.GlobalAveragePooling2D()(x)
        if self._se_kind == "fc":
            x = layers.Dense(dim_scaled)(x)
            x = HSwish()(x)
            x = layers.Dense(dim)(x)
            x = HSigm()(x)
        elif self._se_kind == "conv":
            x = layers.Reshape((1,1,dim))(x)
            x = layers.Conv2D(
                dim_scaled, 1, 1, 'same')(x)
            x = HSwish()(x)
            x = layers.Conv2D(
                dim, 1, 1, 'same')(x)
            x = HSigm()(x)

        x_out = layers.Multiply()([x, x_in])
        return x_out

    def _block_conv2d(self,
                      x_in: layers.Layer,
                      nout: int,
                      k: int,
                      s: int,
                      nl: layers.Layer,
                      bn: bool ) -> layers.Layer:
        """
            2D Convolution Block.
        """

        x = layers.Conv2D(nout, k, s, 'same', use_bias=False)(x_in)
        if bn:
            x = layers.BatchNormalization()(x)
        x_out = nl()(x)
        return x_out

    def _block_mbconv(self,
                      x_in: layers.Layer,
                      nout: int,
                      exp: int,
                      k: int,
                      s: int,
                      nl: layers.Layer,
                      bn: bool,
                      se: bool) -> layers.Layer:
        """
            Inverted Residual Block with SE. (MBCONV)
        """

        nexp = x_in.shape[3] * exp
        x = x_in

        # expansion. Do not apply if expansion factor is 1 due to redundancy.
        if exp > 1:
            x = layers.Conv2D(
                nexp, 1, 1, 'same', use_bias=False)(x)
            if bn:
                x = layers.BatchNormalization()(x)
            x = nl()(x)

        # depthwise conv
        x = layers.DepthwiseConv2D(
            k, s, 'same', data_format="channels_last", use_bias=False)(x)
        if bn:
            x = layers.BatchNormalization()(x)
        x = nl()(x)

        # apply SE if needed
        if se:
            x = self._block_se(x)

        # projection
        x = layers.Conv2D(
            nout, 1, 1, 'same', use_bias=False)(x)
        if bn:
            x = layers.BatchNormalization()(x)

        # residual connection and stochastic dropout
        if s == 2 or x_in.shape[3] != nout:
            x_out = x
        else:
            x = StochasticDropout(self._sd_survival_prob)(x)
            # x = layers.Dropout(
            #     self._sd_survival_prob, noise_shape=(-1,1,1,1))(x)
            identity = x_in
            x_out = layers.Add()([identity, x])
        return x_out

    def _block_pool(self,
                    x_in: layers.Layer) -> layers.Layer:
        """
            Pooling block.
        """

        x = layers.GlobalAveragePooling2D()(x_in)
        x_out = layers.Reshape((1, 1, x_in.shape[3]))(x)
        return x_out

    def _build_network(self) -> Union[layers.Layer, layers.Layer]:
        """
            Build EfficientNet V1 network according to the architecture params.
        """

        # input block
        x_in = layers.Input(self._input_shape)
        x = x_in

        # architecture constructor
        for _, layer_dict in enumerate(self._architecture):
            nlayers_scaled = int(floor( layer_dict["nlayers"] * self._depth_factor))
            for layer_idx in range(nlayers_scaled):
                # only apply stride on the 1st layer of the block
                stride_active = layer_dict["s"] if layer_idx == 0 else 1

                if layer_dict["kind"] == "conv2d":
                    nout_scaled = int(floor(layer_dict["nout"] * self._width_factor))
                    x = self._block_conv2d(x, nout_scaled, layer_dict["k"],
                    stride_active, layer_dict["nl"], layer_dict["bn"])
                elif layer_dict["kind"] == "mbconv":
                    nout_scaled = int(floor(layer_dict["nout"] * self._width_factor))
                    x = self._block_mbconv(
                        x, nout_scaled, layer_dict["exp"], layer_dict["k"],
                        stride_active, layer_dict["nl"], layer_dict["bn"], layer_dict["se"])
                elif layer_dict["kind"] == "pool":
                    x = self._block_pool(x)
                else:
                    raise Exception(f"Invalid layer type: {layer_dict['kind']}")

        # output block
        x = layers.Conv2D(
            self._output_size, 1, 1, 'valid')(x)
        x = layers.Flatten()(x)
        x_out = layers.Softmax()(x)

        return x_in, x_out

    def __call__(self) -> models.Model:
        x_in, x_out = self._build_network()
        model = models.Model(inputs = x_in, outputs = x_out)
        return model


def EfficientNetB0(output_size: int):
    return EfficientNetV1(
        (224, 224, 3), output_size, 1.0,
        1.0, 0.2, BASE_ARCHITECTURE)


def EfficientNetB1(output_size: int):
    return EfficientNetV1(
        (240, 240, 3), output_size, 1.0,
        1.1, 0.2, BASE_ARCHITECTURE)


def EfficientNetB2(output_size: int):
    return EfficientNetV1(
        (260, 260, 3), output_size, 1.1,
        1.2, 0.3, BASE_ARCHITECTURE)


def EfficientNetB3(output_size: int):
    return EfficientNetV1(
        (300, 300, 3), output_size, 1.2,
        1.4, 0.3, BASE_ARCHITECTURE)


def EfficientNetB4(output_size: int):
    return EfficientNetV1(
        (380, 380, 3), output_size, 1.4,
        1.8, 0.4, BASE_ARCHITECTURE)


def EfficientNetB5(output_size: int):
    return EfficientNetV1(
        (456, 456, 3), output_size, 1.6,
        2.2, 0.4, BASE_ARCHITECTURE)


def EfficientNetB6(output_size: int):
    return EfficientNetV1(
        (528, 528, 3), output_size, 1.8,
        2.6, 0.5, BASE_ARCHITECTURE)


def EfficientNetB7(output_size: int):
    return EfficientNetV1(
        (600, 600, 3), output_size, 2.0,
        3.1, 0.5, BASE_ARCHITECTURE)
