from math import floor
from typing import Optional, Tuple, Union, List, Dict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from src.utils.dl_utils import HSwish, HSigm, StochasticDropout


PARAMS_EFNT_B0: List[Dict] = [
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

        # stochastic depth parameters
        self._sd_survival_prob: float = 0.80

    def _block_se(self, x_in: layers.Layer) -> layers.Layer:
        """
            Squeeze and excitation block.
            Can be convolutional or fully connected.
        """

        x = x_in
        dim = x_in.shape[3]
        dim_scaled = max(int(floor(dim / self._se_downsample)), 8)
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
            for layer_idx in range(layer_dict["nlayers"]):

                # only apply stride on the 1st layer of the block
                stride_active = layer_dict["s"] if layer_idx == 0 else 1

                if layer_dict["kind"] == "conv2d":
                    x = self._block_conv2d(x, layer_dict["nout"], layer_dict["k"],
                    stride_active, layer_dict["nl"], layer_dict["bn"])
                elif layer_dict["kind"] == "mbconv":
                    x = self._block_mbconv(
                        x, layer_dict["nout"], layer_dict["exp"], layer_dict["k"],
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


def EfficientNetB0(input_shape: Tuple[int, int, int], output_size: int):
    return EfficientNetV1(input_shape, output_size, PARAMS_EFNT_B0)
