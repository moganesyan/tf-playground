from math import floor
from typing import Optional, Tuple, Union, List, Dict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

# from utils.dl_utils import ReLU6


import tensorflow as tf
from tensorflow.keras import layers


class ReLU6(layers.Layer):
    """
        Custom ReLU6 activation layer
    """

    def __init__(self) -> None:
        super(ReLU6, self).__init__()

    def call(self, input) -> layers.Layer:
        return tf.clip_by_value(
            input,
            clip_value_min = 0.0,
            clip_value_max = 6.0)
            


MOBILENETV3_LARGE = [
    {'type': 'conv2d', 'k': 3, 'exp': None, 'nout': 16, 'se': False, 'nl': ReLU6, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 3, 'exp': 16, 'nout': 16, 'se': False, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 64, 'nout': 24, 'se': False, 'nl': ReLU6, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 3, 'exp': 72, 'nout': 24, 'se': False, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 72, 'nout': 40, 'se': True, 'nl': ReLU6, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 5, 'exp': 120, 'nout': 40, 'se': True, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 120, 'nout': 40, 'se': True, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 240, 'nout': 80, 'se': False, 'nl': ReLU6, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 3, 'exp': 200, 'nout': 80, 'se': False, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 184, 'nout': 80, 'se': False, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 184, 'nout': 80, 'se': False, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 480, 'nout': 112, 'se': True, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 3, 'exp': 672, 'nout': 160, 'se': True, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 672, 'nout': 160, 'se': True, 'nl': ReLU6, 'bn': True, 's': 2}, #
    {'type': 'bneck', 'k': 5, 'exp': 960, 'nout': 160, 'se': True, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'bneck', 'k': 5, 'exp': 960, 'nout': 160, 'se': True, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'conv2d', 'k': 1, 'exp': None, 'nout': 960, 'se': False, 'nl': ReLU6, 'bn': True, 's': 1}, #
    {'type': 'pool', 'k': 7, 'exp': None, 'nout': None, 'se': False, 'nl': None, 'bn': False, 's': 1}, #
    {'type': 'conv2d', 'k': 1, 'exp': None, 'nout': 1280, 'se': False, 'nl': ReLU6, 'bn': False, 's': 1}, #
]



class MobilenetV3():
    """
        Mobilenet V3 Base class
    """

    def __init__(self, architecture: List[Dict]) -> None:
        self._architecture: List[Dict] = architecture
        #
        self._se_factor = 16

    def _inverted_residual_block_v3(self, x_in: layers.Layer, k: int,
                                    exp: int, nout: int, se: bool,
                                    nl: layers.Layer, bn: bool, s: int) -> layers.Layer:
        """
            Mobilenet V3 Inverted Residual Block with SE component
        """

        # expansion
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
            x_se = x
            x_se = layers.GlobalAveragePooling2D()(x_se)
            dim_x_se = x_se.shape[1]
            x_se = layers.Dense(int(floor(dim_x_se / self._se_factor)))(x_se)
            x_se = nl()(x_se)
            x_se = layers.Dense(dim_x_se)(x_se)
            x_se = tf.nn.sigmoid(x_se) # placeholder
            x = layers.Multiply()([x, x_se])
        # projection
        x = layers.Conv2D(nout,1,1,'same')(x)
        if bn:
            x = layers.BatchNormalization()(x)
        x = nl()(x)
        # residual block and output
        if x_in.shape[3] == nout:
            identity = x_in
        else:
            identity = layers.Conv2D(nout, 1, 1, 'same')(x_in)
            if bn:
                identity = layers.BatchNormalization()(identity)
        if s == 2:
            x_out = x
        else:
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
        x_out = layers.AveragePooling2D(k,s,'valid')(x_in)
        return x_out


class MobilenetV3Large(MobilenetV3):
    """
        Mobilenet V3 Large sub-Class
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 n_classes: int,
                 alpha: float = 1.0) -> None:
        """
            TODO: expand this
        """
        super().__init__(MOBILENETV3_LARGE)
        self._input_shape: Tuple[int, int, int] = input_shape
        self._n_classes: int = n_classes
        self._alpha: float = alpha

    def _build_network(self) -> Union[layers.Layer, layers.Layer]:
        """
            Build network from architecture
        """

        x_in = layers.Input(self._input_shape)

        x = x_in
        for layer_idx, layer_dict in enumerate(self._architecture):
            if layer_dict['type'] == 'conv2d':
                x = self._conv2d_block(
                    x, layer_dict['k'], layer_dict['exp'], layer_dict['nout'],
                    layer_dict['se'], layer_dict['nl'], layer_dict['bn'],
                    layer_dict['s'])
            elif layer_dict['type'] == 'bneck':
                x = self._inverted_residual_block_v3(
                    x, layer_dict['k'], layer_dict['exp'], layer_dict['nout'],
                    layer_dict['se'], layer_dict['nl'], layer_dict['bn'],
                    layer_dict['s'])
            elif layer_dict['type'] == 'pool':
                x = self._pool_block(
                    x, layer_dict['k'], layer_dict['exp'], layer_dict['nout'],
                    layer_dict['se'], layer_dict['nl'], layer_dict['bn'],
                    layer_dict['s'])

        x_out = layers.Conv2D(self._n_classes,1,1,'valid')(x)
        x_out = layers.Softmax()(x_out)
        return x_in, x_out

    def __call__(self) -> models.Model:
        """
            Build Keras Model
        """

        x_in, x_out = self._build_network()
        model = models.Model(inputs = x_in, outputs = x_out)

        return model



mobilenetv3 = MobilenetV3Large((224,224,3),10)
model = mobilenetv3()

tf.keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
model.save('mobilenetv3.h5')