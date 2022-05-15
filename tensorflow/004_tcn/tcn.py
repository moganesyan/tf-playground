from typing import List, Dict

import tensorflow as tf

from layers import Dense, GlobalMaxPooling1D
from blocks import ResidualTCNBlock


class TCN(tf.Module):
    """
        Temporal Convolution Neural Network.
    """

    def __init__(self,
                 num_classes: int,
                 architecture: Dict,
                 name: str = None) -> None:
        """
            args:
                num_classes: int - Number of output classes.
                architecture: Dict - Architecture defined in block format.
                    Example:
                        {'kernel': 6, 'num_layers': 9, 'num_channels': 10 , 'dropout': 0.05}
                name: str - Name of the model.
            returns:
                None
        """

        super(TCN, self).__init__(name)
        self._num_classes = num_classes
        self._architecture = architecture

        # placeholder for graph
        self._layers = []

        # flag to build graph on first call
        self._is_built = False

    def _build_architecture(self) -> None:
        """
            Build graph from the architecture definition.
            Store created layers in a list.

            args:
                None
            returns:
                None
        """

        # build convolutional blocks
        for layer_idx in range(self._architecture["num_layers"]):
            # dilation_rate_current = 2 ** layer_idx
            dilation_rate_current = 1
            self._layers.append(
                ResidualTCNBlock(
                    self._architecture["num_channels"],
                    self._architecture["kernel"],
                    dilation_rate_current,
                    self._architecture["dropout"],
                    name = f"residual_block_{layer_idx}"
                )
            )

        # global max pooling
        self._maxpool = GlobalMaxPooling1D(name = "global_maxpool")

        # final dense layer
        self._fc_final = Dense(
                self._num_classes,
                name = "final_dense"
            )

    def __call__(self,
                 x_in: tf.Tensor,
                 training: bool = False) -> tf.Tensor:
        """
            Build graph on first call.

            args:
                x_in: tf.Tensor - Input tensor.
                training: bool - Flag for train and inference time behaviour.
            returns:
                x_out: tf.Tesnor - Output tensor.
        """

        if not self._is_built:
            self._build_architecture()
            self._is_built = True

        for layer_idx, layer in enumerate(self._layers):
            if layer_idx == 0:
                x = layer(x_in, training = training)
            else:
                x = layer(x, training = training)

        x = self._maxpool(x)
        x = self._fc_final(x)
        x_out = tf.nn.softmax(x, name = 'final_softmax')

        return x_out
