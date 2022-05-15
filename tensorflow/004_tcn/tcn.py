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
                 architecture: List[Dict],
                 name: str = None) -> None:
        """
            args:
                num_classes: int - Number of output classes.
                architecture: List[Dict] - Architecture defined in block format.
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
        for block_idx, block_params in enumerate(self._architecture):
            for layer_idx in range(block_params["num_layers"]):
                dilation_rate_current = 2 ** layer_idx
                self._layers.append(
                    ResidualTCNBlock(
                        block_params["num_channels"],
                        block_params["kernel"],
                        dilation_rate_current,
                        block_params["dropout"],
                        name = f"residual_block_{layer_idx}"
                    )
                )
        # global max pooling
        self._layers.append(
            GlobalMaxPooling1D(name = "global_maxpool")
        )
        # final dense layer
        self._layers.append(
            Dense(
                self._num_classes,
                name = "final_dense"
            )
        )

    @tf.function(input_signature = [tf.TensorSpec(shape = (None, 20, 1), dtype = tf.float32)])
    def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
        """
            Build graph on first call.

            args:
                x_in: tf.Tensor - Input tensor.
            returns:
                x_out: tf.Tesnor - Output tensor.
        """

        if not self._is_built:
            self._build_architecture()
            self._is_built = True

        for layer_idx, layer in enumerate(self._layers):
            if layer_idx == 0:
                x = layer(x_in)
            else:
                x = layer(x)
        x_out = tf.nn.softmax(x, name = 'final_softmax')

        return x_out
