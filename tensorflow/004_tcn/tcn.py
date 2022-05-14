import tensorflow as tf

from layers import Dense, GlobalMaxPooling1D
from blocks import ResidualTCNBlock


class TCN(tf.Module):
    """
        Temporal Convolution Neural Network.
    """

    def __init__(self,
                 num_classes: int,
                 name: str = None) -> None:
        """
            args:
                num_classes: int - Number of output classes.
                name: str - Name of the model.
            returns:
                None
        """

        super(TCN, self).__init__(name)
        self._num_classes = num_classes

        self._block_1 = ResidualTCNBlock(100, 2, 2, 0.1, name = 'tcn_block_1')
        self._block_2 = ResidualTCNBlock(100, 2, 4, 0.1, name = 'tcn_block_2')
        self._block_3 = ResidualTCNBlock(200, 3, 8, 0.1, name = 'tcn_block_3')
        self._block_4 = ResidualTCNBlock(200, 3, 16, 0.1, name = 'tcn_block_4')
        self._maxpool = GlobalMaxPooling1D(name = 'global_max_pool')
        self._fc1 = Dense(num_classes, name = 'out_classes')

    # @tf.function(input_signature = [tf.TensorSpec(shape=(None, 100, 50), dtype=tf.float32)])
    def __call__(self,
                 x_in: tf.Tensor,
                 training: bool = False) -> tf.Tensor:
        """
            args:
                x_in: tf.Tensor - Input tensor.
                training: bool - Flag to toggle train time and inference time behaviour.
            returns:
                x_out: tf.Tensor - Output tensor.
        """

        x = self._block_1(x_in, training = training)
        x = self._block_2(x, training = training)
        x = self._block_3(x, training = training)
        x = self._block_4(x, training = training)
        print(x.shape)
        x = self._maxpool(x)
        print(x.shape)
        x = self._fc1(x)
        print(x.shape)

        x_out = tf.nn.softmax(x, name = 'softmax')

        return x_out
