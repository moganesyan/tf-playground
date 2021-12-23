from typing import Union, Optional, Tuple

import tensorflow as tf

import layers


class MBConv6(tf.Module):
    """
        MBConv6 Block.
        This is essentially an 'Inverted Residual' block from the paper.
    """

    def __init__(self,
                 nout: int,
                 exp: int,
                 kernel: int,
                 stride: int,
                 activation: tf.Module,
                 use_bn: bool,
                 name=None):

        super(MBConv6, self).__init__(name=name)

        self.nout: int = nout
        self.exp: int = exp
        self.kernel: int = kernel
        self.stride: int = stride

        self.is_built: bool = False

        # initialise activation
        self.act_expansion= activation("act_expansion")
        self.act_dwise = activation("act_dwise")

        # initialise batch norms
        # TODO: Set up batch normalisations

        # preinitialise conv layers
        self.conv_expansion: Optional[tf.Module] = None
        self.conv_dwise: Optional[tf.Module] = None
        self.conv_projection: Optional[tf.Module] = None

        def __call__(self, x_in: tf.Tensor) -> tf.Tensor:
            """
                Build on first call and evaluate output tensor.
            """

            if not self.is_built:
                n_expanded = x_in.shape[3] * self.exp

                self.conv_expansion = layers.Conv2D(
                    n_expanded, 1, 1, 'same',
                    use_bias = False, name = "conv_expansion"
                )
                self.conv_dwise = layers.DepthwiseConv2D(
                    kernel, stride, 'same',
                    use_bias = False, name = "conv_dwise"
                )
                self.conv_projection = layers.Conv2D(
                    nout, 1, 1, 'same',
                    use_bias = False, name = "conv_projection"
                )

                self.is_built = True

            # TODO: Apply batch normalisations
            x = self.conv_expansion(x_in)
            x = self.act_expansion(x)
            x = self.conv_dwise(x)
            x = self.act_dwise(x)
            x = self.conv_projection(x)

            # TODO: Apply stochastic dropout
            # create conditional residual connection
            identity = tf.identity(x_in)
            if stride > 1 or x_in.shape[3] != self.nout:
                return x
            else:
                return tf.add(identity, x)
