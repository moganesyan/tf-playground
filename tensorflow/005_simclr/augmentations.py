import tensorflow as tf


def get_gaussian_kernel(k: int, sigma: float) -> tf.Tensor:
    """
        Get kxk 2D gaussian kernel.
        args:
            k: int - Kernel size.
            sigma: float - Blur strength.
        returns:
            kernel_gauss: tf.Tensor - Gaussian kernel tensor.
    """

    x = tf.range(-k // 2 + 1, k // 2 + 1, dtype = tf.float32)

    x_gauss = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(sigma, 2))))
    x_gauss = x_gauss / tf.math.sqrt((2 * 3.14159 * tf.pow(sigma, 2)))

    kernel_gauss = tf.tensordot(x_gauss, x_gauss, axes = 0)
    x_scale = tf.reduce_sum(kernel_gauss)

    kernel_gauss = kernel_gauss / x_scale

    return kernel_gauss


def apply_gaussian_noise(x_in: tf.Tensor) -> tf.Tensor:
    """
        Apply 2D gaussian blur to input tensor.

        - Blur input tensor 50% of the time
        - Randomly sample blur strength [0.1, 2.0]
        - Kernel size is 10% of the input tensor height / width

        args:
            x_in: tf.Tensor - Input tensor.
        returns:
            x_out: tf.Tensor - Augmented tensor.
    """

    roll_augment_flag = tf.random.uniform((),0,1)

    if roll_augment_flag >= 0.50:
        blur_strength = tf.random.uniform((),0.1,2.0)
        kernel_size = tf.cast(tf.shape(x_in)[0], tf.float32) * tf.constant(0.10)
        kernel_size = tf.cast(kernel_size, tf.int32)

        kernel = get_gaussian_kernel(kernel_size, blur_strength)
        kernel = tf.expand_dims(kernel,-1)
        kernel = tf.expand_dims(kernel,-1)

        x_in_reshaped = tf.expand_dims(x_in,0)
        x_in_reshaped = tf.expand_dims(x_in_reshaped,-1)

        x_out = tf.nn.conv2d(x_in_reshaped, kernel, [1,1,1,1], 'SAME')
        x_out = tf.squeeze(x_out)
    else:
        x_out = x_in
    return x_out
