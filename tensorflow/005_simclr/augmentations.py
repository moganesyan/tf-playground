from typing import Tuple

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
        kernel_size = tf.cast(x_in.shape[1], tf.float32) * tf.constant(0.10)
        kernel_size = tf.cast(kernel_size, tf.int32)

        kernel = get_gaussian_kernel(kernel_size, blur_strength)
        kernel = kernel[..., tf.newaxis]
        if len(x_in.shape) == 4:
            kernel = tf.tile(
                kernel, tf.constant([1, 1, x_in.shape[-1]]))
        kernel = kernel[..., tf.newaxis]

        x_in_reshaped = x_in
        if len(x_in.shape) < 4:
            x_in_reshaped = x_in_reshaped[..., tf.newaxis]
        x_out = tf.nn.depthwise_conv2d(x_in_reshaped, kernel, [1,1,1,1], 'SAME')
    else:
        x_out = x_in
    return x_out


def random_crop_and_resize(x_in: tf.Tensor,
                           crop_size: Tuple[float, float] = (0.08, 1.0),
                           aspect_range: Tuple[float, float] = (0.75, 1.33),
                           num_tries: int = 100) -> tf.Tensor:
    """
        Random crop and resize based on crop size and aspect ratio ranges.
            1) Sample crop size and aspect ratio.
            2) Get crop dimensions.
            3) Adjust crop dimensions to aspect ratio.
            3) Check that the crop dimensions are valid.
            4) Crop image based on valid crop dimensions and resize to original dimensions.
            5) Return original image if valid crop can't be generated within num_tries.

        args:
            x_in: tf.Tensor - Input image tensor.
            crop_size: Tuple[float, float] - Crop size range (proprtion of input image).
            aspect_range: Tuple[float, float] - Aspect ratio range.
            num_tries: int - Number of tries to generate crop within given constraints.
        returns:
            x_out: tf.Tensor - Cropped image tensor.
    """

    crop_size_min = crop_size[0]
    crop_size_max = crop_size[1]

    aspect_ratio_min = aspect_range[0]
    aspect_ratio_max = aspect_range[1]

    w_original = tf.cast(tf.shape(x_in)[2], tf.float32)
    h_original = tf.cast(tf.shape(x_in)[1], tf.float32)

    for _ in tf.range(num_tries):
        # randomly get crop area and aspect ratio
        crop_size = tf.random.uniform(
            (), minval = crop_size_min, maxval = crop_size_max)
        aspect_ratio = tf.random.uniform(
            (), minval = aspect_ratio_min, maxval = aspect_ratio_max)

        # calculate the desired height and width of crop based on crop size
        num_pixels_original = h_original * w_original
        num_pixels_new = tf.math.floor(num_pixels_original * crop_size)

        w_new = tf.math.floor(tf.math.sqrt(aspect_ratio * num_pixels_new))
        h_new = tf.math.floor(num_pixels_new / w_new)

        if w_new <= w_original and h_new <= h_original:
            # randomly crop based on dimensions
            if len(x_in.shape) < 4:
                crop_dims = (tf.shape(x_in)[0], h_new, w_new)
            else:
                crop_dims = (tf.shape(x_in)[0], h_new, w_new, x_in.shape[3])

            crop = tf.image.random_crop(x_in, crop_dims)
            if len(x_in.shape) < 4:
                crop = crop[..., tf.newaxis]

            resize_dims = [x_in.shape[1], x_in.shape[2]]
            crop_resized = tf.squeeze(tf.image.resize(crop, resize_dims))
            return crop_resized
    return x_in


def colour_jitter(x_in: tf.Tensor) -> tf.Tensor:
    """
        Apply colour jitter.

        1) Tweak brightness.
        2) Tweak contrast.
        3) Tweak saturation.
        4) Tweak hue.

        args:
            x_in: tf.Tensor - Input image tensor.
        returns:
            x_out: tf.Tensor - Augmented image tensor.
    """

    x = tf.image.random_brightness(x_in, max_delta=0.8)
    x = tf.image.random_contrast(x, lower=1-0.8, upper=1+0.8)
    x = tf.image.random_saturation(x, lower=1-0.8, upper=1+0.8)
    x = tf.image.random_hue(x, max_delta=0.2)
    x_out = tf.clip_by_value(x, 0, 1)

    return x_out


def colour_drop(x_in: tf.Tensor) -> tf.Tensor:
    """
        Apply colour jitter.

        1) Convert to grayscale.
        2) Reconvert to RGB.

        args:
            x_in: tf.Tensor - Input image tensor.
        returns:
            x_out: tf.Tensor - Augmented image tensor.
    """

    x = tf.image.rgb_to_grayscale(x_in)
    x_out = tf.tile(x, [tf.shape(x_in)[0], 1, 1, tf.shape(x_in)[3]])

    return x_out


def colour_distortion(x_in: tf.Tensor) -> tf.Tensor:
    """
        Apply colour distortion augmentations.

        1) Apply random colour jitter.
        2) Apply random colour drop.

        args:
            x_in: tf.Tensor - Input image tensor.
        returns:
            x_out: tf.Tensor - Augmented image tensor.
    """

    apply_jitter = tf.random.uniform(
        (), minval = 0, maxval = 1.0, dtype = tf.float32)
    apply_drop = tf.random.uniform(
        (), minval = 0, maxval = 1.0, dtype = tf.float32)

    if len(x_in.shape) < 4:
        x_out = x_in[..., tf.newaxis]

    if apply_jitter <= 0.80:
        x_out = colour_jitter(x_out)
    if len(x_in.shape) == 4:
        if apply_drop <= 0.20:
            x_out = colour_drop(x_out)

    return x_out
