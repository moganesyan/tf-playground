from typing import Tuple
import tensorflow as tf

from augmentations import apply_gaussian_noise, apply_crop_and_resize, apply_colour_distortion


transform_return_type = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


def get_transform_func(num_classes,
                       blur = True,
                       distort_colour = True,
                       crop_resize = True,
                       **kwargs) -> callable:
    """ 
        Get function for data transformation.

        args:
            num_classes (int): Number of classes.
            blur (bool): Toggle gaussian blur augmentation.
            distort_colour (bool): Toggle colour distort augmentation.
            crop_resize (bool): Toggle crop and resize augmentation.
        returns:
            transform_data (callable): Data transformation function.
    """

    @tf.function
    def transform_data(image: tf.Tensor, label: tf.Tensor) -> transform_return_type:
        """
            Transforms image and label_data for contrastive learning.
                1) Convert image to 0-1 range.
                2) Apply random augmentations.
                3) Return augmented and unaugmented images.
                4) One hot encode labels

            args:
                image (tf.Tensor): Input image.
                label (tf.Tensor): Input label.
            returns:
                output_payload (transform_return_type): processed original image, 
                    augmented image, processed original label
        """

        # preprocess image and label
        image = tf.cast(image, tf.float32) / 255.
        image = image[tf.newaxis, ...]
        label = tf.one_hot(label, num_classes)

        image_augmented_1 = image
        image_augmented_2 = image

        # apply augmentations
        if crop_resize:
            image_augmented_1 = apply_crop_and_resize(image_augmented_1, **kwargs)
            image_augmented_2 = apply_crop_and_resize(image_augmented_2, **kwargs)
        if distort_colour:
            image_augmented_1 = apply_colour_distortion(image_augmented_1, **kwargs)
            image_augmented_2 = apply_colour_distortion(image_augmented_2, **kwargs)
        if blur:
            image_augmented_1 = apply_gaussian_noise(image_augmented_1, **kwargs)
            image_augmented_2 = apply_gaussian_noise(image_augmented_2, **kwargs)

        # remove batch dimension
        image_augmented_1 = image_augmented_1[0]
        image_augmented_2 = image_augmented_2[0]

        return image_augmented_1, image_augmented_2, label

    return transform_data
