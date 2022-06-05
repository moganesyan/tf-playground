from typing import Tuple
import tensorflow as tf

from augmentations import apply_gaussian_noise, random_crop_and_resize, colour_distortion


transform_return_type = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


def get_transform_func(num_classes) -> callable:
    """
        Get function for data transformation.

        args:
            num_classes: int - Number of classes.
        returns:
            transform_data: callable - Data transformation function.
    """

    def transform_data(image: tf.Tensor, label: tf.Tensor) -> transform_return_type:
        """
            Transforms image and label_data for contrastive learning.
                1) Convert image to 0-1 range.
                2) Apply random augmentations.
                3) Return augmented and unaugmented images.
                4) One hot encode labels

            args:
                image: tf.Tensor - Input image.
                label: tf.Tensor - Input label.
            returns:
                output_payload: transform_return_type - processed original image, 
                    augmented image, processed original label
        """

        # preprocess image and label
        image = tf.cast(image, tf.float32) / 255.
        image = image[tf.newaxis, ...]
        label = tf.one_hot(label, num_classes)

        # apply augmentations (1)
        image_augmented_1 = image
        image_augmented_1 = random_crop_and_resize(image_augmented_1)
        image_augmented_1 = colour_distortion(image_augmented_1, strength = 0.50)
        image_augmented_1 = apply_gaussian_noise(image_augmented_1)
        image_augmented_1 = image_augmented_1[0]

        # apply augmentations (2)
        image_augmented_2 = image
        image_augmented_2 = random_crop_and_resize(image_augmented_2)
        image_augmented_2 = colour_distortion(image_augmented_2, strength = 0.50)
        image_augmented_2 = apply_gaussian_noise(image_augmented_2)
        image_augmented_2 = image_augmented_2[0]

        return image_augmented_1, image_augmented_2, label

    return transform_data
