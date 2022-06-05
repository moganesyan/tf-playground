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
        label = tf.one_hot(label, num_classes)

        # apply augmentations
        image_augmented = image[tf.newaxis, ...]
        image_augmented = apply_gaussian_noise(image_augmented)
        image_augmented = random_crop_and_resize(image_augmented)
        image_augmented = colour_distortion(image_augmented)
        image_augmented = image_augmented[0]

        return image, image_augmented, label
    
    return transform_data