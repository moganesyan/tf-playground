import numpy as np
import tensorflow as tf

from alexnet import AlexNet


# load CIFAR10 dataset
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# define image processing function
def process_inputs(image, label):
    image = image / 255
    image = tf.image.resize(image, (224,224))
    image = tf.cast(image, tf.float32)

    label = tf.one_hot(label, len(CLASS_NAMES))
    label = tf.reshape(label, (-1,))
    return image, label

# train test split
validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

# Prepare the training dataset.
BATCH_SIZE = 5

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.map(process_inputs)
train_dataset = train_dataset.batch(BATCH_SIZE)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
val_dataset = val_dataset.map(process_inputs)
val_dataset = val_dataset.batch(BATCH_SIZE)

# declare model
model = AlexNet(len(CLASS_NAMES))

# declare loss function
def loss_fn(y_pred, y_true):
    ce = tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, 0.00001, 1.0)), axis = 1)
    return tf.reduce_mean(-ce)

# train model
EPOCHS = 10
LEARNING_RATE = 0.001

for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            y_pred = model(x_batch_train, training = True)
            loss_value = loss_fn(y_pred, y_batch_train)

            grads = tape.gradient(loss_value, model.trainable_variables)

        for grad, weight in zip(grads, model.trainable_variables):
            weight.assign_sub(LEARNING_RATE * grad)

        if step % 10 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))
