import enum
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)

from tcn import TCN

# load mnist datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# create Tensorflow Dataset
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Define experiment parameters
NUM_CLASSES = len(np.unique(y_test))
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
ARCHITECTURE = {'kernel': 7, 'num_layers': 8, 'num_channels': 25 , 'dropout': 0.05}

# create sequential p-mnist dataset
def transform_data(image, label):
    """
        Transforms MNIST data into sequential p-MNIST dataset.
            1) Flatten image tensor
            2) Shuffle image tensor
            3) Convert into float

        Also onehot encodes the label
    """

    image = tf.reshape(image, (-1,1))
    image = tf.random.shuffle(image)
    image = tf.cast(image, tf.float32) / 255.

    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

# create train dataset
train_data = train_data.map(
    transform_data, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.cache()
train_data = train_data.shuffle(buffer_size=1024)
train_data = train_data.batch(BATCH_SIZE, drop_remainder = True)
train_data = train_data.prefetch(tf.data.AUTOTUNE)

# create test dataset
test_data = test_data.map(
    transform_data, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.cache()
test_data = test_data.batch(BATCH_SIZE, drop_remainder = True)
test_data = test_data.prefetch(tf.data.AUTOTUNE)

# get TCN model
model = TCN(NUM_CLASSES, ARCHITECTURE, 'tcn_pmnist')

# define loss and training loop function
def loss_fn(y_pred, y_true):
    """
        Categorical CE loss function.
    """

    _epsilon = tf.constant(0.00001)
    ce = tf.reduce_sum(
        y_true * tf.math.log(tf.clip_by_value(y_pred, _epsilon, 1.0 - _epsilon)),
        axis = 1)
    return tf.reduce_mean(-ce)

@tf.function
def train_step(x_batch, y_batch):
    """
        Training step wrapped in a tf Function.
    """

    with tf.GradientTape() as tape:
        y_pred = model(x_batch, training = True)
        loss_value = loss_fn(y_pred, y_batch)

    grads = tape.gradient(loss_value, model.trainable_variables)
    for grad, weight in zip(grads, model.trainable_variables):
        weight.assign_sub(LEARNING_RATE * grad)

    return loss_value

@tf.function
def test_step(x_batch, y_batch):
    """
        Test step wrapped in a tf Function.
    """

    y_pred = model(x_batch, training = False)
    loss_value = loss_fn(y_pred, y_batch)

    return loss_value

def train():
    """
        Train TCN with sequential p-MNIST dataset.
    """

    losses_train = []
    for epoch in range(EPOCHS):
        print(f"Working on epoch: {epoch}.")
        for train_idx, (x_batch_train, y_batch_train) in enumerate(train_data):
            loss_value_train = train_step(x_batch_train, y_batch_train)
            losses_train.append(loss_value_train.numpy())

            if train_idx % 100 == 0:
                print(f"Loss at step {train_idx} is {loss_value_train.numpy():.2f}.")

        print(f"Finished epoch: {epoch}. Running test inference...")

        losses_test = []
        for test_idx, (x_batch_test, y_batch_test) in enumerate(test_data):
            loss_value_test = test_step(x_batch_test, y_batch_test)
            losses_test.append(loss_value_test.numpy())

        print(f"Test set loss: {np.mean(losses_test):.2f}.")


    return losses_train


# train and plot loss
train_losses = train()








