import tensorflow as tf
from layers import Conv2D, BatchNormalization, Dense


class AlexNet(tf.Module):
    def __init__(self, output_size: int, name: str = 'alexnet'):
        super(AlexNet, self).__init__(name = name)

        self.conv1 = Conv2D(
            96, 11, 4, 'same',
            name = 'conv1', use_bias=False
        )
        self.conv2 = Conv2D(
            256, 5, 1, 'same',
            name = 'conv2', use_bias=False
        )
        self.conv3 = Conv2D(
            384, 3, 1, 'same',
            name = 'conv3', use_bias=False
        )
        self.conv4 = Conv2D(
            384, 3, 1, 'same',
            name = 'conv4', use_bias=False
        )
        self.conv5 = Conv2D(
            256, 3, 1, 'same',
            name = 'conv5', use_bias=False
        )

        self.fc1 = Dense(
            4096, name = 'fc1'
        )
        self.fc2 = Dense(
            4096, name = 'fc2'
        )
        self.fc_out = Dense(
            output_size, name = 'fc_out'
        )

    @tf.Module.with_name_scope
    def __call__(self, x_in: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(x_in)
        x = tf.nn.relu(x, name = 'relu1')
        x = tf.nn.max_pool2d(x, 3, 2, 'SAME', name = 'pool1')
        x = self.conv2(x)
        x = tf.nn.relu(x, name = 'relu2')
        x = tf.nn.max_pool2d(x, 3, 2, 'SAME', name = 'pool2')
        x = self.conv3(x)
        x = tf.nn.relu(x, name = 'relu3')
        x = self.conv4(x)
        x = tf.nn.relu(x, name = 'relu4')
        x = self.conv5(x)
        x = tf.nn.relu(x, name = 'relu5')
        x = tf.nn.max_pool2d(x, 3, 2, 'SAME', name = 'pool3')
        x = tf.reshape(x, (x.shape[0], -1), name = 'flatten')
        x = self.fc1(x)
        x = tf.nn.relu(x, name = 'relu6')
        x = tf.nn.dropout(x, 0.50, name = 'dropout1')
        x = self.fc2(x)
        x = tf.nn.relu(x, name = 'relu7')
        x = tf.nn.dropout(x, 0.50, name = 'dropout2')
        x = self.fc_out(x)
        x_out = tf.nn.softmax(x, name = 'softmax')

        return x_out


class AlexNetBN(tf.Module):
    def __init__(self, output_size: int, name: str = 'alexnet_bn'):
        super(AlexNetBN, self).__init__(name = name)

        self.conv1 = Conv2D(
            96, 11, 4, 'same',
            name = 'conv1', use_bias=False
        )
        self.conv2 = Conv2D(
            256, 5, 1, 'same',
            name = 'conv2', use_bias=False
        )
        self.conv3 = Conv2D(
            384, 3, 1, 'same',
            name = 'conv3', use_bias=False
        )
        self.conv4 = Conv2D(
            384, 3, 1, 'same',
            name = 'conv4', use_bias=False
        )
        self.conv5 = Conv2D(
            256, 3, 1, 'same',
            name = 'conv5', use_bias=False
        )

        self.bn1 = BatchNormalization(name = 'bn1')
        self.bn2 = BatchNormalization(name = 'bn2')
        self.bn3 = BatchNormalization(name = 'bn3')
        self.bn4 = BatchNormalization(name = 'bn4')
        self.bn5 = BatchNormalization(name = 'bn5')

        self.fc1 = Dense(
            4096, name = 'fc1'
        )
        self.fc2 = Dense(
            4096, name = 'fc2'
        )
        self.fc_out = Dense(
            output_size, name = 'fc_out'
        )

    @tf.Module.with_name_scope
    def __call__(self, x_in: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(x_in)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x, name = 'relu1')
        x = tf.nn.max_pool2d(x, 3, 2, 'SAME', name = 'pool1')
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x, name = 'relu2')
        x = tf.nn.max_pool2d(x, 3, 2, 'SAME', name = 'pool2')
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x, name = 'relu3')
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = tf.nn.relu(x, name = 'relu4')
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = tf.nn.relu(x, name = 'relu5')
        x = tf.nn.max_pool2d(x, 3, 2, 'SAME', name = 'pool3')
        x = tf.reshape(x, (x.shape[0], -1), name = 'flatten')
        x = self.fc1(x)
        x = tf.nn.relu(x, name = 'relu6')
        x = tf.nn.dropout(x, 0.50, name = 'dropout1')
        x = self.fc2(x)
        x = tf.nn.relu(x, name = 'relu7')
        x = tf.nn.dropout(x, 0.50, name = 'dropout2')
        x = self.fc_out(x)
        x_out = tf.nn.softmax(x, name = 'softmax')

        return x_out
