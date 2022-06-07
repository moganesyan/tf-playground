import tensorflow as tf


def pairwise_similarity(u: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
    """
        Calculate pairwise similarity between two vectors
            using cos distance.

        args:
            u: tf.Tensor - First input vector.
            v: tf.Tensor - Second input vector.
        returns:
            score: tf.Tensor - Similarity score scalar.
    """

    numer = tf.tensordot(u, v, axes = [[1],[1]])
    denom = tf.norm(u, 2, axis = 1) * tf.norm(v, 2, axis = 1)

    score =  numer / denom
    return score


def ntxent_loss(batch_u: tf.Tensor,
                batch_v: tf.Tensor,
                temp: float = 1.0) -> tf.Tensor:
    """
        Normalised temperature-scaled cross entropy loss.

        args:
            batch_u: tf.Tensor - First batch.
            batch_v: tf.Tebsor - Second batch.
            temp: float - Temperate scale coefficient.
        returns:
            loss: tf.Tensor - Loss scalar.
    """

    n_minibatch = tf.shape(batch_u)[0]
    loss = tf.constant(0.0)

    # get similarity matrix
    sim_mat = pairwise_similarity(batch_u, batch_v)
    sim_mat = tf.math.exp(sim_mat / temp)
    sim_mat_t = tf.transpose(sim_mat)

    # calculate loss
    for k in tf.range(n_minibatch, dtype = tf.int32):
        loss_pairwise_1 = -1.0 * tf.math.log(sim_mat[k, k] / tf.reduce_sum(sim_mat[k,:]))
        loss_pairwise_2 = -1.0 * tf.math.log(sim_mat_t[k, k] / tf.reduce_sum(sim_mat_t[k,:]))
        loss = loss + loss_pairwise_1 + loss_pairwise_2

    return loss / tf.cast((2 * n_minibatch), tf.float32)
