import tensorflow as tf


def pairwise_similarity(u: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
    """
        Calculate pairwise similarity between two sets of vectors
        using cos distance.

        args:
            u: tf.Tensor - First input matrix.
            v: tf.Tensor - Second input matrix.
        returns:
            score: tf.Tensor - Matrix of doct product values.
    """

    numer = tf.tensordot(u, v, axes = [[1],[1]])
    denom = tf.norm(u, 2, axis = 1) * tf.norm(v, 2, axis = 1)

    score =  numer / tf.clip_by_value(denom, 1e-16, 1e16)
    return score


def ntxent_loss(batch_u: tf.Tensor,
                batch_v: tf.Tensor,
                temp: float = 1.0) -> tf.Tensor:
    """
        Normalised temperature-scaled cross entropy loss.

        args:
            batch_u: tf.Tensor - First batch.
            batch_v: tf.Tensor - Second batch.
            temp: float - Temperate scale coefficient.
        returns:
            loss: tf.Tensor - Loss scalar.
    """

    n_minibatch = batch_u.shape[0]

    # normalise embeddings for numerical stability
    batch_u = batch_u / tf.clip_by_value(tf.norm(batch_u, 2, axis = 1), 1e-16, 1e16)
    batch_v = batch_v / tf.clip_by_value(tf.norm(batch_v, 2, axis = 1), 1e-16, 1e16)

    # get similarity matrix
    sim_mat = pairwise_similarity(batch_u, batch_v)
    sim_mat = sim_mat / temp
    sim_mat_t = tf.transpose(sim_mat)
    
    # get simmatch
    sim_pair = tf.linalg.diag_part(sim_mat)
    sim_pair = sim_pair[..., tf.newaxis]
    sim_pair = tf.math.reduce_logsumexp(sim_pair, 1)

    # sum dissimilarities
    # decompose the logarithm and apply logsumexp for numerical stability
    lse = sim_pair - tf.math.reduce_logsumexp(sim_mat, 1)
    lse_t = sim_pair - tf.math.reduce_logsumexp(sim_mat_t, 1)

    # evaluate two sided loss
    loss_pairwise_1 = tf.reduce_sum(-1.0 * lse)
    loss_pairwise_2 = tf.reduce_sum(-1.0 * lse_t)

    loss = (loss_pairwise_1 + loss_pairwise_2) /  (2 * n_minibatch)

    return loss
