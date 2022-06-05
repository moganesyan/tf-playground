import tensorflow as tf


def pairwise_similarity(u: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
    """
        Calculate pairwise similarity between two vectors
            using cos distance.

        args:
            u: tf.Tensor - First input bector.
            v: tf.Tensor - Second input vector.
        returns:
            score: tf.Tensor - Similarity score scalar.
    """

    score = tf.tensordot(tf.transpose(u), v, axes = 1) / (tf.norm(u, 2) * tf.norm(v, 2))
    return score


# def pairwise_loss(batch_u: tf.Tensor,
#                   batch_v: tf.Tensor,

#                   temp: float = 0.001) -> tf.Tensor:
#     """
#         Pairwise loss between two vectors.
#     """

def ntxent_loss(batch_u: tf.Tensor,
                batch_v: tf.Tensor,
                temp: float = 0.001) -> tf.Tensor:
    """
        Normalised temperature-scaled cross entropy loss.

        args:
            batch_u: tf.Tensor - First batch.
            batch_v: tf.Tebsor - Second batch.
            temp: float - Temperate scale coefficient.
        returns:
            loss: tf.Tensor - Loss scalar.
    """

    n_batch = tf.shape(batch_u)[0]
    loss = tf.constant(0)

    # first permutation
    for k in tf.range(n_batch):
        # get u and v bectors
        u_outer = batch_u[k]
        v_outer = batch_v[k]

        # compute pairwise loss between u and v
        pwise_sim = tf.math.exp(pairwise_similarity(u_outer, v_outer) / temp)

        # compute pairwise negative loss for negative examples from batch
        pwise_sim_negative = tf.constant(0)

        # 1) sample from batch u expluding u_outer
        for k_inner in tf.range(1, n_batch):
            v_inner = batch_u[k_inner]
            pwise_sim_negative = pwise_sim_negative + tf.math.exp(pairwise_similarity(u_outer, v_inner) / temp)

        # 2) sample from batch v fully
        for k_inner in tf.range(n_batch):
            v_inner = batch_v[k_inner]
            pwise_sim_negative = pwise_sim_negative + tf.math.exp(pairwise_similarity(u_outer, v_inner) / temp)

        pairwise_loss = -1 * tf.math.log(pwise_sim / pwise_sim_negative)

        loss = loss + pairwise_loss

    # 2nd permutation
    for k in tf.range(n_batch):
        # get u and v bectors
        u_outer = batch_v[k]
        v_outer = batch_u[k]

        # compute pairwise loss between u and v
        pwise_sim = tf.math.exp(pairwise_similarity(u_outer, v_outer) / temp)

        # compute pairwise negative loss for negative examples from batch
        pwise_sim_negative = tf.constant(0)

        # 1) sample from batch u expluding u_outer
        for k_inner in tf.range(1, n_batch):
            v_inner = batch_v[k_inner]
            pwise_sim_negative = pwise_sim_negative + tf.math.exp(pairwise_similarity(u_outer, v_inner) / temp)

        # 2) sample from batch v fully
        for k_inner in tf.range(n_batch):
            v_inner = batch_u[k_inner]
            pwise_sim_negative = pwise_sim_negative + tf.math.exp(pairwise_similarity(u_outer, v_inner) / temp)

        pairwise_loss = -1 * tf.math.log(pwise_sim / pwise_sim_negative)

        loss = loss + pairwise_loss

    return loss / (2 * n_batch)
