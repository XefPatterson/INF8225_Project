from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
import tensorflow as tf
import math


def length_sequence(sequence):
    """
    Compute the length of each sentence in a batch
    @sequence: 3D tensor of shape (batch_size, sequence_length, embedding_size)
    """
    used = tf.sign(tf.reduce_sum(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length  # vector of size (batch_size) containing sentence lengths


def embedded_sequence(inputs,
                      num_symbols,
                      embedding_size,
                      embedding=None):
    with vs.variable_scope("embedded_sequence"):
        if embedding is None:
            # Embedding should have variance = 1
            sqrt3 = math.sqrt(3)
            initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

            # Create embedding for the encoder symbol
            embedding = vs.get_variable("embedding", [num_symbols, embedding_size],
                                        initializer=initializer,
                                        dtype=tf.float32)
            if isinstance(inputs, list):
                inputs = tf.stack(inputs, 1)

        # transform encoder_inputs
        return tf.nn.embedding_lookup(
            embedding, inputs), embedding

def _extract_beam_search(embedding, beam_size, num_symbols, embedding_size):
    def loop_function(prev, i, log_beam_probs, beam_path, beam_symbols):
        # (batch_size + I(i > 1) * beam_size) x nb_symbols
        probs = tf.log(tf.nn.softmax(prev))

        if i > 1:
            # log_beam_probs[-1] is of shape (batch_size x beam_size and it contains the previous probability
            # Sum the accumulated probability with the last probability
            probs = tf.reshape(probs + log_beam_probs[-1],
                               [-1, beam_size * num_symbols])
        if i == 1:
            probs = tf.gather(probs, tf.range(tf.shape(probs)[0] // beam_size))

        # Retrieve the top #beam_size best element for a single batch.
        # If i > 1, then probs is of size batch_size x (beam_size x nb_symbols)
        best_probs, indices = tf.nn.top_k(probs, beam_size)
        indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))

        # If i > 1, then indices contains value larger than vocab_output_size
        # Hence, we need to reduce them
        symbols = indices % num_symbols

        # For each indices, we now know what was the previous sentences, it was used for
        beam_parent = indices // num_symbols  # Which hypothesis it came from.

        # Fill the list
        beam_symbols.append(symbols)
        beam_path.append(beam_parent)
        log_beam_probs.append(best_probs)

        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.

        emb_prev = tf.nn.embedding_lookup(embedding, symbols)
        emb_prev = tf.reshape(emb_prev, [-1, embedding_size])
        return emb_prev

    return loop_function


def _extract_argmax_and_embed(embedding):
    def loop_function(prev, _):
        prev_symbol = math_ops.argmax(prev, 1)
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        return emb_prev

    return loop_function
