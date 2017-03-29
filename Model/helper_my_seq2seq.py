from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
import math


def gs(tensor):
    return tensor.get_shape().as_list()


def length_sequence(sequence):
    """
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

        # transform encoder_inputs
        return tf.nn.embedding_lookup(
            embedding, inputs), embedding


class Decoder:
    def __init__(self,
                 name,
                 num_symbols,
                 cell,
                 beam_size,
                 beam_search,
                 do_attention,
                 embedding_size,
                 share_embedding_with_encoder,
                 output_projection,
                 train_encoder_weight,
                 max_decoder_sequence_length):
        # Name of the decoder (use for variable scope)
        self.name = name
        # Type of cell
        self.cell = cell  # TODO: work only with GRU cell because they contains one state vector and I admit it in the attention function
        # Size of the beam search
        self.beam_size = beam_size
        # Size of the vocabulary
        self.nb_symbols = num_symbols
        # Boolean, True if do beam search
        self.beam_search = beam_search
        # Boolean True if do attention mechanism on encoder input
        self.do_attention = do_attention
        # Size of the input embedding
        self.embedding_size = embedding_size
        # TODO Not use yet (because nb_symbols is small)
        self.output_projection = output_projection
        # Boolean, True, if embedding dictionary is share with the encoder
        self.share_embedding_with_encoder = share_embedding_with_encoder
        # Boolean, True if weights of the encoder is trained with the decoder loss
        self.train_encoder_weight = train_encoder_weight
        # Maximum size of a sequence length
        self.max_decoder_sequence_length = max_decoder_sequence_length
        # Init values, will be filled when calling seq2seq
        self.output = None
        self.train_fn = None
        self.loss = None

        # Current step
        self.global_step = tf.Variable(0, trainable=False)
        # Tensor bool
        # (TODO: should also be used for dropout, dropout is currently set in the test_main with a DropoutWrapper)
        # but it is currenlty used to set the beam search (Beam search only works at testing time)
        self.is_training = tf.placeholder(tf.bool)

        # Placeholder, maximum length of every sequence in the batch! Must be feed
        self.max_length_decoder_in_batch = tf.placeholder(tf.int32)

        self.targets = []
        self.decoder_inputs = []
        for i in range(self.max_decoder_sequence_length):
            self.targets.append(tf.placeholder(tf.int32, shape=[None],
                                               name="decoder{0}".format(i)))

        # decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        self.decoder_inputs = [tf.zeros_like(self.targets[0], dtype=tf.int32, name='GO')] + self.targets[:-1]
        self.target_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.targets]

        # It should always be false
        if output_projection is False:
            self.cell = tf.contrib.rnn.OutputProjectionWrapper(self.cell, self.nb_symbols)

        if beam_size:
            # Init values. Will be filled when calling seq2seq
            self.beam_path = None
            self.beam_symbol = None


def _extract_beam_search(embedding, beam_size, num_symbols, embedding_size):
    def loop_function(prev, i, log_beam_probs, beam_path, beam_symbols):
        # (batch_size + I(i > 1) * beam_size) x nb_symbols
        probs = tf.log(tf.nn.softmax(prev))

        if i > 1:
            # log_beam_probs[-1] is of shape (batch_size x beam_size and it contains the previous probability
            # Sum the accumulated probability with the last probability
            probs = tf.reshape(probs + log_beam_probs[-1],
                               [-1, beam_size * num_symbols])

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


