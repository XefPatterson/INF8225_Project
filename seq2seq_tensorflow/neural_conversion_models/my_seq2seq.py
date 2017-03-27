"""
SEQ2SEQ ENCODER:
* nombre de layers
* bidirectionnal
* dynamic_rnn ou bucket
* different cell (different RNN cell may capture different semantic, long term vs short term dependancies)


SEQ2SEQ DECODER
* multiple decoder (for example, one may try to reconstruct the encoded sentences, and the other one to predict the answer)
* attention mechanism on the encoder cell output
* can receive multiple input state instead of one (for example one encoder is at char level, another one is at word level
* beam search
* allow different loss function
* output_projection
* share embedding with the encoder (yes, all model have different embedding vocab between encoder and decoder

"""
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
import tensorflow as tf
import math
from tensorflow.contrib.layers import fully_connected


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
    with vs.variable_scope("embedded_sequence") as scope:
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
                 inputs,
                 cell,
                 beam_size,
                 beam_search,
                 do_attention,
                 embedding_size,
                 share_embedding_with_encoder,
                 output_projection):
        """
        Simple decoder wrapper
        :param name:
        :param num_symbols:
        :param inputs:
        :param cell:
        :param beam_size:
        :param beam_search:
        :param do_attention:
        :param embedding_size:
        :param share_embedding_with_encoder:
        :param output_projection:
        """
        self.name = name
        self.cell = cell  # TODO: work only with GRU cell because they contains one state vector and I admit it in the attention function
        self.beam_size = beam_size
        self.decoder_inputs = inputs
        self.nb_symbols = num_symbols
        self.beam_search = beam_search
        self.do_attention = do_attention
        self.embedding_size = embedding_size
        self.output_projection = output_projection
        self.share_embedding_with_encoder = share_embedding_with_encoder

        # Init values, will be filled when calling seq2seq
        self.output = None
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
            # log_beam_probs[-1] is of shape (batch_size x beam_size and it contains the probability
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


def beam_attention_decoder(decoder_inputs,
                           attention_states,
                           cell,
                           output_size,
                           loop_function,
                           initial_state,
                           beam_size):
    with vs.variable_scope("attention_decoder"):
        # Size of the batch
        batch_size = tf.shape(decoder_inputs[0])[0]

        # TODO(3): extend to multiple attention vectors
        # Length of the attention vector
        attn_length = attention_states.get_shape()[1].value

        # Size of the attention vectors (output size of the RNN encoder)
        attn_size = attention_states.get_shape()[2].value

        # Reshape for future convolution
        hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
        hidden = tf.tile(hidden, [beam_size, 1, 1, 1])

        # Convolution weights
        k = vs.get_variable("attnW_0", [1, 1, attn_size, attn_size])
        hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")

        # Vector v from the attention formula https://arxiv.org/pdf/1412.7449.pdf
        v = vs.get_variable("attnV_0", [attn_size])

        # State of the encoder (a tuple if num_layers > 1, else a Tensor)
        state = initial_state

        # Retrieve the number of layers
        if isinstance(state, tuple):
            num_layers = len(state)
        else:
            num_layers = 1

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            with vs.variable_scope("attention_0"):
                y = fully_connected(query, attn_size, activation_fn=None)

                y = array_ops.reshape(y, [-1, 1, 1, attn_size])
                # Attention mask is a softmax of v^T * tanh(...).
                features = hidden_features
                s = math_ops.reduce_sum(
                    v * math_ops.tanh(features + y), [2, 3])
                a = nn_ops.softmax(s)
                # Now calculate the attention-weighted vector d.
                d = math_ops.reduce_sum(
                    array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                    [1, 2])
                # for c in range(ct):
                return array_ops.reshape(d, [-1, attn_size])

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])

        # Init attention with zeros for the first decoder input (No attention)
        attns = array_ops.zeros(
            batch_attn_size, dtype=tf.float32)

        log_beam_probs, beam_path, beam_symbols = [], [], []

        for i, inp in enumerate(decoder_inputs):
            # Iterate over all inputs
            if i > 0:
                vs.get_variable_scope().reuse_variables()

            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None:
                with vs.variable_scope("loop_function", reuse=True):
                    # Required that prev has already been computed at least one (i >= 1)
                    if prev is not None:
                        inp = loop_function(prev, i, log_beam_probs, beam_path, beam_symbols)

            # Retrieve the length of the embedding input
            input_size = inp.get_shape().as_list()[1]

            # Linear combination with bias without activation function
            x = tf.squeeze(fully_connected(tf.concat([inp, attns], axis=1), input_size, activation_fn=None))

            # Compute one pass given the vector (decoder_input[i] + att[i]), and the state of cell[i-1]
            cell_output, state = cell(x, state)

            def get_last_state(state):
                if isinstance(state, tuple):
                    return state[num_layers - 1]
                else:
                    return state

            # Run the attention mechanism.
            if i > 0:
                attns = attention(get_last_state(state))

            with vs.variable_scope("attnOutputProjection"):
                # Linear combination between the cell outputs and the attention mechanism
                output = tf.squeeze(
                    fully_connected(tf.concat([cell_output, attns], axis=1), output_size, activation_fn=None))

            if loop_function is not None:
                # Usefull only for attention mechanism
                prev = output
            if i == 0:
                # If num_layers > 1, then replicate every state at each layer so that batch_size become batch_size x beam_size
                if isinstance(state, tuple):
                    state = [tf.tile(state_layer, [beam_size, 1]) for state_layer in state]
                else:
                    # Replicate only for the unique cell
                    state = tf.tile(state, [beam_size, 1])

                with vs.variable_scope(vs.get_variable_scope(), reuse=True):
                    # Compute the attention mechanism given the last state
                    attns = attention(get_last_state(state))

            # Save outputs
            outputs.append(tf.argmax(output, dimension=1))

    # Return outputs, current decoder state, beam_path, beam_symbol
    return outputs, state, tf.reshape(tf.concat(beam_path, 0), [-1, beam_size]), tf.reshape(tf.concat(beam_symbols, 0),
                                                                                            [-1, beam_size])


def rnn_decoder(decoder_inputs,
                embeddings,
                attention_states,
                initial_state,
                nb_symbols,
                embedding_size,
                cell,
                beam_search=True,
                beam_size=10):
    with vs.variable_scope("embedding_attention_decoder"):
        if beam_search:
            loop_function = _extract_beam_search(embeddings,
                                                 beam_size,
                                                 nb_symbols,
                                                 embedding_size)
        else:
            loop_function = _extract_argmax_and_embed(embeddings)

        if beam_search:
            return beam_attention_decoder(
                decoder_inputs=decoder_inputs,
                attention_states=attention_states,
                loop_function=loop_function,
                beam_size=beam_size,
                initial_state=initial_state,
                cell=cell,
                output_size=cell.output_size)
        elif attention_states is not None:
            return tf.contrib.legacy_seq2seq.attention_decoder(
                decoder_inputs,
                initial_state,
                attention_states,
                cell,
                output_size=None,
                num_heads=1,
                loop_function=loop_function,
                initial_state_attention=False)
        else:
            return tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs,
                                                         initial_state,
                                                         cell,
                                                         loop_function)


def seq2seq(encoder_inputs,
            num_encoder_symbols,
            embedding_size_encoder,
            encoder_cell_fw,
            decoders
            ):
    with vs.variable_scope("my_seq2seq"):

        encoder_inputs, encoder_embedding = embedded_sequence(encoder_inputs,
                                                              num_encoder_symbols,
                                                              embedding_size_encoder)
        # Compute the length of the encoder
        length = length_sequence(encoder_inputs)

        do_attention = any([d.do_attention for d in decoders])
        attention_states = None
        # If the encoder is bidirectional

        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
            cell=encoder_cell_fw,
            inputs=encoder_inputs,
            sequence_length=length,
            dtype=tf.float32,
            swap_memory=True)

        if do_attention:
            attention_states = encoder_outputs
            # print(gs(attention_states))

        # Decoder
        for decoder in decoders:
            name = decoder.name
            with vs.variable_scope("one2many_decoder_" + str(
                    name)) as _:

                decoder_attention_states = attention_states if decoder.do_attention else None
                share_embedding_with_encoder = False

                if decoder.share_embedding_with_encoder:
                    assert (decoder.nb_symbols == num_encoder_symbols) \
                           and (decoder.embedding_size == embedding_size_encoder), \
                        "If share embedding is True, then size " \
                        "of the vocabulary and embedding size should " \
                        "be the same between encoder and decoder"
                    share_embedding_with_encoder = True

                # decoder_inputs should be of size [batch_size, sequence_length, decoder.embedding_size)
                decoder_inputs, embeddings = embedded_sequence(decoder.decoder_inputs,
                                                               decoder.nb_symbols,
                                                               decoder.embedding_size,
                                                               embedding=encoder_embedding if share_embedding_with_encoder else None)

                decoder_inputs = tf.unstack(decoder_inputs)
                result = rnn_decoder(decoder_inputs=decoder_inputs,
                                     attention_states=decoder_attention_states,
                                     initial_state=encoder_states,
                                     embeddings=embeddings,
                                     beam_search=decoder.beam_search,
                                     beam_size=decoder.beam_size,
                                     embedding_size=decoder.embedding_size,
                                     cell=decoder.cell,
                                     nb_symbols=decoder.nb_symbols)
                decoder.outputs = result[0]
                if decoder.beam_search:
                    decoder.beam_path = result[2]
                    decoder.beam_symbol = result[3]
