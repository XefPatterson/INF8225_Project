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
                      reuse_scope=False):
    with vs.variable_scope(scope="embedded_sequence") as scope:
        if reuse_scope:
            scope.reuse_variables()
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


class decoder():
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
                 output_projection,
                 loop_function):
        self.name = name
        self.cell = cell
        self.beam_size = beam_size
        self.decoder_inputs = inputs
        self.nb_symbols = num_symbols
        self.beam_search = beam_search
        self.do_attention = do_attention
        self.loop_function = loop_function
        self.embedding_size = embedding_size
        self.output_projection = output_projection
        self.share_embedding_with_encoder = share_embedding_with_encoder

        if output_projection:
            self.cell = tf.contrib.rnn.OutputProjectionWrapper(self.cell, self.nb_symbols)


def _extract_beam_search(embedding, beam_size, num_symbols, embedding_size):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
      embedding: embedding tensor for symbols.
      output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.

    Returns:
      A loop function.
    """

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


def embedding_attention_decoder(decoder,
                                embeddings,
                                attention_states):
    with vs.variable_scope("embedding_attention_decoder"):
        if decoder.beam_search:
            loop_function = _extract_beam_search(
                embedding, beam_size, num_symbols, embedding_size, output_projection,
                update_embedding_for_previous)
        else:
            loop_function = _extract_argmax_and_embed(
                embedding, output_projection,
                update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        if beam_search:
            return beam_attention_decoder(
                emb_inp, initial_state, attention_states, cell, output_size=output_size,
                num_heads=num_heads, loop_function=loop_function,
                initial_state_attention=initial_state_attention, output_projection=output_projection,
                beam_size=beam_size)
        else:
            return attention_decoder(
                emb_inp, initial_state, attention_states, cell, output_size=output_size,
                num_heads=num_heads, loop_function=loop_function,
                initial_state_attention=initial_state_attention)


def seq2seq(encoder_inputs,
            share_embedding_encoder_decoder_dict,
            embedding_size_decoder_dict,
            num_encoder_symbols,
            embedding_size_encoder,
            encoder_cell_fw,
            encoder_cell_bw,
            encoder_is_bidirectionnal,
            decoders
            ):
    with vs.variable_scope(scope="my_seq2seq"):
        encoder_inputs, _ = embedded_sequence(encoder_inputs,
                                              num_encoder_symbols,
                                              embedding_size_encoder)
        # Compute the length of the encoder
        length = length_sequence(encoder_inputs)

        do_attention = any([d.do_attention for d in decoders])
        attention_states = None

        # If the encoder is bidirectionnal
        if encoder_is_bidirectionnal:
            encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell_fw,
                cell_bw=encoder_cell_bw,
                inputs=encoder_inputs,
                sequence_length=length,
                dtype=tf.float32,
                swap_memory=True)  # seems to be faster
            # Transform encoder_outputs for attention
            if do_attention:

                top_states_fw = [
                    array_ops.reshape(e, [-1, 1, encoder_cell_fw.output_size])
                    for e in encoder_outputs[0]
                    ]
                attention_states = [(tf.concat(top_states_fw, 1))]

                # If forward cell and backward cell have the same output_size
                if encoder_cell_fw.output_size == encoder_cell_bw.output_size:
                    top_states_bw = [
                        array_ops.reshape(e, [-1, 1, encoder_cell_bw.output_size])
                        for e in encoder_outputs[1]
                        ]
                    attention_states.append(tf.concat(top_states_bw, 1))

                else:
                    # Need to transform one of them?
                    pass
        else:
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
                cell=encoder_cell_fw,
                inputs=encoder_inputs,
                sequence_length=length,
                dtype=tf.float32,
                swap_memory=True)  # seems to be faster

            if do_attention:
                top_states = [
                    tf.reshape(e, [-1, 1, encoder_cell_fw.output_size])
                    for e in encoder_outputs
                    ]
                attention_states = [tf.concat(top_states, 1)]

        # Decoder
        for decoder in decoders:
            name = decoder.name
            with vs.variable_scope("one2many_decoder_" + str(
                    name)) as scope:

                decoder_attention_states = attention_states if decoder.do_attention else None
                share_embedding_with_encoder = False

                if decoder.share_embedding_with_encoder:
                    assert (decoder.nb_symbols == num_encoder_symbols) \
                           and (decoder.embedding_size == num_encoder_symbols), \
                        "If share embedding is True, then size " \
                        "of the vocabulary and embedding size should " \
                        "be the same between encoder and decoder"
                    share_embedding_with_encoder = True

                decoder.decoder_inputs, embeddings = embedded_sequence(decoder.decoder_inputs,
                                                                       decoder.nb_symbols,
                                                                       decoder.embedding_size,
                                                                       reuse_scope=share_embedding_with_encoder)

                embedding_attention_decoder(decoder=decoder,
                                            attention_states=decoder_attention_states,
                                            embeddings=embeddings)
