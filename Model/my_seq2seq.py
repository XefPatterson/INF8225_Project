from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.layers import fully_connected
import copy
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
import math

# TODO RIGHT CODE IS A TOTAL SHITTY MESS, DONT SAY YOU WERE NOT AWARE :')

"""

UTILITIES

"""


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
            if isinstance(inputs, list):
                inputs = tf.stack(inputs, 1)

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
                 max_decoder_sequence_length,
                 clip_gradient=False):
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
        # Clip gradients TODO (not implemented)
        self.clip_gradient = clip_gradient
        # Init values, will be filled when calling seq2seq
        self.outputs = None
        self.train_fn = None
        self.loss = None

        # Current step
        self.global_step = tf.Variable(0, trainable=False)
        # Tensor bool
        # (TODO: should also be used for dropout, dropout is currently set in the test_main with a DropoutWrapper)
        # but it is currently used to set the beam search (Beam search only works at testing time)
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # Placeholder, maximum length of every sequence in the batch! Must be feed
        self.max_length_decoder_in_batch = tf.placeholder(tf.int32, name="max_length_in_batch")

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

        if beam_size != 1:
            # Init values. Will be filled when calling seq2seq
            self.beam_path = None
            self.beam_symbol = None

        # Check that beam_size is set to 1 if there is no beam model (beam_size is used in loss function)
        if not self.beam_search:
            self.beam_size = 1


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


def sequence_loss(logits,
                  targets,
                  weights,
                  beam_size,
                  average_across_timesteps=True,
                  average_accross_batch=True,
                  sum_accross_batch=False):
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with ops.name_scope(None, "sequence_loss_by_example",
                        logits + targets + weights):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            logit = tf.gather(logit, tf.range(tf.shape(logit)[0] // beam_size))

            target = array_ops.reshape(target, [-1])
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                labels=target, logits=logit)
            log_perp_list.append(crossent * weight)
        cost = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            cost /= total_size

    if sum_accross_batch:
        # Sum the loss across batches
        cost = math_ops.reduce_sum(cost)
    if average_accross_batch:
        # Average across batches
        cost /= math_ops.cast(array_ops.shape(targets[0])[0], cost.dtype)
    return cost


def loss_per_decoder(decoders, optimizer=tf.train.AdamOptimizer()):
    all_variables = tf.trainable_variables()

    for decoder in decoders:
        decoder.loss = sequence_loss(logits=decoder.outputs,
                                     targets=decoder.targets,
                                     weights=decoder.target_weights,
                                     beam_size=decoder.beam_size)

        decoder_variables = [v for v in all_variables if "one2many_decoder_" + str(decoder.name) in v.name]
        # Retrieve encoder parameters
        if decoder.train_encoder_weight:
            decoder_variables.extend(
                [v for v in all_variables if "encoder" + str(decoder.name) in v.name])

        # Compute gradients
        gradients = tf.gradients(decoder.loss, decoder_variables, aggregation_method=2)

        # Training function
        decoder.train_fn = optimizer.apply_gradients(zip(gradients, decoder_variables),
                                                     global_step=decoder.global_step)


# TODO make sure computation is not dependant on the size o the batch
# TODO model with buckets maybe
# TODO nice attention display
# TODO nice beam search display

"""


PYTHON CODE FROM ORIGINAL SEQ2SEQ ALMOST NOT MODIFIED

"""


# Add save attention as parameter
def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       save_attention,
                       # If set to true, then the third output of a call to seq2seq should be an attention
                       per_example_loss=False,
                       name=None):
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                         "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                         "bucket (%d)." % (len(weights), buckets[-1][1]))

    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    losses = []
    outputs = []
    # If seq2seq model does not do attention, then attentions will stay an empty list
    # TODO: right now in the main model, we can call embedding_rnn which return a list of output and states (size 2),
    # TODO but in main, I defined it a lambda x, y, z, so it should raise an error if plugged
    attentions = []

    with ops.name_scope(name, "model_with_buckets", all_inputs):
        for j, bucket in enumerate(buckets):
            with vs.variable_scope(
                    vs.get_variable_scope(), reuse=True if j > 0 else None):
                rest = seq2seq(encoder_inputs[:bucket[0]],
                               decoder_inputs[:bucket[1]])
                outputs.append(rest[0])

                if save_attention:
                    if len(rest) < 2:
                        raise ValueError(
                            "Set save_attention to True, but the seq2seq model does not support attention saving")
                    attentions.append(rest[2])

                if per_example_loss:
                    losses.append(
                        sequence_loss(
                            outputs[-1],
                            targets[:bucket[1]],
                            weights[:bucket[1]],
                            sum_accross_batch=False))
                else:
                    losses.append(
                        sequence_loss(
                            outputs[-1],
                            targets[:bucket[1]],
                            weights[:bucket[1]],
                            sum_accross_batch=True))

    return outputs, losses, attentions


# PYTHON CODE FROM ORIGINAL SEQ2SEQ ALMOST NOT MODIFIED (ADD ALL_ATTENTION LIST)
def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
    linear = core_rnn_cell_impl._linear
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with vs.variable_scope(
                    scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states,
                                   [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = vs.get_variable("AttnW_%d" % a,
                                [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                vs.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = initial_state

        # RETURN ALL_ATTENTIONS
        all_attentions = []

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in range(num_heads):
                with vs.variable_scope("Attention_%d" % a):
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                            [2, 3])
                    a = nn_ops.softmax(s)
                    all_attentions.append(a)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(
                batch_attn_size, dtype=dtype) for _ in range(num_heads)
            ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)

        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                vs.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with vs.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with vs.variable_scope(
                        vs.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with vs.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, state, all_attentions


# PYTHON CODE FROM ORIGINAL SEQ2SEQ NOT MODIFIED
def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with vs.variable_scope(
                    scope or "embedding_attention_decoder", dtype=dtype) as scope:

        embedding = vs.get_variable("embedding",
                                    [num_symbols, embedding_size])
        loop_function = tf.contrib.legacy_seq2seq_extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
            ]

        return attention_decoder(
            emb_inp,
            initial_state,
            attention_states,
            cell,
            output_size=output_size,
            num_heads=num_heads,
            loop_function=loop_function,
            initial_state_attention=initial_state_attention)


# PYTHON CODE FROM ORIGINAL SEQ2SEQ ALMOST NOT MODIFIED (recupere attention)
def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    with vs.variable_scope(
                    scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
    # Encoder.
    encoder_cell = copy.deepcopy(cell)
    encoder_cell = core_rnn_cell.EmbeddingWrapper(
        encoder_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    encoder_outputs, encoder_state = core_rnn.static_rnn(
        encoder_cell, encoder_inputs, dtype=dtype)
    top_states = [
        array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
        ]
    attention_states = array_ops.concat(top_states, 1)
    # Decoder.
    output_size = None
    if output_projection is None:
        cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
        output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
        return embedding_attention_decoder(
            decoder_inputs,
            encoder_state,
            attention_states,
            cell,
            num_decoder_symbols,
            embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous,
            initial_state_attention=initial_state_attention)
        # If feed_previous is a Tensor, we construct 2 graphs and use cond.

    def decoder(feed_previous_bool):
        reuse = None if feed_previous_bool else True
        with vs.variable_scope(
                vs.get_variable_scope(), reuse=reuse):
            outputs, state, attention = embedding_attention_decoder(
                decoder_inputs,
                encoder_state,
                attention_states,
                cell,
                num_decoder_symbols,
                embedding_size,
                num_heads=num_heads,
                output_size=output_size,
                output_projection=output_projection,
                feed_previous=feed_previous_bool,
                update_embedding_for_previous=False,
                initial_state_attention=initial_state_attention)
            state_list = [state]
            if nest.is_sequence(state):
                state_list = nest.flatten(state)
            return outputs + state_list, attention

    outputs_and_state, attention = control_flow_ops.cond(feed_previous,
                                                         lambda: decoder(True),
                                                         lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
        state = nest.pack_sequence_as(
            structure=encoder_state, flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state, attention


"""

BEAM IMPLEMENTATION

ATTENTION + BEAM
"""


def beam_attention_decoder(decoder_inputs,
                           attention_states,
                           cell,
                           output_size,
                           max_length_encoder_in_batch,
                           loop_function,
                           initial_state,
                           beam_size,
                           is_training):
    with vs.variable_scope("attention_decoder"):
        # Size of the batch
        batch_size = tf.shape(decoder_inputs[0])[0]
        # Size of the attention vectors (output size of the RNN encoder)
        attn_size = attention_states.get_shape()[2].value

        # Reshape for future convolution
        hidden = array_ops.reshape(attention_states, [-1, max_length_encoder_in_batch, 1, attn_size])
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
                    array_ops.reshape(a, [-1, max_length_encoder_in_batch, 1, 1]) * hidden,
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
            if i == 0:
                state = [tf.tile(state[0], (beam_size, 1))]
            inp = tf.tile(inp, (beam_size, 1))
            # Iterate over all inputs
            if i > 0:
                vs.get_variable_scope().reuse_variables()

            # If loop_function is set, we use it instead of decoder_inputs.
            with vs.variable_scope("loop_function", reuse=True):
                # Required that prev has already been computed at least one (i >= 1)
                if prev is not None:
                    inp = tf.cond(is_training, lambda: inp,
                                  lambda: loop_function(prev, i, log_beam_probs, beam_path, beam_symbols))

            # Retrieve the length of the decoder embedding input
            input_size = inp.get_shape().as_list()[1]

            # Linear combination with bias without activation function
            x = fully_connected(tf.concat([inp, attns], axis=1), input_size, activation_fn=None)

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

                # Useful only for attention mechanism
            prev = output
            if i == 0:
                # If num_layers > 1, then replicate every state at each layer so that batch_size become batch_size x beam_size
                if isinstance(state, tuple):

                    state = tf.cond(is_training,
                                    lambda: state,
                                    lambda: [tf.tile(state_layer, [beam_size, 1]) for state_layer in state])
                    if isinstance(state, tf.Tensor):
                        state = [state]
                else:
                    state = tf.cond(is_training,
                                    lambda: state,
                                    lambda: tf.tile(state, [beam_size, 1]))
                with vs.variable_scope(vs.get_variable_scope(), reuse=True):
                    # Compute the attention mechanism given the last state
                    attns = attention(get_last_state(state))

            # Save outputs
            outputs.append(output)

    # Return outputs, current decoder state, beam_path, beam_symbol
    return outputs, state, beam_path, beam_symbols


"""

ONLY BEAM

"""


# SEEMS TO WORK
def beam_rnn_decoder(decoder_inputs,
                     initial_state,
                     cell,
                     output_size,
                     beam_size,
                     loop_function,
                     is_training,
                     scope=None):
    with vs.variable_scope(scope or "beam_rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        log_beam_probs, beam_path, beam_symbols = [], [], []
        for i, inp in enumerate(decoder_inputs):
            if i == 0:
                state = [tf.tile(state[0], (beam_size, 1))]
            inp = tf.tile(inp, (beam_size, 1))

            if i > 0:
                vs.get_variable_scope().reuse_variables()
            if prev is not None:
                with vs.variable_scope("loop_function", reuse=True):
                    inp = tf.cond(is_training, lambda: inp,
                                  lambda: loop_function(prev, i, log_beam_probs, beam_path, beam_symbols))

            output, state = cell(inp, state)
            output = fully_connected(output, output_size)

            outputs.append(output)
            prev = output

    return outputs, state, beam_path, beam_symbols


"""

RNN DECODER

"""


def rnn_decoder(decoder_inputs,
                embeddings,
                attention_states,
                initial_state,
                nb_symbols,
                embedding_size,
                cell,
                beam_search,
                max_length_encoder_in_batch,
                beam_size,
                is_training):
    with vs.variable_scope("embedding_attention_decoder"):
        if beam_search:
            loop_function = _extract_beam_search(embeddings,
                                                 beam_size,
                                                 nb_symbols,
                                                 embedding_size)
        else:
            loop_function = _extract_argmax_and_embed(embeddings)

        if beam_search and (attention_states is not None):
            # BEAM SEARCH + ATTENTION STATES
            return beam_attention_decoder(
                decoder_inputs=decoder_inputs,
                attention_states=attention_states,
                loop_function=loop_function,
                beam_size=beam_size,
                initial_state=initial_state,
                max_length_encoder_in_batch=max_length_encoder_in_batch,
                cell=cell,
                output_size=cell.output_size,
                is_training=is_training)
        elif attention_states is not None:
            # ONLY ATTENTION
            return attention_decoder(
                decoder_inputs,
                initial_state,
                attention_states,
                cell,
                output_size=None,
                num_heads=1,
                loop_function=loop_function,
                initial_state_attention=False)
        elif beam_search:
            # ONLY BEAM SEARCH
            return beam_rnn_decoder(decoder_inputs=decoder_inputs,
                                    initial_state=initial_state,
                                    cell=cell,
                                    beam_size=beam_size,
                                    output_size=cell.output_size,
                                    loop_function=loop_function,
                                    is_training=is_training)
        else:
            # NOTHING FANCY
            return tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs,
                                                         initial_state,
                                                         cell,
                                                         loop_function)


def myseq2seq(encoder_inputs,
              max_length_encoder_in_batch,
              num_encoder_symbols,
              embedding_size_encoder,
              encoder_cell_fw,
              decoders
              ):
    with vs.variable_scope("my_seq2seq"):
        with vs.variable_scope("encoder"):
            # encoder_inputs has a shape [batch_size, max_encode_sequence_length, encoder_emebdding_size]
            encoder_inputs, encoder_embedding = embedded_sequence(encoder_inputs,
                                                                  num_encoder_symbols,
                                                                  embedding_size_encoder)
            encoder_inputs = tf.transpose(encoder_inputs, [1, 0, 2])

            encoder_inputs = tf.gather(encoder_inputs, tf.range(max_length_encoder_in_batch))
            encoder_inputs = tf.transpose(encoder_inputs, [1, 0, 2])

            # Compute the length of the encoder
            length = length_sequence(encoder_inputs)

            do_attention = any([d.do_attention for d in decoders])
            attention_states = None

            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
                cell=encoder_cell_fw,
                inputs=encoder_inputs,
                sequence_length=length,
                dtype=tf.float32,
                swap_memory=True)

            if do_attention:
                attention_states = encoder_outputs

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
                decoder_inputs = tf.stack(decoder.decoder_inputs, 0)
                decoder_inputs = tf.transpose(decoder_inputs)
                decoder_inputs, embeddings = embedded_sequence(decoder_inputs,
                                                               decoder.nb_symbols,
                                                               decoder.embedding_size,
                                                               embedding=encoder_embedding if share_embedding_with_encoder else None)

                decoder_inputs = tf.unstack(decoder_inputs, axis=1)
                # print(len(decoder_inputs))

                # Contains output, states, beam_path, beam_symbols
                result = rnn_decoder(decoder_inputs=decoder_inputs,
                                     attention_states=decoder_attention_states,
                                     initial_state=encoder_states,
                                     embeddings=embeddings,
                                     beam_search=decoder.beam_search,
                                     beam_size=decoder.beam_size,
                                     embedding_size=decoder.embedding_size,
                                     cell=decoder.cell,
                                     nb_symbols=decoder.nb_symbols,
                                     is_training=decoder.is_training,
                                     max_length_encoder_in_batch=max_length_encoder_in_batch)

                decoder.outputs = result[0]
                if decoder.beam_search:
                    decoder.beam_path = result[2]
                    decoder.beam_symbol = result[3]
