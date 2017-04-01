from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
import tensorflow as tf
from tf_utils import length_sequence, embedded_sequence, _extract_beam_search, _extract_argmax_and_embed


def sequence_loss(logits,
                  targets,
                  weights,
                  beam_size,
                  average_across_timesteps=True,
                  sum_accross_batch=False,
                  average_accross_batch=True):
    """
    Compute sequence length loss.

    Compute softmax on logits, then cross_entropy.
    Average at each timestep for every example with average_across_timesteps
    Sum all batch example loss with sum_accross_batch
    Average all batch example loss with average_accross_batch

    This function is not relevant in the case where beam_search is used and it's not training time
    This function is relevant in the case where beam_search is used and it's training time, and beam_size parameter is set!

    :param logits: List of size max_decoder_length of tf.Tensor ((batch_size x beam_size) x vocab_size)
        Contains all outputs during the decoding scheme
    :param targets: List of size max_decoder_length of tf.Tensor (batch_size x 1)
        All targets index in the dictionnary
    :param weights: List of size max_decoder_length of tf.Tensor (batch_size x 1)
        0 or 1 to mask the loss
    :param beam_size: Size of the beam search
        During training time, there is no beam search, however the graph is not dynamic, so there must as much
        input during training and testing. Hence decoder_inputs are tilled beam_size.
    :param average_across_timesteps: Boolean (default: True)
        Average accross all timesteps, preserving respective length of different batch example
    :param average_accross_batch: Boolean (default: True)
        Average accross batch examples
    :param sum_accross_batch: Boolean, (default: False)
        Sum
    :return:
    """
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
    """
    Method to compute all loss for every decoders.
    It computes the loss of the decoder (cross_entropy)
    Then it retrieve weights that are trained on (if a decoder has train_encoder_weight set to True, then all encoder weights
    are also retrieved
    Then it adds a train_fn to the decoder, which corresponds to the training function to call in a tf.Session
    :param decoders: A list of Decoder
        Every decoder used in the model.
        Every decoder should have its parameter outputs, targets, target_weights defined
    :param optimizer:
    :return:
    """
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
    """
    Original code from Tensorflow. I just add a list to retrieve attention values.
    :param decoder_inputs:
    :param initial_state:
    :param attention_states:
    :param cell:
    :param output_size:
    :param num_heads:
    :param loop_function:
    :param dtype:
    :param scope:
    :param initial_state_attention:
    :return:
    """
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


def beam_attention_decoder(decoder_inputs,
                           attention_states,
                           cell,
                           output_size,
                           max_length_encoder_in_batch,
                           loop_function,
                           initial_state,
                           beam_size,
                           is_training):
    """
    Implementation of beam search with attention mechanism.
    decoder_inputs is the input of the decoder
    attention_states is a matrix of tensor containing all vectors used for computing the attention.


    :param decoder_inputs: List of size max_decoder_length of tf.Tensor (batch_size x embedding_size_decoder)
        Decoder inputs
    :param attention_states: 3D tf.Tensor of size (batch_size x max_length_encoder_in_batch x encoder.cell.output_size)

    :param cell: Tensorflow RNN cell
        YOLO
    :param output_size: Tensor
        Dimension of the output of the RNN cell. If the hidden vector has a dimension larger than output_size, then a
        one layer feedforward neural network reshape it to output a vector of size output_size
    :param max_length_encoder_in_batch: TensorShape
        Number of vector for the attention mechanism
    :param loop_function: Python Function TODO: refactore because this parameter is useless
        Should not be anything else than _extract_beam_search
    :param initial_state: Tensor of shape (batch_size, cell.state_size)
        Initial state of the cell at timestep 0. Should not be None
    :param beam_size: Python Scalar
        Size of the beam search
    :param is_training: Tensor
        Variable to represent if we are training or testing. During training, the decoder_inputs first dimension is resized
        to (batch_size x beam_size). It is not efficient, but it was the only way to allow beam_search at test time.
    :return:
    """
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

        # RETURN ALL_ATTENTIONS
        all_attentions = []

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
                all_attentions.append(a)

                # Now calculate the attention-weighted vector d.
                d = math_ops.reduce_sum(
                    array_ops.reshape(a, [-1, max_length_encoder_in_batch, 1, 1]) * hidden,
                    [1, 2])
                # for c in range(ct):
                return array_ops.reshape(d, [-1, attn_size])

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size * beam_size, attn_size])

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
                with vs.variable_scope(vs.get_variable_scope(), reuse=True):
                    # Compute the attention mechanism given the last state
                    attns = attention(get_last_state(state))

            # Save outputs
            outputs.append(output)

    # Return outputs, current decoder state, beam_path, beam_symbol, log_beam_probs, all_attentions
    return outputs, state, beam_path, beam_symbols, log_beam_probs, all_attentions


def beam_rnn_decoder(decoder_inputs,
                     initial_state,
                     cell,
                     output_size,
                     beam_size,
                     loop_function,
                     is_training,
                     scope=None):
    """

    :param decoder_inputs: List of size max_decoder_length of tf.Tensor (batch_size x embedding_size_decoder)
        Decoder inputs
    :param initial_state: Tensor of shape (batch_size, cell.state_size)
        Initial state of the cell at timestep 0. Should not be None
    :param cell: Tensorflow RNN cell
        YOLO
    :param output_size: Tensor
        Dimension of the output of the RNN cell. If the hidden vector has a dimension larger than output_size, then a
        one layer feedforward neural network reshape it to output a vector of size output_size
    :param loop_function: Python Function
        Should not be anything else than _extract_beam_search
    :param beam_size: Python Scalar
        Size of the beam search
    :param is_training: Tensor
        Variable to represent if we are training or testing. During training, the decoder_inputs first dimension is resized
        to (batch_size x beam_size). It is not efficient, but it was the only way to allow beam_search at test time.
    :param scope: TODO Don't know if it useful
    :return:
    """
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

    return outputs, state, beam_path, beam_symbols, log_beam_probs


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
    """
    Given the function parameters, it select the model to use
    TODO: refactor
    :param decoder_inputs:
    :param embeddings:
    :param attention_states
    :param initial_state:
    :param nb_symbols:
    :param embedding_size:
    :param cell:
    :param beam_search:
    :param max_length_encoder_in_batch:
    :param beam_size:
    :param is_training:
    :return:
    """
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


def seq2seq_builder(encoder_inputs,
                    max_length_encoder_in_batch,
                    num_encoder_symbols,
                    embedding_size_encoder,
                    encoder_cell_fw,
                    decoders
                    ):
    """
    Seq2seq model implemented.
    This is the main function. It is responsible to creating a encoder and a decoder
    :param encoder_inputs: A list of size max_encoder_length of tf.Tensor of size (batch_size x 1)
        The encoder inputs
    :param max_length_encoder_in_batch: A tf.Tensor
        Given that the encoder inputs placeholder can hold the maximum length in all batches, there is no
        need to keep all zeros padded inputs. max_length_encoder_in_batch remove at the beginning all padded tf.Tensor
        which are beyond the longest sequence in the batch
    :param num_encoder_symbols: Python scalar
        ~ Size of the encoder vocabulary
    :param embedding_size_encoder: Python scalar
        Size of the embeddings
    :param encoder_cell_fw: a Tensorflow RNN cell
        YO
    :param decoders: A list of Decoders

    :return: Nothing,
        All decoders objects are responsible to keep their outputs, losses, training function, beam_path, and attentions
    """
    with vs.variable_scope("my_seq2seq"):
        with vs.variable_scope("encoder"):
            # encoder_inputs has a shape [batch_size, max_encode_sequence_length, encoder_embedding_size]
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
                    decoder.log_beam_probs = result[4]
                    if decoder.do_attention:
                        decoder.all_attentions = result[5]
                elif decoder.do_attention:
                    decoder.all_attentions = result[2]
