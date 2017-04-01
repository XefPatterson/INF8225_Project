import tensorflow as tf


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

        # Placeholder to retrieve values after a session run an iteration
        if beam_size != 1:
            self.beam_path = None
            self.beam_symbol = None
            self.log_beam_probs = None

        # Placeholder to retrieve values after a session run an iteration
        if do_attention:
            self.all_attentions = None

        # Check that beam_size is set to 1 if there is no beam model (beam_size is used in loss function)
        if not self.beam_search:
            self.beam_size = 1
