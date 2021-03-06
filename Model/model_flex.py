import numpy as np
import tensorflow as tf
from queues import create_queues_for_bucket
from termcolor import cprint

FLAGS = None


class Seq2Seq(object):
    def __init__(self,
                 buckets,
                 forward_only=False):
        """
        Seq2Seq model
        :param buckets: List of pairs
            Each pair correspond to (max_size_in_bucket_for_encoder_sentence, max_size_in_bucket_for_decoder_sentence)
        :param forward_only: Boolean (False)
            Whether to update the model, or only predict.
            Now it only supports False, but it should not be a big deal
        """

        self.max_gradient_norm = FLAGS.max_gradient_norm
        self.learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_decay_op = tf.train.exponential_decay(FLAGS.learning_rate,
                                                                 self.global_step,
                                                                 FLAGS.decay_learning_rate_step,
                                                                 FLAGS.learning_rate_decay_factor,
                                                                 staircase=True)
        self.buckets = buckets

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.targets = []
        self.target_weights = []

        for i in range(self.buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in range(self.buckets[-1][1]):
            self.targets.append(tf.placeholder(tf.int32, shape=[None],
                                               name="decoder{0}".format(i)))

        # decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        self.decoder_inputs = [tf.zeros_like(self.targets[0], dtype=tf.int64, name='GO')] + self.targets[:-1]

        #Binary mask useful for padded sequences.
        self.target_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.targets]

        self.gradient_norms = []
        self.updates = []

        self.forward_only = forward_only

        # Which bucket to extract examples
        self.bucket_id = tf.placeholder_with_default(0, [], name="bucket_id")

    def _build_queues(self):
        """
        Build the queues
        :return:
        """
        self.queues, self.op_starting_queue = create_queues_for_bucket(FLAGS.batch_size, "train", self.buckets)
        q = tf.QueueBase.from_list(self.bucket_id, self.queues)

        inputs = tf.squeeze(q.dequeue())
        self._questions = inputs[0]
        self._answers = inputs[1]

    def build(self):
        """
        Build the model
        :return:
        """
        cprint("[*] Building model (G)", color="yellow")
        single_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=FLAGS.keep_prob)
        if FLAGS.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * FLAGS.num_layers)
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=FLAGS.vocab_size,
                num_decoder_symbols=FLAGS.vocab_size,
                embedding_size=128,
                output_projection=None,
                feed_previous=do_decode)

        with tf.variable_scope("seq2seq") as scope:
            self.train_outputs, self.train_losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.target_weights,
                self.buckets,
                lambda x, y: seq2seq_f(x, y, False))

            scope.reuse_variables()

            self.test_outputs, self.test_losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.target_weights,
                self.buckets,
                lambda x, y: seq2seq_f(x, y, True))

        cprint("[*] Building model (D)", color="yellow")
        # Question encoder for D.
        # Should be similar to the one used for G.
        single_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
        disc_q_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=FLAGS.keep_prob)
        if FLAGS.num_layers > 1:
            disc_q_cell = tf.contrib.rnn.MultiRNNCell([single_cell] * FLAGS.num_layers)
        disc_q_outputs, disc_q_states = tf.nn.dynamic_rnn(
            cell=disc_q_cell,
            inputs=self.encoder_inputs, # TODO, we need a 2D tensor instead of list, should be this with Louis'code.
            sequence_length=question_length, #TODO use length
            dtype=tf.float32,
            swap_memory=True)

        # TODO: Combine Real and Fake answers into a single minibatch. Use DECODER max length.
        # TODO: Should we use vs.get_variable?
        # Another "classic" dynamic RNN, therefore, transfer LIST inputs into 2D tensor inputs.
        real_fake_answers = tf.placeholder(tf.int32, shape=[self.batch_size*2, self.max_decoder_sequence_length],
                       name="real_and_fake_answers")

        for t in range(self.max_decoder_sequence_length):
            real_fake_answers[:self.batch_size, t] = self.decoder_inputs[t]
            real_fake_answers[self.batch_size:, t] = self.decoder_outputs[0][t] #TODO use eventual G_decoder outputs

        # TODO: Produce target weights (binary mask) for this too!

        # TODO: get char/word embeddings of real+fake inputs - use Louis' function :
        # We assume d_embeddings.shape = (batch, T, embedding_size) - similar to dynamic RNN outputs.
        d_inputs, d_embeddings = embedded_sequence(real_fake_answers,
                                                      num_encoder_symbols,
                                                      embedding_size_encoder)

        # TODO: Get length of minibatch for dynamic RNN - use Louis's function:
        answer_length = length_sequence(d_inputs)

        # TODO: take last state or output question encoder ...
        # TODO: Do we need a call to tf.expand_dims to keep the time dimension (to further allow broadcast durin concat)?
        self.question_representation = disc_q_outputs[0][:, question_length-1, :] #shape = (batch, 1, cell_size)

        # TODO:  ... and double the values for the double batch (real - fake) ...
        self.question_representation = tf.concat(0, [question_representation, question_representation],
                                            name="question_representation" )
        # TODO: ... and concat to each embedding_t
        self.question_representation = tf.concat(2, [question_representation, d_embeddings],
                                            name="question_representation" )

        # TODO: Feed this to a "basic" stacked gru/lstm
        single_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
        disc_a_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=FLAGS.keep_prob)
        last_cell = tf.contrib.rnn.GRUCell(1)
        if FLAGS.num_layers > 1:
            disc_a_cell = tf.contrib.rnn.MultiRNNCell([single_cell] * FLAGS.num_layers + [last_cell])
        else:
            disc_a_cell = tf.contrib.rnn.MultiRNNCell([single_cell, last_cell])

        disc_a_outputs, disc_a_states = tf.nn.dynamic_rnn(
            cell=disc_a_cell,
            inputs=self.question_representation,
            sequence_length=answer_length,
            dtype=tf.float32,
            swap_memory=True)

        # TODO: use sigmoid to classify each timestep as real/fake.
        d_probs = tf.sigmoid(disc_a_outputs)

        # TODO: define optimization for D and for G
        d_loss = - tf.reduce_mean(tf.log(d_probs))
        g_rewards = 1.0 - d_probs[self.batch_size:]
        g_loss = tf.reduce_mean(tf.log(d_probs[self.batch_size:]))

        # Optimization :
        params = tf.trainable_variables()
        if not self.forward_only:
            opt = tf.train.AdamOptimizer(self.learning_rate)
            for b in range(len(self.buckets)):
                cprint("Constructing the forward pass for bucket {}".format(b))

                gradients = tf.gradients(self.train_losses[b], params, aggregation_method=2)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 self.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self._summary()
        cprint("[!] Model built", color="green")

    def _summary(self, scope_name="summary"):
        """
        Create an operation to retrieve all summaries
        added to the graph
        :return:
        """
        with tf.variable_scope(scope_name) as _:
            self.merged_summary = []
            for bucket_id in range(len(self.buckets)):
                self.merged_summary.append(
                    tf.summary.scalar(name="training_loss_bucket_{}".format(bucket_id),
                                      tensor=self.train_losses[bucket_id]))

            tf.summary.scalar(name="bucket_id", tensor=self.bucket_id)

    def forward_with_feed_dict(self, bucket_id, session, questions, answers):
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {self.bucket_id: bucket_id}

        # Instead of an array of dim (batch_size, bucket_length),
        # the model is passed a list of sized batch_size, containing vector of size bucket_length
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = questions[:, l]

        # Same for decoder_input
        for l in range(decoder_size):
            input_feed[self.targets[l].name] = answers[:, l]
            input_feed[self.target_weights[l].name] = np.not_equal(answers[:, l], 0).astype(np.float32)

        #input_feed[self.decoder_inputs[decoder_size].name] = np.zeros_like(answers[:, 0], dtype=np.int64)

        output_feed = [self.merged_summary[bucket_id],  # Summary operation
                       self.global_step,  # Current global step
                       self.updates[bucket_id],  # Nothing
                       self.gradient_norms[bucket_id],  # A scalar the gradient norm
                       self.train_losses[bucket_id]]  # Training loss, a scalar

        for l in range(decoder_size):
            # Will return a numpy array [batch_size x size_vocab x 1]. Value are not restricted to [-1, 1]
            output_feed.append(self.train_outputs[bucket_id][l])

        # outputs is a list of size (3 + decoder_size)
        outputs = session.run(output_feed, input_feed)
        return outputs

    def predict(self, bucket_id, session, questions, answers):
        """
                Forward pass and backward
                :param bucket_id:
                :param session:
                :return:
                """
        # Retrieve size of sentence for this bucket
        encoder_size, decoder_size = self.buckets[bucket_id]

        input_feed = {self.bucket_id: bucket_id}
        # questions, answers = session.run([self._questions, self._answers], input_feed)

        # Instead of an array of dim (batch_size, bucket_length),
        # the model is passed a list of sized batch_size, containing vector of size bucket_length
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = questions[:, l]

        # Same for decoder_input
        for l in range(decoder_size):
            input_feed[self.targets[l].name] = answers[:, l]
            input_feed[self.target_weights[l].name] = np.not_equal(answers[:, l], 0).astype(np.float32)

        output_feed = []
        for l in range(decoder_size):
            output_feed.append(self.test_outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        return outputs, questions, answers
