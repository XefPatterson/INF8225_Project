import numpy as np
import tensorflow as tf
from termcolor import cprint
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_seq2seq
from my_seq2seq import embedding_attention_seq2seq, model_with_buckets

FLAGS = None


class Seq2Seq(object):
    def __init__(self,
                 buckets,
                 forward_only=False,
                 do_attention=False):
        """
        Seq2Seq model
        :param buckets: List of pairs
            Each pair correspond to (max_size_in_bucket_for_encoder_sentence, max_size_in_bucket_for_decoder_sentence)
        :param forward_only: Boolean (False)
            Whether to update the model, or only predict.
            Now it only supports False, but it should not be a big deal
        """
        self.seq2seq_model = embedding_attention_seq2seq if do_attention else embedding_rnn_seq2seq

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

        # Binary mask useful for padded sequences.
        self.target_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.targets]

        self.gradient_norms = []
        self.updates = []

        self.forward_only = forward_only

        # Whether we should feed the previous output from the decoder
        self.feed_previous = tf.placeholder(tf.bool)

    def build(self):
        """
        Build the model
        :return:
        """
        cprint("[*] Building model", color="yellow")
        # self._build_queues()
        single_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=FLAGS.keep_prob)
        if FLAGS.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * FLAGS.num_layers)

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return self.seq2seq_model(encoder_inputs, decoder_inputs, cell,
                                      num_encoder_symbols=FLAGS.vocab_size,
                                      num_decoder_symbols=FLAGS.vocab_size,
                                      embedding_size=128,
                                      output_projection=None,
                                      save_attention=True,
                                      feed_previous=do_decode)

        with tf.variable_scope("seq2seq") as scope:
            self.outputs, self.losses, _ = model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.target_weights,
                self.buckets,
                lambda x, y, z: seq2seq_f(x, y, self.feed_previous),
                save_attention=True)

            scope.reuse_variables()

        params = tf.trainable_variables()
        if not self.forward_only:
            opt = tf.train.AdamOptimizer(self.learning_rate)
            for b in range(len(self.buckets)):
                cprint("Constructing the forward pass for bucket {}".format(b))

                gradients = tf.gradients(self.losses[b], params, aggregation_method=2)
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
                                      tensor=self.losses[bucket_id]))

    def forward_with_feed_dict(self, bucket_id, session, questions, answers):
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {self.feed_previous: False}

        # Instead of an array of dim (batch_size, bucket_length),
        # the model is passed a list of sized batch_size, containing vector of size bucket_length
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = questions[:, l]

        # Same for decoder_input
        for l in range(decoder_size):
            input_feed[self.targets[l].name] = answers[:, l]
            input_feed[self.target_weights[l].name] = np.not_equal(answers[:, l], 0).astype(np.float32)

        # input_feed[self.decoder_inputs[decoder_size].name] = np.zeros_like(answers[:, 0], dtype=np.int64)

        output_feed = [self.merged_summary[bucket_id],  # Summary operation
                       self.global_step,  # Current global step
                       self.updates[bucket_id],  # Nothing
                       self.gradient_norms[bucket_id],  # A scalar the gradient norm
                       self.losses[bucket_id]]  # Training loss, a scalar

        for l in range(decoder_size):
            # Will return a numpy array [batch_size x size_vocab x 1]. Value are not restricted to [-1, 1]
            output_feed.append(self.outputs[bucket_id][l])

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

        input_feed = {self.feed_previous: True}
        # questions, answers = session.run([self._questions, self._answers], input_feed)

        # Instead of an array of dim (batch_size, bucket_length),
        # the model is passed a list of sized batch_size, containing vector of size bucket_length
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = questions[:, l]

        # Same for decoder_input
        for l in range(decoder_size):
            input_feed[self.targets[l].name] = answers[:, l]
            input_feed[self.target_weights[l].name] = np.not_equal(answers[:, l], 0).astype(np.float32)

        # input_feed[self.decoder_inputs[decoder_size].name] = np.zeros_like(answers[:, 0], dtype=np.int64)
        # input_feed[self.target_weights[decoder_size - 1].name] = np.zeros_like(answers[:, 0], dtype=np.int64)

        output_feed = []
        for l in range(decoder_size):
            output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        return outputs, questions, answers
