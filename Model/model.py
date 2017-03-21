import numpy as np
import tensorflow as tf
from queues import create_queues_for_bucket
from termcolor import cprint

FLAGS = None


class Seq2Seq:
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
        # TODO support decoder output is fed back into input

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
        self.target_weights = []

        for i in range(self.buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in range(self.buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))
        self.targets = [self.decoder_inputs[i + 1]
                        for i in range(len(self.decoder_inputs) - 1)]

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
        cprint("[*] Building model", color="yellow")
        self._build_queues()

        single_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=FLAGS.keep_prob)
        if FLAGS.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * FLAGS.num_layers)

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=FLAGS.vocab_size,
                num_decoder_symbols=FLAGS.vocab_size,
                embedding_size=FLAGS.hidden_size,
                output_projection=None,
                feed_previous=do_decode)

        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            self.encoder_inputs,
            self.decoder_inputs,
            self.targets,
            self.target_weights,
            self.buckets,
            lambda x, y: seq2seq_f(x, y, False),
            softmax_loss_function=None)

        params = tf.trainable_variables()
        if not self.forward_only:
            opt = tf.train.AdamOptimizer(self.learning_rate)
            for b in range(len(self.buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 self.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))
        cprint("[!] Model built :)", color="green")

    def forward(self, bucket_id, session):
        """
        Forward pass and backward
        :param bucket_id:
        :param session:
        :return:
        """
        cprint("[*] One iteration for examples in bucket {}".format(bucket_id), color="yellow", end="")
        # Retrieve size of sentence for this bucket
        encoder_size, decoder_size = self.buckets[bucket_id]

        # Retrieve a batch of example from the bucket {bucket_id}
        # Ideally we should not call twice sess.run(), in one iteration, but i don't know how to sove it:'(
        input_feed = {self.bucket_id: bucket_id}
        questions, answers = session.run([self._questions, self._answers], input_feed)

        # Instead of an array of dim (batch_size, bucket_length),
        # the model is passed a list of sized batch_size, containing vector of size bucket_length
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = questions[:, l]

        # Same for decoder_input
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = answers[:, l]
            input_feed[self.target_weights[l].name] = np.not_equal(answers[:, l+1], 0).astype(np.float32)

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros_like(answers[:, 0], dtype=np.int64)

        output_feed = [self.updates[bucket_id],
                       self.gradient_norms[bucket_id],
                       self.losses[bucket_id]]

        for l in range(decoder_size):
            output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        cprint(": [SUCCEED]".format(bucket_id), color="green")

        return outputs
