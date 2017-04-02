import numpy as np
import tensorflow as tf
from termcolor import cprint
import seq2seq

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
        self.is_training = tf.placeholder(tf.bool)

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

        # decoder inputs : 'GO' + [ y_1, y_2, ... y_t-1 ]
        self.decoder_inputs = [tf.zeros_like(self.targets[0], dtype=tf.int64, name='GO')] + self.targets[:-1]

        # Binary mask useful for padded sequences.
        self.target_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.targets]

        self.gradient_norms = []
        self.updates = []

        # Extract graph from graph to plot them
        self.retrieve_attentions = True

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
            return seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=FLAGS.vocab_size,
                num_decoder_symbols=FLAGS.vocab_size,
                embedding_size=128,
                output_projection=None,
                feed_previous=do_decode)

        with tf.variable_scope("seq2seq") as _:
            model_infos = seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.target_weights,
                self.buckets,
                lambda x, y: seq2seq_f(x, y, self.is_training))

            self.outputs = model_infos[0]
            self.losses = model_infos[1]

            if self.retrieve_attentions:
                self.attentions = model_infos[2]

        # cprint("[*] Building model (D)", color="yellow")
        # Question encoder :
        # single_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
        # disc_q_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=FLAGS.keep_prob)
        # if FLAGS.num_layers > 1:
        #    disc_q_cell = tf.contrib.rnn.MultiRNNCell([single_cell] * FLAGS.num_layers)
        # disc_q_outputs, disc_q_states = tf.nn.dynamic_rnn(
        #    cell=disc_q_cell,
        #    inputs=self.encoder_inputs, # TODO, we need a 2D tensor instead of list
        #    sequence_length=length, #TODO use length
        #    dtype=tf.float32,
        #    swap_memory=True)

        # TODO: Combine Real and Fake answers into a single minibatch

        # TODO: get char/word embeddings of real/fake inputs

        # TODO: take last state or output question encoder and concat to each embedding_t

        # TODO: Feed this to a "basic" stacked gru/lstm

        # TODO: use softmax to classify each timestep as real/fake.

        # TODO: define optimization for D and for G


        # Optimization :
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.learning_rate)

        for b in range(len(self.buckets)):
            cprint("Constructing the forward pass for bucket {}".format(b))
            gradients = tf.gradients(self.losses[b], params, aggregation_method=2)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             self.max_gradient_norm)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))

        cprint("[!] Model built", color="green")

    def forward_with_feed_dict(self, bucket_id, session, questions, answers, is_training=False):
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {self.is_training: is_training}

        # Instead of an array of dim (batch_size, bucket_length),
        # the model is passed a list of sized batch_size, containing vector of size bucket_length
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = questions[:, l]

        # Same for decoder_input
        for l in range(decoder_size):
            input_feed[self.targets[l].name] = answers[:, l]
            input_feed[self.target_weights[l].name] = np.not_equal(answers[:, l], 0).astype(np.float32)
        if is_training:
            output_feed = [
                self.global_step,  # Current global step
                self.updates[bucket_id],  # Nothing
                self.gradient_norms[bucket_id],  # A scalar the gradient norm
                self.losses[bucket_id]]  # Training loss, a scalar
        else:
            output_feed = []
        if self.retrieve_attentions:
            output_feed.append(self.attentions[bucket_id])

        for l in range(decoder_size):
            # Will return a numpy array [batch_size x size_vocab x 1]. Value are not restricted to [-1, 1]
            output_feed.append(self.outputs[bucket_id][l])

        # outputs is a list of size (3 + decoder_size)
        outputs = session.run(output_feed, input_feed)

        # Cleaner output dic
        outputs_dic = {
            "predictions": outputs[-decoder_size:]
        }
        if self.retrieve_attentions:
            outputs_dic["attentions"] = outputs[-decoder_size - 1]

        if is_training:
            outputs_dic["losses"] = outputs[3]

        return outputs_dic
