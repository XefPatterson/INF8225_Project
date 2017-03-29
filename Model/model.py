import numpy as np
import tensorflow as tf
from termcolor import cprint
import my_seq2seq
import copy

FLAGS = None


class Seq2Seq(object):
    def __init__(self, *args):
        """
        Seq2Seq model
        :param buckets: List of pairs
            Each pair correspond to (max_size_in_bucket_for_encoder_sentence, max_size_in_bucket_for_decoder_sentence)
        :param forward_only: Boolean (False)
            Whether to update the model, or only predict.
            Now it only supports False, but it should not be a big deal
        """
        self.encoder_inputs = []

        self.max_encoder_sequence_length = 100
        self.max_decoder_sequence_length = 100
        self.max_length_encoder_in_batch = tf.placeholder(tf.int32)

        self.batch_size = FLAGS.batch_size

        for i in range(self.max_encoder_sequence_length):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))

        cell_state_size = 256

        cell = tf.contrib.rnn.GRUCell(cell_state_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * 1)

        num_symbol_decoder = 55
        embedding_size_decoder = 64
        decoder1 = my_seq2seq.Decoder(name="decoder1",
                                      num_symbols=num_symbol_decoder,
                                      cell=cell,
                                      beam_size=10,
                                      beam_search=True,
                                      do_attention=True,
                                      embedding_size=embedding_size_decoder,
                                      share_embedding_with_encoder=False,
                                      output_projection=False,
                                      train_encoder_weight=False,
                                      max_decoder_sequence_length=self.max_decoder_sequence_length)

        self.all_decoders = [decoder1]

        num_symbol_encoder = 55
        embedding_size_encoder = 128
        cell_encoder = copy.deepcopy(cell)
        my_seq2seq.myseq2seq(encoder_inputs=self.encoder_inputs,
                             max_length_encoder_in_batch=self.max_length_encoder_in_batch,
                             num_encoder_symbols=num_symbol_encoder,
                             embedding_size_encoder=embedding_size_encoder,
                             encoder_cell_fw=cell_encoder,
                             decoders=self.all_decoders)

        my_seq2seq.loss_per_decoder(self.all_decoders)
        # COMPILE UP TO HERE !! WOUHOUHOU !!

    def forward_with_feed_dict(self, bucket_id, session, questions, answers):
        decoder_to_use = self.all_decoders[0]
        encoder_size, decoder_size = 40, 40

        input_feed = {
            self.max_encoder_sequence_length: encoder_size,
            decoder_to_use.max_length_decoder_in_batch: decoder_size,
            decoder_to_use.is_training: True
        }

        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = questions[:, l]

        for l in range(decoder_size):
            input_feed[decoder_to_use.targets[l].name] = answers[:, l]
            input_feed[decoder_to_use.target_weights[l].name] = np.not_equal(answers[:, l], 0).astype(np.float32)

        output_feed = [decoder_to_use.train_fn]

        for l in range(decoder_size):
            output_feed.append(decoder_to_use.outputs[l])
        
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
