import copy

import tensorflow as tf

from Model import my_seq2seq


class Seq2Seq(object):
    def __init__(self):
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
        self.max_decoder_sequence_length = 101
        self.max_length_encoder_in_batch = tf.placeholder(tf.int32)

        self.batch_size = 12

        for i in range(self.max_encoder_sequence_length):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))

        cell_state_size = 256
        num_symbol_encoder = 55
        embedding_size_encoder = 128

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

        cell_encoder = copy.deepcopy(cell)
        my_seq2seq.seq2seq(encoder_inputs=self.encoder_inputs,
                           max_length_encoder_in_batch=self.max_length_encoder_in_batch,
                           num_encoder_symbols=num_symbol_encoder,
                           embedding_size_encoder=embedding_size_encoder,
                           encoder_cell_fw=cell_encoder,
                           decoders=[decoder1])


seq2seq = Seq2Seq()
