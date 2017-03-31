import numpy as np
import tensorflow as tf
import my_seq2seq
import copy
import decoder

FLAGS = None


class Seq2Seq(object):
    def __init__(self, ):
        """
        Seq2Seq model
        :param buckets: List of pairs
            Each pair correspond to (max_size_in_bucket_for_encoder_sentence, max_size_in_bucket_for_decoder_sentence)
        :param forward_only: Boolean (False)
            Whether to update the model, or only predict.
            Now it only supports False, but it should not be a big deal
        """
        self.encoder_inputs = []

        self.max_encoder_sequence_length = FLAGS.max_encoder_sequence_length
        self.max_decoder_sequence_length = FLAGS.max_decoder_sequence_length
        self.max_length_encoder_in_batch = tf.placeholder(tf.int32)

        self.batch_size = FLAGS.batch_size

        for i in range(self.max_encoder_sequence_length):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))

        cell_state_size = FLAGS.hidden_size

        cell = tf.contrib.rnn.GRUCell(cell_state_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * FLAGS.num_layers)

        num_symbol_decoder = FLAGS.vocab_size
        embedding_size_decoder = FLAGS.size_embedding_decoder
        decoder1 = decoder.Decoder(name="decoder1",
                                   num_symbols=num_symbol_decoder,
                                   cell=cell,
                                   beam_size=1,
                                   beam_search=True,
                                   do_attention=True,
                                   embedding_size=embedding_size_decoder,
                                   share_embedding_with_encoder=False,
                                   output_projection=False,
                                   train_encoder_weight=False,
                                   max_decoder_sequence_length=self.max_decoder_sequence_length)

        self.all_decoders = [decoder1]

        num_symbol_encoder = FLAGS.vocab_size
        embedding_size_encoder = FLAGS.size_embedding_encoder
        cell_encoder = copy.deepcopy(cell)
        my_seq2seq.myseq2seq(encoder_inputs=self.encoder_inputs,
                             max_length_encoder_in_batch=self.max_length_encoder_in_batch,
                             num_encoder_symbols=num_symbol_encoder,
                             embedding_size_encoder=embedding_size_encoder,
                             encoder_cell_fw=cell_encoder,
                             decoders=self.all_decoders)

        my_seq2seq.loss_per_decoder(self.all_decoders)

    def forward_with_feed_dict(self, session, questions, answers, batch_max_encoder_size, batch_max_decoder_size,
                               is_training, decoder_to_use=0):
        assert batch_max_decoder_size < self.max_decoder_sequence_length, "Decoding sequence is larger than model capability"
        assert batch_max_encoder_size < self.max_encoder_sequence_length, "Encoding sequence is larger than model capability"

        decoder_to_use = self.all_decoders[decoder_to_use]
        encoder_size, decoder_size = batch_max_encoder_size, batch_max_encoder_size

        input_feed = {
            self.max_length_encoder_in_batch: encoder_size,
            decoder_to_use.max_length_decoder_in_batch: batch_max_decoder_size,
            decoder_to_use.is_training: is_training
        }

        for l in range(self.max_encoder_sequence_length):
            input_feed[self.encoder_inputs[l].name] = questions[:, l]

        for l in range(self.max_decoder_sequence_length):
            input_feed[decoder_to_use.targets[l].name] = answers[:, l]
            input_feed[decoder_to_use.target_weights[l].name] = np.not_equal(answers[:, l], 0).astype(np.float32)

        output_feed = []
        if is_training:
            output_feed.append(decoder_to_use.train_fn)

        for l in range(decoder_size):
            output_feed.append(decoder_to_use.outputs[l])

        outputs = session.run(output_feed, input_feed)
        return outputs
