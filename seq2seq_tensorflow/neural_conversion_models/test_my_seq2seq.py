import tensorflow as tf
import my_seq2seq

buckets = [(32, 32), (48, 48)]


class Seq2Seq(object):
    def __init__(self,
                 buckets):
        """
        Seq2Seq model
        :param buckets: List of pairs
            Each pair correspond to (max_size_in_bucket_for_encoder_sentence, max_size_in_bucket_for_decoder_sentence)
        :param forward_only: Boolean (False)
            Whether to update the model, or only predict.
            Now it only supports False, but it should not be a big deal
        """

        self.buckets = buckets

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.targets = []
        self.target_weights = []

        self.max_sequence_length = 40
        self.batch_size = 12
        self.encoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_sequence_length],
                                             name="encoder")
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[self.batch_size],
                                              name="decoder{}".format(i)) for i in range(self.max_sequence_length + 1)]

        # Our targets are decoder inputs shifted by one.

        cell_state_size = 256
        num_symbol_decoder = 55
        num_symbol_encoder = 55
        embedding_size_encoder = 128
        embedding_size_decoder = 64

        cell = tf.contrib.rnn.GRUCell(cell_state_size)
        single_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
        cell = tf.contrib.rnn.MultiRNNCell([single_cell] * 1)

        decoder_cell1 = cell
        decoder1 = my_seq2seq.Decoder(name="decoder1",
                                      num_symbols=num_symbol_decoder,
                                      inputs=self.decoder_inputs,
                                      cell=decoder_cell1,
                                      beam_size=10,
                                      beam_search=True,
                                      do_attention=True,  # TODO: Right now beam search is supported only with attention
                                      embedding_size=embedding_size_decoder,
                                      share_embedding_with_encoder=False,
                                      output_projection=False)
        import copy
        decoder_cell2 = copy.deepcopy(cell)
        decoder2 = my_seq2seq.Decoder(name="decoder2",
                                      num_symbols=num_symbol_decoder,
                                      inputs=self.decoder_inputs,
                                      cell=decoder_cell2,
                                      beam_size=1,
                                      beam_search=False,
                                      do_attention=False,
                                      embedding_size=embedding_size_encoder,
                                      share_embedding_with_encoder=True,
                                      output_projection=False)  # TODO output_projection is not used yet

        cell_encoder = copy.deepcopy(cell)
        my_seq2seq.seq2seq(encoder_inputs=self.encoder_inputs,
                           num_encoder_symbols=num_symbol_encoder,
                           embedding_size_encoder=embedding_size_encoder,
                           encoder_cell_fw=cell_encoder,
                           decoders=[decoder1, decoder2])


seq2seq = Seq2Seq(buckets)
