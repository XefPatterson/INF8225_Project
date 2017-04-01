import tensorflow as tf
from tqdm import trange
import numpy as np
import model
import pickle
import os
import utils

flags = tf.app.flags
flags.DEFINE_integer("nb_epochs", 100000, "Epoch to train [100 000]")
flags.DEFINE_integer("nb_iter_per_epoch", 100, "Epoch to train [100]")

flags.DEFINE_integer("out_frequency", 200, "Output frequency")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("vocab_size", 55, "The size of the vocabulary [64]")
flags.DEFINE_float("keep_prob", 0.9, "Dropout ratio [0.5]")

flags.DEFINE_integer("hidden_size", 19, "Hidden size of RNN cell [128]")
flags.DEFINE_integer("num_layers", 1, "Num of layers [1]")
flags.DEFINE_integer("size_embedding_encoder", 64, "Size of encoder embedding")
flags.DEFINE_integer("size_embedding_decoder", 64, "Size of decoder embedding")

flags.DEFINE_integer("max_encoder_sequence_length", 300, "Maximum length of any sequence for the encoder")
flags.DEFINE_integer("max_decoder_sequence_length", 300, "Maximum length of any sequence for the decoder")

FLAGS = flags.FLAGS

if __name__ == '__main__':
    with open(os.path.join('..', 'Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
        idx_to_char = pickle.load(f)

    file_name = os.path.dirname(os.path.abspath(__file__))

    # Load data in RAM:
    with open(os.path.join('..', 'Data', 'MovieQA', 'QA_Pair_Buckets.pkl'), 'rb') as f:
        data = pickle.load(f)
        qa_pairs = data['qa_pairs']
        bucket_sizes = data['bucket_sizes']
        bucket_lengths = data[
            'bucket_lengths']  # Bucketing files are still interesting because close sequence length batch examples for the encoder is optimally used
        # However there is no improvement for the decoder. I couldn't make tf.while_loop works ...

    max_encoder_length = bucket_lengths[-1][0]
    max_decoder_length = bucket_lengths[-1][1]

    model.FLAGS = FLAGS
    seq2seq = model.Seq2Seq()

    sess = tf.Session()
    group_init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(group_init_ops)

    # saver, summary_writer = utils.restore(seq2seq, sess)
    # global_step = sess.run(seq2seq.global_step)

    global_step = 0
    while global_step < FLAGS.nb_epochs:

        # Run training iterations
        for _ in trange(FLAGS.nb_iter_per_epoch, leave=False):
            # Select bucket for the epoch
            chosen_bucket_id = utils.get_random_bucket_id_pkl(bucket_sizes)

            questions, answers = utils.get_batch(qa_pairs, chosen_bucket_id, FLAGS.batch_size, max_encoder_length,
                                                 max_decoder_length)

            out = seq2seq.forward_with_feed_dict(sess, questions, answers, max_encoder_length, max_decoder_length,
                                                 is_training=True)
            utils.reconstruct_beam_search(questions, answers, out, FLAGS.batch_size)

            out = seq2seq.forward_with_feed_dict(sess, questions, answers, max_encoder_length, max_decoder_length,
                                                 is_training=False)
            utils.reconstruct_beam_search(questions, answers, out, FLAGS.batch_size)


            # Save losses
            # summary_writer.add_summary(out[0], out[1])

        # Save model
        # saver.save(sess, "model/model", global_step)

        # Run testing iterations
        chosen_bucket_id = utils.get_random_bucket_id_pkl(bucket_sizes)
        chosen_bucket_id = 0

        questions, answers = utils.get_batch(qa_pairs, bucket_lengths, chosen_bucket_id, FLAGS.batch_size)
        predictions, questions, answers = seq2seq.forward_with_feed_dict(sess, questions, answers, is_training=False)
        from IPython import embed

        embed()
        # (64, 30)

        utils.decrypt(questions, answers, predictions, idx_to_char)
