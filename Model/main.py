import tensorflow as tf
from tqdm import trange
import model
import pickle
import os
import utils
import numpy as np

debug = True  # Fast testing (keep only the first two buckets)
verbose = True

flags = tf.app.flags
flags.DEFINE_integer("nb_epochs", 100000, "Epoch to train [100 000]")
flags.DEFINE_integer("save_frequency", 1000, "Output frequency")
flags.DEFINE_integer("nb_iter_per_epoch", 100, "Output frequency")

# Optimization
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0001")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 64, "The size of the batch [64]")

# Vocabulary
flags.DEFINE_integer("num_samples", 1024, "Number of samples for sampled softmax.")
flags.DEFINE_integer("vocab_size_words", 8003, "The size of the vocabulary [64]")
flags.DEFINE_integer("vocab_size_chars", 55, "The size of the vocabulary [64]")
flags.DEFINE_integer("is_char_level_encoder", False, "Is the encoder char level based")
flags.DEFINE_integer("is_char_level_decoder", False, "Is the decoder char level based")

flags.DEFINE_float("keep_prob", 0.9, "Dropout ratio [0.9]")

flags.DEFINE_integer("hidden_size", 256, "Hidden size of RNN cell [256]")
flags.DEFINE_integer("num_layers", 1, "Num of layers [3]")

FLAGS = flags.FLAGS

if __name__ == '__main__':
    with open(os.path.join('..', 'Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
        idx_to_char = pickle.load(f)

    with open(os.path.join('..', 'Data', 'MovieQA', 'idx_to_words.pkl'), 'rb') as f:
        idx_to_words = pickle.load(f)

    file_name = os.path.dirname(os.path.abspath(__file__))

    # Load data in RAM:
    with open(os.path.join('..', 'Data', 'MovieQA', 'QA_Pairs_Chars_Buckets.pkl'), 'rb') as f:
        data = pickle.load(f)
        qa_pairs = data['qa_pairs']
        bucket_sizes = data['bucket_sizes']
        bucket_lengths = data['bucket_lengths']

    # TODO: support Python3
    with open(os.path.join('..', 'Data', 'MovieQA', 'QA_Pairs_Words_Buckets.pkl'), 'rb') as f:
        data_words = pickle.load(f, encoding='latin1')
        qa_pairs_words = data_words['qa_pairs']
        bucket_sizes_words = data_words['bucket_sizes']
        bucket_lengths_words = data_words['bucket_lengths']

    if debug:
        # Faster testing
        bucket_lengths = [bucket_lengths[0]]
        bucket_lengths_words = [bucket_lengths_words[0]]
        bucket_sizes = [bucket_sizes[0]]
        bucket_sizes_words = [bucket_sizes_words[0]]
        FLAGS.nb_iter_per_epoch = 100
        FLAGS.hidden_size = 256
        FLAGS.num_layers = 1
        FLAGS.batch_size = 64

    assert len(bucket_lengths) == len(bucket_lengths_words) , "Not the same number of buckets!"
    mix_bucket_lengths = []

    if FLAGS.is_char_level_encoder and FLAGS.is_char_level_decoder: # CHAR | CHAR
        mix_bucket_lengths = bucket_lengths

    elif not FLAGS.is_char_level_encoder and not FLAGS.is_char_level_decoder: # WORD | WORD
        mix_bucket_lengths = bucket_lengths_words

    else:
        for i in range(len(bucket_lengths)):
            if not FLAGS.is_char_level_encoder and FLAGS.is_char_level_decoder: # WORD | CHAR
                mix_bucket_lengths.append((bucket_lengths_words[i][0], bucket_lengths[i][1]))

            if FLAGS.is_char_level_encoder and not FLAGS.is_char_level_decoder: # CHAR | WORD
                mix_bucket_lengths.append((bucket_lengths[i][0], bucket_lengths_words[i][1]))

    if verbose :
        print("\n [Verbose] Zipped buckets :", mix_bucket_lengths, "\n")

    model.FLAGS = FLAGS
    seq2seq = model.Seq2Seq(buckets=mix_bucket_lengths)
    seq2seq.build()

    sv = tf.train.Supervisor(logdir="model",
                             global_step=seq2seq.global_step,
                             save_model_secs=FLAGS.save_frequency)

    with sv.managed_session() as sess:
        while not sv.should_stop():
            for _ in trange(FLAGS.nb_iter_per_epoch, leave=False):
                # Pick bucket
                chosen_bucket_id = utils.get_random_bucket_id_pkl(bucket_sizes)
                # Retrieve examples
                # questions, answers = utils.get_batch(qa_pairs, bucket_lengths, chosen_bucket_id,
                #                                      FLAGS.batch_size)

                # TODO: Activate
                questions, answers = utils.get_mix_batch(qa_pairs, qa_pairs_words,
                                                         bucket_lengths, bucket_lengths_words,
                                                         FLAGS.is_char_level_encoder, FLAGS.is_char_level_decoder,
                                                         chosen_bucket_id,
                                                         FLAGS.batch_size)

                # Run session
                out = seq2seq.forward_with_feed_dict(chosen_bucket_id, sess, questions, answers, is_training=True)

            # Run testing
            chosen_bucket_id = utils.get_random_bucket_id_pkl(bucket_sizes)
            questions, answers = utils.get_mix_batch(qa_pairs, qa_pairs_words,
                                                         bucket_lengths, bucket_lengths_words,
                                                         FLAGS.is_char_level_encoder, FLAGS.is_char_level_decoder,
                                                         chosen_bucket_id,
                                                         FLAGS.batch_size)
            out = seq2seq.forward_with_feed_dict(chosen_bucket_id, sess, questions, answers,
                                                 is_training=False)
            # Decrypt and display answers
            utils.decrypt(questions, answers, out["predictions"], idx_to_char, idx_to_words, FLAGS.batch_size,
                          char_encoder=FLAGS.is_char_level_encoder, char_decoder=FLAGS.is_char_level_decoder)
            # Plot attentions
            utils.plot_attention(questions, out["attentions"], out["predictions"], idx_to_char, idx_to_words,
                                 FLAGS.batch_size, FLAGS.is_char_level_encoder, FLAGS.is_char_level_decoder)
