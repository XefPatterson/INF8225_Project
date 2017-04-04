import tensorflow as tf
from tqdm import trange
import model
import pickle
import os
import utils
import numpy as np

debug = False  # Fast testing (keep only the first two buckets)
verbose = True

flags = tf.app.flags
flags.DEFINE_integer("nb_epochs", 100000, "Epoch to train [100 000]")
flags.DEFINE_integer("save_frequency", 600, "Output frequency")
flags.DEFINE_integer("nb_iter_per_epoch", 100, "Output frequency")

# Optimization
flags.DEFINE_float("learning_rate", 0.00005, "Learning rate of for adam [0.0001")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 64, "The size of the batch [64]")

# Vocabulary
flags.DEFINE_integer("num_samples", 8003, "Number of samples for sampled softmax.")
flags.DEFINE_integer("vocab_size_words", 8003, "The size of the word vocabulary [8003]")
flags.DEFINE_integer("vocab_size_chars", 55, "The size of the char vocabulary [55]")
flags.DEFINE_integer("is_char_level_encoder", False, "Is the encoder char level based")
flags.DEFINE_integer("is_char_level_decoder", True, "Is the decoder char level based")

flags.DEFINE_float("keep_prob", 0.9, "Dropout ratio [0.9]")
flags.DEFINE_integer("num_layers", 3, "Num of layers [3]")
flags.DEFINE_integer("hidden_size", 512, "Hidden size of RNN cell [256]")
flags.DEFINE_integer("embedding_size", 512, "Symbol embedding size")

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
        FLAGS.save_frequency = 30

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

    if verbose:
        print("\n [Verbose] Zipped buckets :", mix_bucket_lengths)
        print(" [Verbose] Bucket sizes :", bucket_sizes_words, "\n")

    model.FLAGS = FLAGS
    seq2seq = model.Seq2Seq(buckets=mix_bucket_lengths)
    seq2seq.build()

    enc_name = "char" if FLAGS.is_char_level_encoder else "word"
    dec_name = "char" if FLAGS.is_char_level_decoder else "word"
    enc_dec_name = enc_name+"2"+dec_name
    log_dir = enc_dec_name + "_" + str(FLAGS.num_layers) + "x" + str(FLAGS.hidden_size) + "_embed" + str(FLAGS.embedding_size)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    sv = tf.train.Supervisor(logdir=log_dir,
                             global_step=seq2seq.global_step,
                             save_model_secs=FLAGS.save_frequency)

    # Test : Start small, expand.
    n_bucket_to_train = 1
    epochs_before_increase = 1
    until_increase_to_more_buckets = epochs_before_increase
    if verbose:
        print("\n [Verbose] Using buckets :", bucket_sizes[:n_bucket_to_train], "\n")

    with sv.managed_session() as sess:
        print(sess.run(seq2seq.global_step))
        while not sv.should_stop():
            iter = seq2seq.global_step.eval(sess)
            if verbose:
                print("\n [Verbose] Iter :", iter, '\n')

            if until_increase_to_more_buckets == 0 and n_bucket_to_train < len(bucket_sizes)-1:
                if verbose:
                    print("\n [Verbose] Expanding buckets to learn from... \n")
                n_bucket_to_train += 1
                until_increase_to_more_buckets = epochs_before_increase
            else:
                until_increase_to_more_buckets -= 1

            train_losses = []

            # Start epoch :
            for _ in trange(FLAGS.nb_iter_per_epoch, leave=False):
                # Pick bucket
                chosen_bucket_id = utils.get_random_bucket_id_pkl(bucket_sizes[:n_bucket_to_train])
                questions, answers = utils.get_mix_batch(qa_pairs, qa_pairs_words,
                                                         bucket_lengths, bucket_lengths_words,
                                                         FLAGS.is_char_level_encoder, FLAGS.is_char_level_decoder,
                                                         chosen_bucket_id, FLAGS.batch_size)

                # Run session
                out = seq2seq.forward_with_feed_dict(chosen_bucket_id, sess, questions, answers, is_training=True)
                train_losses.append(out['losses'])
            if verbose:
                print(" [Verbose] Average loss for epoch =", np.mean(out['losses']), '\n')

            # Run testing
            chosen_bucket_id = utils.get_random_bucket_id_pkl(bucket_sizes[:n_bucket_to_train])
            questions, answers = utils.get_mix_batch(qa_pairs, qa_pairs_words,bucket_lengths, bucket_lengths_words,
                                                     FLAGS.is_char_level_encoder, FLAGS.is_char_level_decoder,
                                                     chosen_bucket_id, FLAGS.batch_size)
            out = seq2seq.forward_with_feed_dict(chosen_bucket_id, sess, questions, answers,
                                                 is_training=False)

            # Decrypt and display answers
            if verbose:
                utils.decrypt(questions, answers, out["predictions"], idx_to_char, idx_to_words, FLAGS.batch_size,
                              char_encoder=FLAGS.is_char_level_encoder, char_decoder=FLAGS.is_char_level_decoder)
                print("\n [Verbose] Test batch loss =", out['losses'], '\n')

            # Plot attentions
            utils.plot_attention(questions, out["attentions"], out["predictions"], idx_to_char, idx_to_words,
                                FLAGS.batch_size, FLAGS.is_char_level_encoder, FLAGS.is_char_level_decoder)