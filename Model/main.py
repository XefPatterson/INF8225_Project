import tensorflow as tf
from tqdm import trange
import model
import pickle
import os
import utils
import numpy as np

debug = False  # Fast testing (keep only the first two buckets)
verbose = True
private = False

flags = tf.app.flags
flags.DEFINE_integer("nb_epochs", 100000, "Epoch to train [100 000]")
flags.DEFINE_integer("save_frequency", 1800, "Output frequency")
flags.DEFINE_integer("nb_iter_per_epoch", 250, "Output frequency")

# Optimization
flags.DEFINE_float("learning_rate", 0.00003, "Learning rate of for adam [0.0001")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 16, "The size of the batch [64]")

# Vocabulary
flags.DEFINE_integer("num_samples", 256, "Number of samples for sampled softmax.")
flags.DEFINE_integer("vocab_size_words", 8003, "The size of the word vocabulary [8003]")
flags.DEFINE_integer("vocab_size_chars", 55, "The size of the char vocabulary [55]")
flags.DEFINE_integer("is_char_level_encoder", True, "Is the encoder char level based")
flags.DEFINE_integer("is_char_level_decoder", True, "Is the decoder char level based")

flags.DEFINE_float("keep_prob", 0.75, "Dropout ratio [0.9]")
flags.DEFINE_integer("num_layers", 3, "Num of layers [3]")
flags.DEFINE_integer("hidden_size", 256, "Hidden size of RNN cell [256]")
flags.DEFINE_integer("embedding_size", 128, "Symbol embedding size")
flags.DEFINE_integer("use_attention", False, "Use attention mechanism?")
flags.DEFINE_integer("valid_start", 0.98, "Validation set start ratio")

flags.DEFINE_string("dataset", "messenger", "Dataset to use")

FLAGS = flags.FLAGS

if __name__ == '__main__':
    with open(os.path.join('..', 'Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
        idx_to_char = pickle.load(f)

    with open(os.path.join('..', 'Data', 'MovieQA', 'idx_to_words.pkl'), 'rb') as f:
        idx_to_words = pickle.load(f)

    file_name = os.path.dirname(os.path.abspath(__file__))

    if FLAGS.dataset == "movie":
        # Load data in RAM:
        with open(os.path.join('..', 'Data', 'MovieQA', 'QA_Pairs_Chars_Buckets.pkl'), 'rb') as f:
            data = pickle.load(f)
            qa_pairs = data['qa_pairs']
            bucket_sizes = data['bucket_sizes']
            bucket_lengths = data['bucket_lengths']

        with open(os.path.join('..', 'Data', 'MovieQA', 'QA_Pairs_Words_Buckets.pkl'), 'rb') as f:
            data_words = pickle.load(f, encoding='latin1')
            qa_pairs_words = data_words['qa_pairs']
            bucket_sizes_words = data_words['bucket_sizes']
            bucket_lengths_words = data_words['bucket_lengths']
    else:
        with open(os.path.join('..', 'Data', 'Messenger', 'QA_Pairs_Chars_Buckets_FJ.pkl'), 'rb') as f:
            data = pickle.load(f)
            qa_pairs = data['qa_pairs']
            bucket_sizes = data['bucket_sizes']
            bucket_lengths = data['bucket_lengths']
            bucket_sizes = bucket_sizes[:-1]
            bucket_lengths = bucket_lengths[:-1]

        # Flemme de modifier le code en profondeur ^^
        data_words = data
        qa_pairs_words = qa_pairs
        bucket_sizes_words = bucket_sizes
        bucket_lengths_words = bucket_lengths

        # Un bon petit assert suffira, je pense
        assert FLAGS.is_char_level_encoder == FLAGS.is_char_level_decoder and FLAGS.is_char_level_encoder is True, "With Messenger dataset, encoder and decoder should be at char level"

    if debug:
        # Faster testing
        to_bucket = 2
        bucket_lengths = bucket_lengths[:to_bucket]
        bucket_lengths_words = bucket_lengths_words[:to_bucket]
        bucket_sizes = bucket_sizes[:to_bucket]
        bucket_sizes_words = bucket_sizes_words[:to_bucket]
        FLAGS.nb_iter_per_epoch = 100
        FLAGS.hidden_size = 256
        FLAGS.num_layers = 1
        FLAGS.batch_size = 64
        FLAGS.save_frequency = 120

    assert len(bucket_lengths) == len(bucket_lengths_words), "Not the same number of buckets!"
    mix_bucket_lengths = []

    if FLAGS.is_char_level_encoder and FLAGS.is_char_level_decoder:  # CHAR | CHAR
        mix_bucket_lengths = bucket_lengths

    elif not FLAGS.is_char_level_encoder and not FLAGS.is_char_level_decoder:  # WORD | WORD
        mix_bucket_lengths = bucket_lengths_words

    else:
        for i in range(len(bucket_lengths)):
            if not FLAGS.is_char_level_encoder and FLAGS.is_char_level_decoder:  # WORD | CHAR
                mix_bucket_lengths.append((bucket_lengths_words[i][0], bucket_lengths[i][1]))

            if FLAGS.is_char_level_encoder and not FLAGS.is_char_level_decoder:  # CHAR | WORD
                mix_bucket_lengths.append((bucket_lengths[i][0], bucket_lengths_words[i][1]))

    #mix_bucket_lengths = mix_bucket_lengths[:-1]
    #bucket_sizes = bucket_sizes[:-1]
    #bucket_sizes_words = bucket_sizes_words[:-1]

    if verbose:
        print("\n [Verbose] Zipped buckets :", mix_bucket_lengths)
        print(" [Verbose] Bucket sizes :", bucket_sizes_words, "\n")

    model.FLAGS = FLAGS
    seq2seq = model.Seq2Seq(buckets=mix_bucket_lengths)
    seq2seq.build()

    # Relevant log dir:
    enc_name = "char" if FLAGS.is_char_level_encoder else "word"
    dec_name = "char" if FLAGS.is_char_level_decoder else "word_nSamples" + str(FLAGS.num_samples)
    enc_dec_name = enc_name + "2" + dec_name
    if FLAGS.dataset != "movie":
        enc_dec_name += "_messenger"
    if FLAGS.use_attention:
        enc_dec_name += "Att"
    log_dir = enc_dec_name + "_" + str(FLAGS.num_layers) + "x" + str(FLAGS.hidden_size) + "_embed" + str(
        FLAGS.embedding_size)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    sv = tf.train.Supervisor(logdir=log_dir,
                             global_step=seq2seq.global_step,
                             save_model_secs=FLAGS.save_frequency)
    avg_train_losses = []
    valid_losses = []
    with sv.managed_session() as sess:
        print(sess.run(seq2seq.global_step))
        while not sv.should_stop():
            iter = seq2seq.global_step.eval(sess)
            if verbose:
                print("\n [Verbose] Iter :", iter, '\n')

            train_losses = []
            # Start epoch :
            for _ in trange(FLAGS.nb_iter_per_epoch, leave=False):
                # Pick bucket
                chosen_bucket_id = utils.get_random_bucket_id_pkl(bucket_sizes)
                questions, answers = utils.get_mix_batch(qa_pairs, qa_pairs_words,
                                                         bucket_lengths, bucket_lengths_words,
                                                         FLAGS.is_char_level_encoder, FLAGS.is_char_level_decoder,
                                                         chosen_bucket_id, FLAGS.batch_size, train=True,
                                                         valid_start=FLAGS.valid_start)

                # Run session
                out = seq2seq.forward_with_feed_dict(chosen_bucket_id, sess, questions, answers, is_training=True)
                train_losses.append(out['losses'])

            avg_train_losses.append(np.mean(out['losses']))

            # Run testing
            chosen_bucket_id = utils.get_random_bucket_id_pkl(bucket_sizes)
            questions, answers = utils.get_mix_batch(qa_pairs, qa_pairs_words, bucket_lengths, bucket_lengths_words,
                                                     FLAGS.is_char_level_encoder, FLAGS.is_char_level_decoder,
                                                     chosen_bucket_id, FLAGS.batch_size, train=False,
                                                     valid_start=FLAGS.valid_start)

            out = seq2seq.forward_with_feed_dict(chosen_bucket_id, sess, questions, answers,
                                                 is_training=False)
            valid_losses.append(out['losses'])
            # Decrypt and display answers
            if verbose:
                if not private:
                    utils.decrypt(questions, answers, out["predictions"], idx_to_char, idx_to_words, FLAGS.batch_size,
                              char_encoder=FLAGS.is_char_level_encoder, char_decoder=FLAGS.is_char_level_decoder)
                print(" [Verbose] TRAIN average loss for epoch =", avg_train_losses[-1], '\n')
                print(" [Verbose] TEST batch loss =", out['losses'], '\n')

            # Plot attentions
            if FLAGS.use_attention:
                utils.plot_attention(questions, out["attentions"], out["predictions"], idx_to_char, idx_to_words,
                                     FLAGS.batch_size, FLAGS.is_char_level_encoder, FLAGS.is_char_level_decoder,
                                     path=os.path.join(log_dir, "attention.png"))

            # Cheap save for now.
            if os.path.exists("losses.pkl"):
                with open(os.path.join(log_dir, "losses.pkl"), "rb") as f:
                    t_losses = pickle.load(f)['train_losses']
                    v_losses = pickle.load(f)['valid_losses']

                t_losses.append(avg_train_losses[-1])
                v_losses.append(valid_losses[-1])

                with open(os.path.join(log_dir, "losses.pkl"), "wb") as f:
                    pickle.dump({"train_losses": t_losses, "valid_losses": v_losses}, f)

                utils.plot_curves(t_losses, v_losses, os.path.join(log_dir, "curves.png"))

            else:
                with open(os.path.join(log_dir, "losses.pkl"), "wb") as f:
                    pickle.dump({"train_losses": avg_train_losses, "valid_losses": valid_losses}, f)
                utils.plot_curves(avg_train_losses, valid_losses, os.path.join(log_dir, "curves.png"))

            with open(os.path.join(log_dir, "session_run_losses.pkl"), "wb") as f:
                pickle.dump({"train_losses": avg_train_losses, "valid_losses": valid_losses}, f)
