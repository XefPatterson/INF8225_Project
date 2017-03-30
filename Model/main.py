import tensorflow as tf
from tqdm import trange
import numpy as np
import model
import pickle
import os
import utils

# _buckets = [(30, 30), (60, 60)]  # , (100, 100), (150, 150)

flags = tf.app.flags
flags.DEFINE_integer("nb_epochs", 100000, "Epoch to train [100 000]")
flags.DEFINE_integer("nb_iter_per_epoch", 100, "Epoch to train [100]")

flags.DEFINE_integer("out_frequency", 200, "Output frequency")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0001")
flags.DEFINE_float("decay_learning_rate_step", 10000, "Step to decay the learning rate [10000]")
flags.DEFINE_float("learning_rate_decay_factor", 0.96, "Learning rate decay [0.96]")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("vocab_size", 55, "The size of the vocabulary [64]")
flags.DEFINE_float("keep_prob", 0.9, "Dropout ratio [0.5]")

flags.DEFINE_integer("hidden_size", 19, "Hidden size of RNN cell [128]")
flags.DEFINE_integer("num_layers", 1, "Num of layers [1]")

FLAGS = flags.FLAGS

if __name__ == '__main__':
    with open(os.path.join('..', 'Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
        idx_to_char = pickle.load(f)

    file_name = os.path.dirname(os.path.abspath(__file__))
    """
    path_to_save_example = os.path.join(file_name, os.pardir, "Examples", "stat_example_file.pkl")
    if not os.path.exists(path_to_save_example):
        import tf_records
        tf_records.create_tf_examples(_buckets, saved_stats_for_set=True)
    with open(path_to_save_example, 'rb') as f:
        size_tf_records = pickle.load(f)
    """

    # Load data in RAM:
    with open(os.path.join('..', 'Data', 'MovieQA', 'QA_Pair_Buckets.pkl'), 'rb') as f:
        data = pickle.load(f)
        qa_pairs = data['qa_pairs']
        bucket_sizes = data['bucket_sizes']
        bucket_lengths = data['bucket_lengths']

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
            chosen_bucket_id = 0
            questions, answers = utils.get_batch(qa_pairs, bucket_lengths, chosen_bucket_id, FLAGS.batch_size)

            out = seq2seq.forward_with_feed_dict(sess, questions, answers)
            # print(out[1])
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
