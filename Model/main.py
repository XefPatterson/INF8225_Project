"""
    Run the model
"""

import tensorflow as tf
import model
import pickle
import numpy as np
import os

_buckets = [(30, 30), (60, 60)]  # , (100, 100), (150, 150)

flags = tf.app.flags
flags.DEFINE_integer("iterations", 100000, "Epoch to train [100 000]")
flags.DEFINE_integer("out_frequency", 200, "Output frequency")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0001")
flags.DEFINE_float("decay_learning_rate_step", 10000, "Step to decay the learning rate [10000]")
flags.DEFINE_float("learning_rate_decay_factor", 0.96, "Learning rate decay [0.96]")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("vocab_size", 55, "The size of the vocabulary [64]")
flags.DEFINE_float("keep_prob", 0.9, "Dropout ratio [0.5]")

flags.DEFINE_integer("hidden_size", 128, "Hidden size of RNN cell [128]")
flags.DEFINE_integer("num_layers", 1, "Num of layers [1]")

FLAGS = flags.FLAGS


def decode(out_sequence, idx_to_char):
    return "YOLO"


if __name__ == '__main__':
    # run before
    # import tf_records
    # tf_records.create_tf_examples(_buckets)

    with open(os.path.join('Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
        idx_to_char = pickle.load(f)

    model.FLAGS = FLAGS
    seq2seq = model.Seq2Seq(buckets=_buckets)
    seq2seq.build()

    sess = tf.Session()

    group_init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(group_init_ops)

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess)

    global_step = sess.run(seq2seq.global_step)
    sess.run(seq2seq.op_starting_queue)

    while global_step < FLAGS.iterations:

        out = seq2seq.forward(np.random.randint(len(_buckets)), sess)

        if global_step % FLAGS.out_frequency == 0:
            predictions = seq2seq.predict(np.random.randint(len(_buckets)), sess)
            print(len(predictions))
            print(predictions[0].shape)
            char_seq = decode(predictions, idx_to_char)
