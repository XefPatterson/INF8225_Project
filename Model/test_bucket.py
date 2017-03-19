import tensorflow as tf
import numpy as np

"""
   Note: This file is a mess, I am experimenting to create bucket queues




    TFRECORDS functions




"""

_buckets = [(30, 30), (60, 60), (100, 100), (150, 150)]


def make_single_example(data):
    ex = tf.train.SequenceExample()

    # Sequential (with different sizes) features
    fl_question = ex.feature_lists.feature_list["question"]
    fl_answer = ex.feature_lists.feature_list["answer"]

    ex.context.feature["length_question"].int64_list.value.append(len(data[0]))
    ex.context.feature["length_answer"].int64_list.value.append(len(data[1]))

    # Question sequence (list of int)
    for index in data[0]:
        fl_question.feature.add().int64_list.value.append(index)

    # Answer sequence (list of int)
    for index in data[1]:
        fl_answer.feature.add().int64_list.value.append(index)
    return ex


def create_tf_examples(buckets=_buckets,
                       val_p=0.0,
                       test_p=0.0,
                       saved_stats_for_set=False):
    """
    Merge every example in three TFRecord files (train-test-validation set)
    :param val_p: float (default : 0.1)
        Percentage for validation set
    :param test_p: float (default : 0.1)
        Percentage for test set
    :return:
    """
    from termcolor import cprint
    from tqdm import tqdm
    import os
    import pickle

    cprint("[*] Save example in tfRecords file", color="green")
    cprint("[!] Load question from pickle file", color="yellow", end="\r")

    file_name = os.path.dirname(os.path.abspath(__file__))
    qa_path = os.path.join(file_name, os.pardir, "Data", "MovieQA")
    qa_pairs = np.array(pickle.load(open(os.path.join(qa_path, "QA_Pairs.pkl"), "rb")))
    path_to_save_example = os.path.join(file_name, os.pardir, "Examples")

    cprint("[!] Loaded questions", color="green")

    # Split between training / validation / test set
    indexes = np.arange(len(qa_pairs))
    np.random.shuffle(indexes)
    train_indices, valid_indices, test_indices = np.split(indexes, [int((1 - test_p - val_p) * len(indexes)),
                                                                    int((1 - test_p) * len(indexes))])

    # Dictionnary containing TfRecords size (number of examples per file)
    saved_stats = {}

    # Iterate over every set
    for data, name in zip([qa_pairs[train_indices], qa_pairs[valid_indices], qa_pairs[test_indices]],
                          ["train", "val", "test"]):

        # Open as many FileWriter as buckets
        writers = []
        for bucket_id in range(len(buckets)):
            writers.append(tf.python_io.TFRecordWriter(
                os.path.join(path_to_save_example, "{}{}.tfrecords".format(name, bucket_id))))

        # Create a TFRecordWriter
        stat_set = [0 for _ in range(len(writers))]

        # Iterate over all examples
        for example in tqdm(data, desc="Creating {} record file".format(name)):
            for bucket_id, (question_length, answer_length) in enumerate(buckets):

                # Write an example in its respective bucket file
                if len(example[0]) < question_length and len(example[1]) < answer_length:
                    writers[bucket_id].write(make_single_example(example).SerializeToString())
                    stat_set[bucket_id] += 1
                    break

        # Close buckets
        for writer in writers:
            writer.close()

        # Add statistics
        saved_stats[name] = stat_set

    # Saved statistics
    if saved_stats_for_set:
        pickle.dump(saved_stats,
                    open(path_to_save_example, "stat_example_file.pkl"),
                    protocol=pickle.HIGHEST_PROTOCOL)


"""







    QUEUE functions







"""


def create_single_queue(bucket_id, filename):
    filename_queue = tf.train.string_input_producer("{}{}.tfrecords".format(filename, bucket_id))
    reader = tf.TFRecordReader()

    # Read a single example
    _, serialized_example = reader.read(filename_queue)

    context_features = {
        "length_question": tf.FixedLenFeature([], dtype=tf.int64),
        "length_answer": tf.FixedLenFeature([], dtype=tf.int64)
    }

    sequence_features = {
        "question": tf.VarLenFeature(dtype=tf.int64),
        "answer": tf.VarLenFeature(dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    batch_size = 1
    capacity = 10 * batch_size
    min_after_dequeue = 9 * batch_size

    length_question = context_parsed["length_question"]
    question = sequence_parsed["question"]

    length_answer = context_parsed["length_answer"]
    answer = sequence_parsed["answer"]

    # Pad questions to the maximum size in their bucket
    question = tf.pad(question, [0, _buckets[bucket_id][0] - length_question])
    answer = tf.pad(answer, [0, _buckets[bucket_id][1] - length_answer])

    # Shuffle queue
    queue = tf.RandomShuffleQueue(capacity,
                                  min_after_dequeue,
                                  [tf.int64, tf.int64])
    queue.enqueue([question, answer])

    # Dequeue a single element
    return queue.dequeue()


def create_queues_for_bucket(batch_size, filename):
    # For every buckets, create a ShuffleExample which return a single
    # element in that bucket
    shuffle_queues = []
    for bucket_id in range(len(_buckets)):
        shuffle_queues.append(create_single_queue(bucket_id, filename))

    capacity = 30 * batch_size

    # For every buckets, create a queue which return batch_size example
    # of that bucket
    all_queues = []
    for bucket_id in range(len(_buckets)):
        all_queues.append(tf.train.batch(shuffle_queues[bucket_id],
                                         batch_size,
                                         capacity=capacity))
    return all_queues


"""





    MODEL FUNCTION




"""
flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0001")
flags.DEFINE_float("decay_learning_rate_step", 10000, "Step to decay the learning rate [10000]")
flags.DEFINE_float("learning_rate_decay_factor", 0.96, "Learning rate decay [0.96]")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("vocab_size", 44, "The size of the vocabulary [64]")
flags.DEFINE_float("keep_prob", 0.5, "Dropout ratio [0.5]")

flags.DEFINE_integer("hidden_size", 128, "Hidden size of RNN cell [128]")
flags.DEFINE_integer("num_layers", 1, "Num of layers [1]")

FLAGS = flags.FLAGS


class Seq2Seq_Char_Level:
    def __init__(self,
                 buckets=_buckets,
                 forward_only=False):

        self.max_gradient_norm = FLAGS.max_gradient_norm
        self.learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_decay_op = tf.train.exponential_decay(FLAGS.learning_rate,
                                                                 self.global_step,
                                                                 FLAGS.decay_learning_rate_step,
                                                                 FLAGS.learning_rate_decay_factor,
                                                                 staircase=True)
        self.buckets = buckets

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        self.gradient_norms = []
        self.updates = []

        self.forward_only = forward_only

        self.bucket_id = tf.placeholder_with_default(0, tf.int32, name="bucket_id")

    def inputs(self):
        queues = create_queues_for_bucket(FLAGS.batch_size, filename="train")
        print(queues)
        # for i in range(self.buckets[-1][0]):  # Last bucket is the biggest one.
        #     self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
        #                                               name="encoder{0}".format(i)))
        # for i in range(self.buckets[-1][1] + 1):
        #     self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
        #                                               name="decoder{0}".format(i)))
        #     self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
        #                                               name="weight{0}".format(i)))

    def build(self):
        single_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=FLAGS.keep_prob)
        if FLAGS.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * FLAGS.num_layers)

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=FLAGS.vocab_size,
                num_decoder_symbols=FLAGS.vocab_size,
                embedding_size=FLAGS.hidden_size,
                output_projection=None,
                feed_previous=do_decode)

        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]

        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, self.buckets, tf.cond(self.forward_only,
                                                       lambda x, y: seq2seq_f(x, y, False),
                                                       lambda x, y: seq2seq_f(x, y, True)),
            softmax_loss_function=None)

        params = tf.trainable_variables()
        if not self.forward_only:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            am = 2  # tf.gradients.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
            for b in range(len(self.buckets)):
                gradients = tf.gradients(self.losses[b], params, aggregation_method=am)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

    def forward(self, bucket_id, session):
        encoder_size, decoder_size = self.buckets[bucket_id]

        input_feed = {self.bucket_id: bucket_id}

        output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                       self.gradient_norms[bucket_id],  # Gradient norm.
                       self.losses[bucket_id]]  # Loss for this batch.
        for l in xrange(decoder_size):  # Output logits.
            output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        return outputs


"""





    TRAIN function



"""
model = Seq2Seq_Char_Level()

