import tensorflow as tf
import os


def create_single_queue(bucket_id, filename, batch_size, buckets):
    """
    Return a shuffle_queue which output element from {bucket_id} bucket
    :param bucket_id: int
    :param filename: str
    :param batch_size: int
    :param buckets: list
    :return:
    """
    file_name = os.path.dirname(os.path.abspath(__file__))
    path_to_save_example = os.path.join(file_name, os.pardir, "Examples")
    filename = os.path.join(path_to_save_example, "{}{}.tfrecords".format(filename, bucket_id))
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()

    # Read a single example
    _, serialized_example = reader.read(filename_queue)

    # Scalar features
    context_features = {
        "length_question": tf.FixedLenFeature([], dtype=tf.int64),
        "length_answer": tf.FixedLenFeature([], dtype=tf.int64)
    }

    # Tensor features
    sequence_features = {
        "question": tf.VarLenFeature(dtype=tf.int64),
        "answer": tf.VarLenFeature(dtype=tf.int64)
    }

    # Parse a single example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    batch_size = batch_size
    capacity = 10 * batch_size
    min_after_dequeue = 9 * batch_size

    # Basically, pad question with zeros if shorter than buckets[bucket_id][0]
    length_question = context_parsed["length_question"]
    question = sequence_parsed["question"]
    question = tf.sparse_tensor_to_dense(question)
    question = tf.reshape(question, [-1])
    pad_question = tf.zeros(shape=[buckets[bucket_id][0] - tf.cast(length_question, tf.int32)], dtype=tf.int64)
    question = tf.concat([question, pad_question], axis=0)
    question.set_shape(buckets[bucket_id][0])

    # Basically, pad answer with zeros if shorter than buckets[bucket_id][1]
    length_answer = context_parsed["length_answer"]
    answer = sequence_parsed["answer"]
    answer = tf.sparse_tensor_to_dense(answer)
    answer = tf.reshape(answer, [-1])
    pad_answer = tf.zeros(shape=[buckets[bucket_id][0] - tf.cast(length_answer, tf.int32)], dtype=tf.int64)
    answer = tf.concat([answer, pad_answer], axis=0)
    answer.set_shape(buckets[bucket_id][1])

    # Shuffle queue
    return tf.train.shuffle_batch([question, answer],
                                  batch_size,
                                  capacity,
                                  min_after_dequeue)


def create_queues_for_bucket(batch_size, filename, buckets):
    """
    For every buckets, create a ShuffleQueue
    Then create a FIFOQueue on top of this queues (used for filtering queues)
    :param batch_size: int
    :param filename: str
    :param buckets: list
    :return:
    """
    shuffle_queues = []
    for bucket_id in range(len(buckets)):
        shuffle_queues.append(create_single_queue(bucket_id, filename, batch_size, buckets))

    capacity = 30 * batch_size

    # For every buckets, create a queue which return batch_size example
    # of that bucket
    all_queues, enqueue_ops = [], []
    for bucket_id in range(len(buckets)):
        queue = tf.FIFOQueue(capacity=capacity, dtypes=[tf.int64, tf.int64])
        all_queues.append(queue)

        enqueue_op = queue.enqueue(shuffle_queues[bucket_id])
        enqueue_ops.append(enqueue_op)
    return all_queues, enqueue_ops
