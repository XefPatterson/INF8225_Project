import tensorflow as tf
import numpy as np


def make_single_example(data):
    """
    Create a tf.train.SequenceExample for every pair of question-answer
    :param data: tuple
        A pair containing a question and an answer as lists of integer
    :return: tf.train.SequenceExample
    """
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


def create_tf_examples(buckets,
                       val_p=0.0,
                       test_p=0.0,
                       saved_stats_for_set=False):
    """
    Create {len(buckets) * 3} TFRecord files (3 for train, validation, test set)
    :param buckets: list
    :param val_p: float (default : 0.0)
        Percentage for validation set
    :param test_p: float (default : 0.0)
        Percentage for test set
    :param saved_stats_for_set: bool (default: False)
        Save number of example per file in a Python dictionary
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