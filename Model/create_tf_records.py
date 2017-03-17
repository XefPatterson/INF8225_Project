import tensorflow as tf
import os
import pickle
from tqdm import tqdm
import numpy as np
from termcolor import cprint


def make_single_example(data):
    ex = tf.train.SequenceExample()

    # Sequential (with different sizes) features
    fl_question = ex.feature_lists.feature_list["question"]
    fl_answer = ex.feature_lists.feature_list["answer"]

    # Question sequence (list of int)
    for index in data[0]:
        fl_question.feature.add().int64_list.value.append(index)

    # Answer sequence (list of int)
    for index in data[1]:
        fl_answer.feature.add().int64_list.value.append(index)
    return ex


def create_tf_examples(val_p=0.1,
                       test_p=0.1):
    """
    Merge every example in three TFRecord files (train-test-validation set)
    :param val_p: float (default : 0.1)
        Percentage for validation set
    :param test_p: float (default : 0.1)
        Percentage for test set
    :return:
    """
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

    for data, name in zip([qa_pairs[train_indices], qa_pairs[valid_indices], qa_pairs[test_indices]],
                          ["train", "val", "test"]):

        # Create a TFRecordWriter
        writer = tf.python_io.TFRecordWriter(os.path.join(path_to_save_example, name + ".tfrecords"))

        # Iterate over all examples
        for example in tqdm(data, desc="Creating {} record file".format(name)):
            ex = make_single_example(example)
            writer.write(ex.SerializeToString())
        writer.close()


if __name__ == '__main__':
    create_tf_examples()
