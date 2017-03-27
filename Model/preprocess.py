import numpy as np
import os
import cPickle
from tqdm import tqdm

bucket_lengths = [(10, 10), (25, 25), (50, 50), (100, 100), (150, 150)]


def create_buckets(qa_pairs):
    """
    Creates a dict of buckets of format bucket_id : list of tuples
    :param qa_pairs:
    :return: Dictionary of buckets
    """
    # Init buckets:
    buckets = {}
    for i in range(len(bucket_lengths)):
        buckets[i] = []

    # Fill buckets :
    for qa in tqdm(qa_pairs, desc="Creating buckets"):
        for i in range(len(bucket_lengths)):
            # Q and A are shorter than bucket size
            if len(qa[0]) <= bucket_lengths[i][0] and len(qa[1]) <= bucket_lengths[i][1]:
                buckets[i].append(qa)
                break

    return buckets


def parse_Cornwell_dataset():
    chars = ['<PAD>', '<UNK>', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
             'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0',
             '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ',
             ',', '.', ':', ';', "'", '!', '?', '$', '%', '&',
             '(', ')', '=', '+', '-', '<EOS>']

    chars_to_idx = {}
    index = 0
    for c in chars:
        chars_to_idx[c] = index
        index += 1

    idx_to_chars = {}
    for k, i in chars_to_idx.items():
        idx_to_chars[i] = k

    with open(os.path.join('Data', 'MovieQA', "idx_to_chars.pkl"), 'wb') as f:
        cPickle.dump(idx_to_chars, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def stringToIndices(s, chars_to_idx, lower=True):
        if lower:
            s = s.lower()

        v_seq = np.zeros(shape=(len(s)+1), dtype=np.int32)
        for i in range(len(s)):
            v_seq[i] = chars_to_idx.get(s[i], 1)
        v_seq[-1] = chars_to_idx['<EOS>']
        return v_seq

    movieQA_folder = os.path.join('Data', 'MovieQA')

    # Load text files into nupmy arrays;
    movie_convs_txt = os.path.join(movieQA_folder, 'movie_conversations.txt')
    movie_lines_txt = os.path.join(movieQA_folder, 'movie_lines.txt')

    movie_convs_np = np.loadtxt(movie_convs_txt, dtype='string', delimiter=' +++$+++ ', comments=None)
    movie_lines_np = np.loadtxt(movie_lines_txt, dtype='string', delimiter=' +++$+++ ', comments=None)

    line_to_one_hot = {}
    len_sentences = []
    for line in tqdm(movie_lines_np, desc="String to character index"):
        line_to_one_hot[line[0]] = stringToIndices(line[-1], chars_to_idx, lower=True)
        len_sentences.append(len(line_to_one_hot[line[0]]))

    qa_pairs = []
    for conversation in tqdm(movie_convs_np, desc="Match pairs of Q-A"):
        subID = 0
        lines = eval(conversation[-1])
        while subID < (len(lines) - 1):
            qa_pairs.append((line_to_one_hot[lines[subID]], line_to_one_hot[lines[subID + 1]]))
            subID += 1

    # Memory management
    del line_to_one_hot

    # Create buckets
    qa_pairs = create_buckets(qa_pairs)

    # Save stats for buckets:
    bucket_sizes = []
    for k, v in qa_pairs.items():
        bucket_sizes.append(len(v))

    print("Saving file")
    qa_pairs_pkl = os.path.join(movieQA_folder, 'QA_Pair_Buckets.pkl')
    with open(qa_pairs_pkl, 'wb') as f:
        cPickle.dump({"qa_pairs": qa_pairs,
                      "bucket_sizes": bucket_sizes,
                      "bucket_lengths": bucket_lengths}, f, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parse_Cornwell_dataset()
