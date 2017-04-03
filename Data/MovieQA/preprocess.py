import numpy as np
import os
import cPickle
from tqdm import tqdm
from termcolor import cprint
import random
import nltk
import itertools
import pickle
import copy

bucket_lengths_chars = [(10,10), (25,25), (50,50), (100,100), (150,150)]
bucket_lengths_words = None # Defined empirically after being created


def create_buckets(qa_pairs_chars, qa_pairs_words):
    """
    Creates a dict of buckets of format bucket_id : list of tuples
    :param qa_pairs:
    :return: Dictionary of buckets
    """
    # Init buckets:
    chars_buckets = {}
    for i in range(len(bucket_lengths_chars)):
        chars_buckets[i] = []

    words_buckets = copy.deepcopy(chars_buckets)

    # Fill buckets :
    for i in tqdm(range(len(qa_pairs_chars)), desc="Creating buckets"):
        for j in range(len(bucket_lengths_chars)):
            # Q and A are shorter than bucket size
            if len(qa_pairs_chars[i][0]) <= bucket_lengths_chars[j][0] and \
               len(qa_pairs_chars[i][1]) <= bucket_lengths_chars[j][1]:
                chars_buckets[j].append(qa_pairs_chars[i])
                words_buckets[j].append(qa_pairs_words[i])
                break

    return chars_buckets, words_buckets

'''
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''
def get_id2line():
    lines=open(os.path.join('raw_data', 'movie_lines.txt')).read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line

'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''
def get_conversations():
    conv_lines = open(os.path.join('raw_data', 'movie_conversations.txt')).read().split('\n')
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    return convs

'''
    1. Get each conversation
    2. Get each line from conversation
    3. Save each conversation to file
'''
def extract_conversations(convs,id2line,path=''):
    idx = 0
    for conv in convs:
        f_conv = open(path + str(idx)+'.txt', 'w')
        for line_id in conv:
            f_conv.write(id2line[line_id])
            f_conv.write('\n')
        f_conv.close()
        idx += 1

'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''
def gather_dataset(convs, id2line):
    questions = []; answers = []

    for conv in convs:
        if len(conv) %2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i%2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers


"""
------------------------------------
PARSE CORNELL DATASET INTO CHARS
------------------------------------
"""


def parse_Cornwell_dataset_into_chars():
    chars = ['<PAD>', '<UNK>', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
             'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0',
             '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ',
             ',', '.', ':', ';', "'", '!', '?', '$', '%', '&',
             '(', ')', '=', '+', '-', '<EOS>']

    idx_to_chars = {i:c for i,c in enumerate(chars)}
    chars_to_idx = {c:i for i,c in enumerate(chars)}

    with open("idx_to_chars.pkl", 'wb') as f:
        cPickle.dump(idx_to_chars, f, protocol=cPickle.HIGHEST_PROTOCOL)

    with open("chars_to_idx.pkl", 'wb') as f:
        cPickle.dump(chars_to_idx, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def stringToIndices(s, chars_to_idx, lower=True):
        if lower:
            s = s.lower()

        v_seq = np.zeros(shape=(len(s) + 1), dtype=np.int32)
        for i in range(len(s)):
            v_seq[i] = chars_to_idx.get(s[i], 1)
        v_seq[-1] = chars_to_idx['<EOS>']
        return v_seq

    # PROCESS THE DATA
    id2line = get_id2line()
    convs = get_conversations()
    questions, answers = gather_dataset(convs, id2line)

    # change to lower case (just for en)
    questions = [stringToIndices(line, chars_to_idx) for line in questions]
    answers = [stringToIndices(line, chars_to_idx) for line in answers]

    qa_pairs_chars = zip(questions, answers)

    return qa_pairs_chars

"""
------------------------------------
PARSE CORNELL DATASET INTO WORDS
------------------------------------
"""


def parse_Cornwell_dataset_into_words():
    EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '  # space is included in whitelist
    EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

    limit = {
        'maxq': 30,
        'minq': 1,
        'maxa': 30,
        'mina': 1
    }

    UNK = 'unk'
    VOCAB_SIZE = 8000

    bucket_lengths_words = [(2, 2), (4, 4), (8, 8), (16, 16), (30, 30)]

    '''
     remove anything that isn't in the vocabulary
        return str(pure en)

    '''

    def filter_line(line, whitelist):
        return ''.join([ch for ch in line if ch in whitelist])

    '''
     filter too long and too short sequences
        return tuple( filtered_ta, filtered_en )

    '''

    def filter_data(qseq, aseq):
        filtered_q, filtered_a = [], []
        raw_data_len = len(qseq)

        assert len(qseq) == len(aseq)

        for i in range(raw_data_len):
            # qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
            # if qlen >= limit['minq'] and qlen <= limit['maxq']:
            #    if alen >= limit['mina'] and alen <= limit['maxa']:
            filtered_q.append(qseq[i])
            filtered_a.append(aseq[i])

        # print the fraction of the original data, filtered
        filt_data_len = len(filtered_q)
        filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
        print(str(filtered) + '% filtered from original data')

        return filtered_q, filtered_a

    '''
     read list of words, create index to word,
      word to index dictionaries
        return tuple( vocab->(word, count), idx2w, w2idx )

    '''

    def index_(tokenized_sentences, vocab_size):
        # get frequency distribution
        freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        # get vocabulary of 'vocab_size' most used words
        vocab = freq_dist.most_common(vocab_size)
        with open('words.pkl', 'wb') as f:
            pickle.dump(vocab, f)

        # Dictionnaries
        listOfWords = ['<PAD>'] + ['<UNK>'] + [x[0] for x in vocab] + ['<EOS>']
        idx_to_words = {i: w for i, w in enumerate(listOfWords)}
        words_to_idx = {w: i for i, w in enumerate(listOfWords)}
        return idx_to_words, words_to_idx

    '''
     filter based on number of unknowns (words not in vocabulary)
      filter out the worst sentences

    '''

    def filter_unk(qtokenized, atokenized, w2idx):
        data_len = len(qtokenized)

        filtered_q, filtered_a = [], []

        for qline, aline in zip(qtokenized, atokenized):
            unk_count_q = len([w for w in qline if w not in w2idx])
            unk_count_a = len([w for w in aline if w not in w2idx])
            if unk_count_a <= 2:
                if unk_count_q > 0:
                    if unk_count_q / len(qline) > 0.2:
                        pass
                filtered_q.append(qline)
                filtered_a.append(aline)

        # print the fraction of the original data, filtered
        filt_data_len = len(filtered_q)
        filtered = int((data_len - filt_data_len) * 100 / data_len)
        print(str(filtered) + '% filtered from original data')

        return filtered_q, filtered_a

    '''
     create the final dataset : 
      - convert list of items to arrays of indices
          return ( [array_en([indices]), array_ta([indices]) )
     
    '''

    def pack_together(qtokenized, atokenized, w2idx):
        # num of rows
        data_len = len(qtokenized)

        # lists to store indices
        idx_q = []
        idx_a = []

        for i in range(data_len):
            q_indices = convert_to_idx(qtokenized[i], w2idx)
            a_indices = convert_to_idx(atokenized[i], w2idx)

            # print(len(idx_q[i]), len(q_indices))
            # print(len(idx_a[i]), len(a_indices))
            idx_q.append(np.array(q_indices))
            idx_a.append(np.array(a_indices))

        # Makes a list of all the examples
        #   each element of this list is a tuple (question,answer)
        examples = zip(idx_q, idx_a)

        return examples

    '''
     replace words with indices in a sequence
      replace with unknown if word not in lookup
        return [list of indices]

    '''

    def convert_to_idx(seq, lookup):
        indices = []
        for word in seq:
            if word in lookup:
                indices.append(lookup[word])
            else:
                indices.append(lookup["<UNK>"])
        indices.append(lookup['<EOS>'])
        return indices

    # PROCESS THE DATA

    id2line = get_id2line()
    print('>> gathered id2line dictionary.\n')
    convs = get_conversations()
    print(convs[121:125])
    print('>> gathered conversations.\n')
    questions, answers = gather_dataset(convs, id2line)

    # change to lower case (just for en)
    questions = [line.lower() for line in questions]
    answers = [line.lower() for line in answers]

    # filter out unnecessary characters
    print('\n>> Filter lines')
    questions = [filter_line(line, EN_WHITELIST) for line in questions]
    answers = [filter_line(line, EN_WHITELIST) for line in answers]

    # filter out too long or too short sequences
    #print('\n>> 2nd layer of filtering')
    #qlines, alines = filter_data(questions, answers)

    #for q, a in zip(qlines[141:145], alines[141:145]):
    #    print('q : [{0}]; a : [{1}]'.format(q, a))

    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in questions]
    atokenized = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in answers]
    print('\n:: Sample from segmented list of words')

    for q, a in zip(qtokenized[141:145], atokenized[141:145]):
        print('q : [{0}]; a : [{1}]'.format(q, a))

    # indexing -> idx2w, w2idx 
    print('\n >> Index words')
    idx2w, w2idx = index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    # filter out sentences with too many unknowns
    #print('\n >> Filter Unknowns')
    #qtokenized, atokenized = filter_unk(qtokenized, atokenized, w2idx)
    #print('\n Final dataset len : ' + str(len(qtokenized)))

    print('\n >> Packing data up')
    qa_pairs_words = pack_together(qtokenized, atokenized, w2idx)

    # write to disk the indices_to_words dictionnary
    with open('idx_to_words.pkl', 'wb') as f:
        cPickle.dump(idx2w, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # write to disk the words_to_indices dictionnary
    with open('words_to_idx.pkl', 'wb') as f:
        cPickle.dump(w2idx, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return qa_pairs_words

"""
------------------------------------
MAIN
------------------------------------
"""

if __name__ == '__main__':
    cprint("Parsing Dataset", color="green")
    qa_pairs_words = parse_Cornwell_dataset_into_words()
    qa_pairs_chars = parse_Cornwell_dataset_into_chars()

    # Create buckets
    cprint("Creating buckets", color="green")
    qa_pairs_chars, qa_pairs_words = create_buckets(qa_pairs_chars, qa_pairs_words)

    # Calculate bucket_lengths_words
    cprint("Calculating bucket_lengths_words", color="green")
    bucket_lengths_words = []
    for bucket in qa_pairs_words.values():
        bucket_length = 0
        for pair in bucket:
            if max(len(pair[0]), len(pair[1])) > bucket_length:
                bucket_length = max(len(pair[0]), len(pair[1]))

        bucket_lengths_words.append((bucket_length, bucket_length))

    print
    bucket_lengths_words
    print
    bucket_lengths_chars

    """
    WORDS
    """
    # Save stats for buckets:
    bucket_sizes_words = []
    for k, v in qa_pairs_words.items():
        bucket_sizes_words.append(len(v))

    print('\n >> Save numpy arrays to pickle file')
    # save them
    with open('QA_Pairs_Words_Buckets.pkl', 'wb') as f:
        cPickle.dump({"qa_pairs": qa_pairs_words,
                      "bucket_sizes": bucket_sizes_words,
                      "bucket_lengths": bucket_lengths_words}, f, protocol=cPickle.HIGHEST_PROTOCOL)

    """
    CHARS
    """
    # Save stats for buckets:
    bucket_sizes_chars = []
    for k, v in qa_pairs_chars.items():
        bucket_sizes_chars.append(len(v))

    print("Saving file")
    with open('QA_Pairs_Chars_Buckets.pkl', 'wb') as f:
        cPickle.dump({"qa_pairs": qa_pairs_chars,
                      "bucket_sizes": bucket_sizes_chars,
                      "bucket_lengths": bucket_lengths_chars}, f, protocol=cPickle.HIGHEST_PROTOCOL)

