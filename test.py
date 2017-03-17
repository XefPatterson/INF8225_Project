import numpy as np
import os
import cPickle
from tqdm import tqdm

"""
Similar code, I just change one hot vector with index value (smaller size in memory)
"""
chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         ' ', ',', '.', ':', ';', "'", '!', '?', '$', '%', '&', '(', ')', '=', '+', '-', '<EOS>']

chars_to_idx = {}
index = 0
for c in chars:
    chars_to_idx[c] = index
    index += 1

idx_to_chars = {}
for k, i in chars_to_idx.items():
    idx_to_chars[i] = k


def stringToOneHot(s, chars_to_idx, lower=True):
    if lower:
        s = s.lower()

    # Add an UNKNOWN char
    # Add the <EOS> at the end
    v_seq = np.zeros((len(s) + 1, len(chars_to_idx.keys()) + 1), dtype=np.float16)

    for i in range(len(s)):
        # Is s[i] a known character?
        try:
            v_seq[i, chars_to_idx[s[i]]] = 1.0
        # If not, then unknown = 1
        except KeyError:
            v_seq[i, -1] = 1.0

    v_seq[-1, chars_to_idx['<EOS>']] = 1.0
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
    line_to_one_hot[line[0]] = [np.argmax(v) for v in stringToOneHot(line[-1], chars_to_idx, lower=True)]
    len_sentences.append(len(line_to_one_hot[line[0]]))

qa_pairs = []
for conversation in tqdm(movie_convs_np, desc="Match pairs of Q-A"):
    subID = 0
    lines = eval(conversation[-1])
    while subID < (len(lines) - 1):
        qa_pairs.append((line_to_one_hot[lines[subID]], line_to_one_hot[lines[subID + 1]]))
        subID += 1

del line_to_one_hot

print("Saving file")
qa_pairs_pkl = os.path.join(movieQA_folder, 'QA_Pairs.pkl')
with open(qa_pairs_pkl, 'wb') as f:
    cPickle.dump(qa_pairs, f, protocol=cPickle.HIGHEST_PROTOCOL)
