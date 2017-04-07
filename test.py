import tensorflow as tf
import os
import pickle
import numpy as np
from termcolor import cprint

debug = False
buckets = [(50, 50), (100, 100), (150, 150), (300, 300)]

if debug:
    to_bucket = 2
    buckets = buckets[:to_bucket]

def encrypt_single(string, symbol_to_idx):
    return np.array([symbol_to_idx[char] for char in string.lower()])


def decrypt_single(sentence, idx_to_symbol):
    return "".join([idx_to_symbol[idx] for idx in sentence])


class GraphHandler:
    def __init__(self):
        cprint("[!] Load graph, may be long", color="yellow")
        saver = tf.train.import_meta_graph('Model/char2char_1x256_embed30/model.ckpt-0.meta')
        graph = tf.get_default_graph()

        # Input Tensor
        cprint("[!] Create placeholder", color="yellow")
        self.is_training = graph.get_tensor_by_name("Placeholder:0")
        self.encoder_inputs = [graph.get_tensor_by_name('encoder{}:0'.format(i)) for i in
                               range(buckets[-1][0])]
        self.targets = [graph.get_tensor_by_name('decoder{}:0'.format(i)) for i in range(buckets[-1][0])]
        self.target_weights = graph.get_tensor_by_name('ones_like:0') + [
            graph.get_tensor_by_name('ones_like_{}:0'.format(i)) for i in
            range(1, buckets[-1][0])]


        # Output Tensor
        self.outputs = [None] * len(buckets)
        self.outputs[0] = [graph.get_tensor_by_name('seq2seq/model_with_buckets/seq2seq/embedding_rnn_seq2seq/cond/Merge:0')]
        for i in range(1, buckets[0][1]):
            self.outputs[0].append(graph.get_tensor_by_name('seq2seq/model_with_buckets/seq2seq/embedding_rnn_seq2seq/cond/Merge_{}:0'.format(i)))

        for j in range(1, len(buckets)):
            self.outputs[j] = [graph.get_tensor_by_name('seq2seq/model_with_buckets/seq2seq_{}/embedding_rnn_seq2seq/cond/Merge:0'.format(j))]
            for i in range(1, buckets[j][1]):
                self.outputs[j].append(graph.get_tensor_by_name(
                               'seq2seq/model_with_buckets/seq2seq_{}/embedding_rnn_seq2seq/cond/Merge_{}:0'.format(j, i)))
        
        self.sess = tf.Session()
        cprint("[!] Init variables", color="yellow")
        self.sess.run(tf.global_variables_initializer())
        cprint("[!] Ready", color="green")
        self.__load_file__()

    def __load_file__(self):
        cprint("[!] Load vocabulary", color="yellow")
        with open(os.path.join('Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
            self.idx_to_chars = pickle.load(f)

        # Load the chars_to_idx dictionary
        with open(os.path.join('Data', 'MovieQA', 'chars_to_idx.pkl'), 'rb') as f:
            self.chars_to_idx = pickle.load(f)

    def feed_new_sentence(self, sentence="Bonjour commment ca va?"):
        len_sentence = len(sentence)
        if len_sentence > buckets[-1][0]:
            sentence = sentence[:buckets[-1][0]]
        bucket_id = 0
        while len_sentence > buckets[bucket_id][0]:
            bucket_id += 1

        q = encrypt_single(sentence, self.chars_to_idx)
        a = encrypt_single("", self.chars_to_idx)

        q_pads = np.zeros([1, buckets[bucket_id][0]])
        a_pads = np.zeros([1, buckets[bucket_id][1]])
        q_pads[0][:q.shape[0]] = q
        a_pads[0][:a.shape[0]] = a

        encoder_size, decoder_size = buckets[bucket_id]
        input_feed = {self.is_training: False}

        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = q_pads[:, l]

        # Same for decoder_input
        for l in range(decoder_size):
            input_feed[self.targets[l].name] = a_pads[:, l]
            input_feed[self.target_weights[l].name] = np.not_equal(a_pads[:, l], 0).astype(np.float32)

        output_feed = []
        for l in range(decoder_size):
            output_feed.append(self.outputs[bucket_id][l])

        outputs = self.sess.run(output_feed, input_feed)
        from IPython import embed
        embed()
        outputs = np.squeeze(outputs)
        outputs = np.argmax(outputs, axis=1)
        output_string = decrypt_single(list(outputs), self.idx_to_chars)
        print(output_string)
        return output_string


g = GraphHandler()
g.feed_new_sentence()
