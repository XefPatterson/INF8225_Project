import tensorflow as tf
import os
import pickle
import numpy as np
from termcolor import cprint
from flask import Flask, request
import requests

debug = False
buckets = [(50, 50), (100, 100), (150, 150), (300, 300)]
model_name = "Model/char2char_2x256_embed128/model.ckpt-64223.meta"
ACCESS_TOKEN = "EAAERfcfHLrIBAPjI6W8IO0L43wNhET7FZBp5ZA4HOJ2xrzm6xrKNwPYdt4IzD4flnDizcwjYqlLTyWE4KmAkzxDqc2oS0pkGja3sUUx4ZBLn6xLksmUmmaJ0PHWaCTaJX0k4NJ0SZAiCDdltJf86JWCvkBukWSljLshtU7wNpQZDZD"


def encrypt_single(string, symbol_to_idx):
    return np.array([symbol_to_idx.get(char, 1) for char in string.lower()])


def find_str(s, char):
    index = 0
    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index + len(char)] == char:
                    return index
            index += 1
    return -1


def decrypt_single(sentence, idx_to_symbol):
    return "".join([idx_to_symbol[idx] for idx in sentence])


class GraphHandler:
    def __init__(self):
        cprint("[!] Load graph, may be long", color="yellow")
        # tf.train.import_meta_graph('')
        tf.train.import_meta_graph(model_name)
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
        self.outputs[0] = [
            graph.get_tensor_by_name('seq2seq/model_with_buckets/seq2seq/embedding_rnn_seq2seq/cond/Merge:0')]
        for i in range(1, buckets[0][1]):
            self.outputs[0].append(graph.get_tensor_by_name(
                'seq2seq/model_with_buckets/seq2seq/embedding_rnn_seq2seq/cond/Merge_{}:0'.format(i)))

        for j in range(1, len(buckets)):
            self.outputs[j] = [graph.get_tensor_by_name(
                'seq2seq/model_with_buckets/seq2seq_{}/embedding_rnn_seq2seq/cond/Merge:0'.format(j))]
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

    def feed_new_sentence(self, sentence):
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
        print("Question ", q_pads)
        print("Answer ", a_pads)
        encoder_size, decoder_size = buckets[bucket_id]
        input_feed = {self.is_training: False}

        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = q_pads[:, l]

        # Same for decoder_input
        for l in range(decoder_size):
            input_feed[self.targets[l].name] = a_pads[:, l]
            # input_feed[self.target_weights[l].name] = np.not_equal(a_pads[:, l], 0).astype(np.float32)
            # break
        output_feed = []
        for l in range(decoder_size):
            output_feed.append(self.outputs[bucket_id][l])

        outputs = self.sess.run(output_feed, input_feed)
        outputs = np.squeeze(outputs)
        outputs = np.argmax(outputs, axis=1)
        output_string = decrypt_single(list(outputs), self.idx_to_chars)

        end_index = find_str(output_string, '<EOS>')
        if end_index == -1:
            return output_string
        else:
            return output_string[:end_index]


def reply(user_id, msg):
    data = {
        "recipient": {"id": user_id},
        "message": {"text": msg}
    }
    resp = requests.post("https://graph.facebook.com/v2.6/me/messages?access_token=" + ACCESS_TOKEN, json=data)
    print("Message send", resp.content)


app = Flask(__name__)
if debug:
    to_bucket = 2
    buckets = buckets[:to_bucket]
    model_name = "Model/char2char_1x256_embed30/model.ckpt-0.meta"

g = GraphHandler()

# g = None

timestamps = []


@app.route('/', methods=['POST'])
def handle_incoming_messages(graph=g):
    data = request.json
    sender = data['entry'][0]['messaging'][0]['sender']['id']
    print("Incoming message", data['entry'][0]['messaging'][0])
    # try:
    if data['entry'][0]['messaging'][0]["sender"]["id"] == "388754244807763":
        print("received self message")
        return "ok"
    if data['entry'][0]['messaging'][0]["timestamp"] in timestamps:
        return "ok"
    else:
        timestamps.append(data['entry'][0]['messaging'][0]["timestamp"])
    message = data['entry'][0]['messaging'][0]['message']['text']
    print("Text message:", message)
    out = graph.feed_new_sentence(sentence=message)
    reply(sender, out)
    # except:
    #     print("Error while receiving a message")
    return "ok"


@app.route('/', methods=['GET'])
def handle_verification():
    return request.args['hub.challenge']


if __name__ == '__main__':
    app.run(debug=False)
