import tensorflow as tf
import os
import pickle
import numpy as np
from termcolor import cprint
from flask import Flask, request
import requests

debug = False
buckets = [(50, 50), (100, 100), (150, 150), (300, 300)]
logdir = "Model/char2char_2x256_embed128"
model_name = os.path.join(logdir, "model.ckpt-64223.meta")
ACCESS_TOKEN = "EAAERfcfHLrIBAHfXCWfc8aOIi12HDlW24EuZArIVaVhrIuHQyKKxy4ZCtEElZCZCulgXOxnnNKaAHs6HRKeREM1qM9AZCb9SJZBfP3c20522xmBoXdf1LPb6BwA1Jad5ueXVE1V0ZBTpqLd6SQCmOtjHtdDa9XVza0UB5W8p1vKuQZDZD"

BOT_ID = 0
BOT_NAME = "louis"


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
        cprint("[*] Load graph... may be long", color="yellow")
        # tf.train.import_meta_graph('')
        tf.train.import_meta_graph(model_name)
        graph = tf.get_default_graph()

        # Restore Input Tensor
        cprint("[*] Create placeholder", color="yellow")
        self.is_training = graph.get_tensor_by_name("Placeholder:0")
        self.encoder_inputs = [graph.get_tensor_by_name('encoder{}:0'.format(i)) for i in
                               range(buckets[-1][0])]
        self.targets = [graph.get_tensor_by_name('decoder{}:0'.format(i)) for i in range(buckets[-1][0])]
        self.target_weights = graph.get_tensor_by_name('ones_like:0') + [
            graph.get_tensor_by_name('ones_like_{}:0'.format(i)) for i in
            range(1, buckets[-1][0])]

        # Restore Output Tensor
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

        cprint("[*] Init variables", color="yellow")
        self.sess = tf.InteractiveSession()
        last_saved_model = tf.train.latest_checkpoint(logdir)
        group_init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(group_init_ops)
        if last_saved_model is not None:
            tf.train.Saver().restore(self.sess, last_saved_model)
            cprint("[*] Was able to restore a model", color="green")
        else:
            cprint("[!] Failed to restore a model, CREATE AN NEW ONE!", color="red")

        cprint("[*] Ready", color="green")
        self.__load_file__()

    def __load_file__(self):
        cprint("[!] Load vocabulary", color="yellow")
        with open(os.path.join('Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
            self.idx_to_chars = pickle.load(f)

        # Load the chars_to_idx dictionary
        with open(os.path.join('Data', 'MovieQA', 'chars_to_idx.pkl'), 'rb') as f:
            self.chars_to_idx = pickle.load(f)

    def feed_new_sentence(self, sentence):
        """
        Feed a new sentences
        :param sentence: str
        :return:
        """
        len_sentence = len(sentence)
        if len_sentence > buckets[-1][0]:
            sentence = sentence[:buckets[-1][0]]
        bucket_id = 0
        while len_sentence > buckets[bucket_id][0]:
            bucket_id += 1

        # Transform
        q = encrypt_single(sentence, self.chars_to_idx)
        a = encrypt_single("", self.chars_to_idx)

        # Pad
        encoder_size, decoder_size = buckets[bucket_id]
        q_pads = np.zeros([1, encoder_size])
        a_pads = np.zeros([1, decoder_size])
        q_pads[0][:q.shape[0]] = q
        a_pads[0][:a.shape[0]] = a

        # Feed placeholder
        input_feed = {self.is_training: False}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = q_pads[:, l]
        for l in range(decoder_size):
            input_feed[self.targets[l].name] = a_pads[:, l]
            input_feed[self.target_weights[l].name] = np.not_equal(a_pads[:, l], 0).astype(np.float32)

        # Retrieve output
        output_feed = []
        for l in range(decoder_size):
            output_feed.append(self.outputs[bucket_id][l])

        # Run session
        outputs = self.sess.run(output_feed, input_feed)

        # Process answer from list to string
        outputs = np.squeeze(outputs)
        outputs = np.argmax(outputs, axis=1)
        output_string = decrypt_single(list(outputs), self.idx_to_chars)

        # Stop the sentences at <EOS>
        end_index = find_str(output_string, '<EOS>')
        print(output_string)
        if end_index == -1:
            return output_string
        else:
            return output_string[:end_index]


import fbmq
from fbmq import MessageType
import random

if debug:
    to_bucket = 2
    buckets = buckets[:to_bucket]
    model_name = "Model/char2char_1x256_embed30/model.ckpt-0.meta"

# g = GraphHandler()
g = None
timestamps = []

app = Flask(__name__)
page = fbmq.Page(ACCESS_TOKEN, BOT_ID, BOT_NAME)


@app.route('/', methods=['POST'])
def webhook():
    data = request.get_data(as_text=True)
    sender_id, next_bot_id = page.get_user_identity(data)
    page.handle_webhook(data, sender_id, next_bot_id)
    return "ok"


from pprint import pprint


@page.handle_echo
def message_handler(event, sender_id, next_bot_id):
    print("IN ECHO MESSAGE")
    pprint([[k, var] for k, var in event.__dict__.items()])
    sender_fb_id = event.sender_id
    # page.typing_on(sender_fb_id)

    print("Message from {} to {}".format(sender_id, next_bot_id))

    if sender_id == BOT_ID and next_bot_id != BOT_ID:
        print("Receive its ECHO, quit")
        return "ok"
    to_answer = (next_bot_id == BOT_ID)
    if event.is_text_message and not to_answer:
        if next_bot_id == MessageType.UNKNOWN_TURN:
            next_bot_id = hash(event.timestamp) % len(page.all_bots)
            print(next_bot_id)
            to_answer = (next_bot_id == BOT_ID)

        elif next_bot_id == MessageType.HUMAN_TURN:
            to_answer = False
            metadata = "{}-{}-{}".format(BOT_ID, BOT_NAME, MessageType.NOTIFY_HUMAN)
            page.send(sender_fb_id,
                      "{} is saying: Human it is your turn".format(BOT_NAME), metadata=metadata)

        elif next_bot_id == MessageType.NOTIFY_HUMAN:
            # Do nothing, it is human turn
            pass
    print("Gonna answer a message: {}".format(to_answer))
    if to_answer:
        next_bot_id = random.randint(0, len(page.all_bots))
        print("Number of bots {}".format(len(page.all_bots)))
        metadata = "{}~{}~{}".format(BOT_ID, BOT_NAME, next_bot_id)

        # Check if it is human turn to answer!
        if next_bot_id == len(page.all_bots):
            metadata = "{}~{}~{}".format(BOT_ID, BOT_NAME, MessageType.HUMAN_TURN)
        print("Metadata: {}".format(metadata))

        page.send(page.user_id,
                  "{} is saying: What's your favorite movie genre?. For {}".format(BOT_NAME, next_bot_id),
                  metadata=metadata)


@page.handle_message
def message_handler(event, sender_id, next_bot_id):
    print("IN HANDLE MESSAGE")
    pprint([[k, var] for k, var in event.__dict__.items()])

    sender_fb_id = event.sender_id
    message = event.message_text
    page.typing_on(sender_fb_id)

    print("Message from {} to {}".format(sender_id, next_bot_id))

    to_answer = (next_bot_id == BOT_ID)
    if event.is_text_message:
        if next_bot_id == MessageType.UNKNOWN_TURN:
            next_bot_id = hash(event.timestamp) % len(page.all_bots)
            # print(next_bot_id)
            to_answer = (next_bot_id == BOT_ID)

        elif next_bot_id == MessageType.HUMAN_TURN:
            to_answer = False
            metadata = "{}~{}~{}".format(BOT_ID, BOT_NAME, MessageType.NOTIFY_HUMAN)
            page.send(sender_fb_id,
                      "{} is saying: Human it is your turn".format(BOT_NAME), metadata=metadata)

        elif next_bot_id == MessageType.NOTIFY_HUMAN:
            # Do nothing, it is human turn
            pass
    print("Gonna answer a message: {}".format(to_answer))
    if to_answer:
        next_bot_id = random.randint(0, len(page.all_bots))
        print("Number of bots {}".format(len(page.all_bots)))
        metadata = "{}~{}~{}".format(BOT_ID, BOT_NAME, next_bot_id)
        # Check if it is human turn to answer!
        if next_bot_id == len(page.all_bots):
            metadata = "{}~{}~{}".format(BOT_ID, BOT_NAME, MessageType.HUMAN_TURN)
        print("Metadata: {}".format(metadata))
        page.send(page.user_id,
                  "{} is saying: What's your favorite movie genre?. For {}".format(BOT_NAME, next_bot_id),
                  metadata=metadata)


@page.after_send
def after_send(payload, response):
    """:type payload: fbmq.Payload"""
    print("complete")


if __name__ == '__main__':
    app.run(debug=False)
