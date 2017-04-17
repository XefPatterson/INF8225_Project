import pickle
import tensorflow as tf
import numpy as np
from termcolor import cprint
import os
import time

import sys
sys.path.append(os.path.join('..', '..', 'Model'))
import model
from model import Seq2Seq
import utils

MODE = 'WORDS2WORDS' # WORDS2WORDS, CHARS2CHARS

if MODE == 'WORDS2WORDS':
    # WODS2WORDS
    flags = tf.app.flags
    flags.DEFINE_integer("nb_epochs", 100000, "Epoch to train [100 000]")
    flags.DEFINE_integer("save_frequency", 1800, "Output frequency")
    flags.DEFINE_integer("nb_iter_per_epoch", 250, "Output frequency")

    # Optimization
    flags.DEFINE_float("learning_rate", 0.0003, "Learning rate of for adam [0.0001")
    flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
    flags.DEFINE_integer("batch_size", 16, "The size of the batch [64]")

    # Vocabulary
    flags.DEFINE_integer("num_samples", 1024, "Number of samples for sampled softmax.")
    flags.DEFINE_integer("vocab_size_words", 8003, "The size of the word vocabulary [8003]")
    flags.DEFINE_integer("vocab_size_chars", 55, "The size of the char vocabulary [55]")
    flags.DEFINE_integer("is_char_level_encoder", False, "Is the encoder char level based")
    flags.DEFINE_integer("is_char_level_decoder", False, "Is the decoder char level based")

    flags.DEFINE_float("keep_prob", 0.75, "Dropout ratio [0.9]")
    flags.DEFINE_integer("num_layers", 3, "Num of layers [3]")
    flags.DEFINE_integer("hidden_size", 256, "Hidden size of RNN cell [256]")
    flags.DEFINE_integer("embedding_size", 128, "Symbol embedding size")
    flags.DEFINE_integer("use_attention", True, "Use attention mechanism?")
    flags.DEFINE_integer("valid_start", 0.98, "Validation set start ratio")

elif MODE == 'CHARS2CHARS':
    # CHARS2CHARS
    flags = tf.app.flags
    flags.DEFINE_integer("nb_epochs", 100000, "Epoch to train [100 000]")
    flags.DEFINE_integer("save_frequency", 1800, "Output frequency")
    flags.DEFINE_integer("nb_iter_per_epoch", 250, "Output frequency")

    # Optimization
    flags.DEFINE_float("learning_rate", 0.0003, "Learning rate of for adam [0.0001")
    flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
    flags.DEFINE_integer("batch_size", 16, "The size of the batch [64]")

    # Vocabulary
    flags.DEFINE_integer("num_samples", 1024, "Number of samples for sampled softmax.")
    flags.DEFINE_integer("vocab_size_words", 8003, "The size of the word vocabulary [8003]")
    flags.DEFINE_integer("vocab_size_chars", 55, "The size of the char vocabulary [55]")
    flags.DEFINE_integer("is_char_level_encoder", True, "Is the encoder char level based")
    flags.DEFINE_integer("is_char_level_decoder", True, "Is the decoder char level based")

    flags.DEFINE_float("keep_prob", 0.75, "Dropout ratio [0.9]")
    flags.DEFINE_integer("num_layers", 2, "Num of layers [3]")
    flags.DEFINE_integer("hidden_size", 512, "Hidden size of RNN cell [256]")
    flags.DEFINE_integer("embedding_size", 128, "Symbol embedding size")
    flags.DEFINE_integer("use_attention", True, "Use attention mechanism?")
    flags.DEFINE_integer("valid_start", 0.98, "Validation set start ratio")


FLAGS = flags.FLAGS
model.FLAGS = FLAGS

class seq2seq_chat(Seq2Seq):
    
    def __init__(self, buckets, forward_only):
        super(seq2seq_chat, self).__init__(buckets)
        
        # Load the idx_to_symbol dictionary
        if FLAGS.is_char_level_encoder : idx_to_symbol_File = 'idx_to_chars.pkl'
        else : idx_to_symbol_File = 'idx_to_words.pkl'
        with open(os.path.join('..', 'Data', 'MovieQA', idx_to_symbol_File), 'rb') as f:
            self.idx_to_symbol = pickle.load(f)

        # Load the symbol_to_idx dictionary
        if FLAGS.is_char_level_encoder : symbol_to_idx_File = 'chars_to_idx.pkl'
        else : symbol_to_idx_File = 'words_to_idx.pkl'
        with open(os.path.join('..', 'Data', 'MovieQA', symbol_to_idx_File), 'rb') as f:
            self.symbol_to_idx = pickle.load(f)

    def reply(self, question_string, session):
        print("PREDICTING")
        # User asks something...
        q = utils.encrypt_single(question_string, self.symbol_to_idx, words=not(FLAGS.is_char_level_encoder))
        a = utils.encrypt_single("", self.symbol_to_idx)


        # Equivalent to utils.get_batch but for one example
        #   Prepare the batch (batch_size = 1)
        buckets = [(25, 25)]
        bucket_id = 0
        q_pads = np.zeros([1, buckets[bucket_id][0]])
        a_pads = np.zeros([1, buckets[bucket_id][1]])

        q_pads[0][:q.shape[0]] = q
        a_pads[0][:a.shape[0]] = a

        # Processing
        out = self.forward_with_feed_dict(bucket_id=0, session=session, questions=q_pads, answers=a_pads, is_training=False)
        outputs = out["predictions"]
        
        #outputs, questions, answers = self.predict(bucket_id=0, session=session, questions=q_pads, answers=a_pads)

        # The model replies
        outputs = np.squeeze(outputs)
        outputs = np.argmax(outputs, axis=1)
        
        output_string = utils.decrypt_single(list(outputs), self.idx_to_symbol, words=not(FLAGS.is_char_level_decoder))

        """
        q_pads = np.squeeze(q_pads)
        q_string = utils.decrypt_single(list(q_pads), self.idx_to_symbol)

        a_pads = np.squeeze(a_pads)
        a_string = utils.decrypt_single(list(a_pads), self.idx_to_symbol)

        cprint("Q : " + q_string.split("<PAD>")[0], color="green")
        cprint("A : " + a_string.split("<PAD>")[0], color="yellow")
        """
        cprint("O : " + output_string, color="red")
        
        return output_string
