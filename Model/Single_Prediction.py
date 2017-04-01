import pickle
import tensorflow as tf
import numpy as np
from termcolor import cprint

import os
import model
import utils
import time

flags = tf.app.flags
flags.DEFINE_integer("nb_epochs", 100000, "Epoch to train [100 000]")
flags.DEFINE_integer("nb_iter_per_epoch", 100, "Epoch to train [100]")

flags.DEFINE_integer("out_frequency", 200, "Output frequency")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0001")
flags.DEFINE_float("decay_learning_rate_step", 10000, "Step to decay the learning rate [10000]")
flags.DEFINE_float("learning_rate_decay_factor", 0.96, "Learning rate decay [0.96]")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")

flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("vocab_size", 55, "The size of the vocabulary [64]")
flags.DEFINE_float("keep_prob", 0.9, "Dropout ratio [0.5]")

flags.DEFINE_integer("hidden_size", 256, "Hidden size of RNN cell [128]")
flags.DEFINE_integer("num_layers", 1, "Num of layers [1]")

FLAGS = flags.FLAGS
model.FLAGS = FLAGS

s1 = time.time()
# Load the idx_to_chars dictionary
with open(os.path.join('..', 'Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
    idx_to_chars = pickle.load(f)

# Load the chars_to_idx dictionary
with open(os.path.join('..', 'Data', 'MovieQA', 'chars_to_idx.pkl'), 'rb') as f:
    chars_to_idx = pickle.load(f)

# Instanciates and build the model for feedforward only
seq2seq = model.Seq2Seq(buckets=[(25, 25)], forward_only=True)
seq2seq.build()

# Restore the trained model's parameters from checkpoint file
sess = tf.Session()
saver, summary_writer = utils.restore(seq2seq, sess, save_name="model_saved/")

# ----------------------------------------------------------------------------------------------
print("PREDICTING")
# User asks something...
question_string = "Hello Jack!"
answer_string = "Hey Steve!"
q = utils.encrypt_single(question_string, chars_to_idx)
a = utils.encrypt_single(answer_string, chars_to_idx)

# Equivalent to utils.get_batch but for one example
#   Prepare the batch (batch_size = 1)
buckets = [(25, 25)]
bucket_id = 0
q_pads = np.zeros([1, buckets[bucket_id][0]])
a_pads = np.zeros([1, buckets[bucket_id][1]])

q_pads[0][:q.shape[0]] = q
a_pads[0][:a.shape[0]] = a

# Processing
outputs, questions, answers = seq2seq.predict(bucket_id=0, session=sess, questions=q_pads, answers=a_pads)

# The model replies
outputs = np.squeeze(outputs)
outputs = np.argmax(outputs, axis=1)
output_string = utils.decrypt_single(list(outputs), idx_to_chars)

questions = np.squeeze(questions)
q_string = utils.decrypt_single(list(questions), idx_to_chars)

answers = np.squeeze(answers)
a_string = utils.decrypt_single(list(answers), idx_to_chars)

cprint("Q : " + q_string.split("<PAD>")[0], color="green")
cprint("A : " + a_string.split("<PAD>")[0], color="yellow")
cprint("O : " + output_string, color="red")
print(int(time.time() - s1), " seconds")
