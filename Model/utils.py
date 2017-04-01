from termcolor import cprint
import tensorflow as tf
import numpy as np


def restore(model, session, save_name="model/"):
    """
    Retrieve last model saved if possible
    Create a main Saver object
    Create a SummaryWriter object
    Init variables
    :param save_name: string (default : model)
        Name of the model
    :return:
    """
    saver = tf.train.Saver(max_to_keep=2)
    # Try to restore an old model
    last_saved_model = tf.train.latest_checkpoint(save_name)

    group_init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(group_init_ops)
    summary_writer = tf.summary.FileWriter('logs/',
                                           graph=session.graph,
                                           flush_secs=20)
    if last_saved_model is not None:
        saver.restore(session, last_saved_model)
        cprint("[*] Restoring model  {}".format(last_saved_model), color="green")
    else:
        tf.train.global_step(session, model.global_step)
        cprint("[*] New model created", color="green")
    return saver, summary_writer


def get_random_bucket_id_pkl(bucket_sizes):
    odds = bucket_sizes / np.sum(bucket_sizes)
    bucket_id = np.argmax(np.random.multinomial(1, odds, 1))
    return bucket_id


def get_batch(data, buckets, bucket_id, batch_size):
    indices = np.random.choice(len(data[bucket_id]), size=batch_size)
    pairs = np.array(data[bucket_id])[indices]

    q_pads = np.zeros([batch_size, buckets[bucket_id][0]])
    a_pads = np.zeros([batch_size, buckets[bucket_id][1]])

    for i, (q, a) in enumerate(pairs):
        q_pads[i][:q.shape[0]] = q
        a_pads[i][:a.shape[0]] = a
    return q_pads, a_pads

# The argument idx_to_symbol can be both idx_to_char or idx_to_word dictionary
def decrypt(questions, answers, predictions, idx_to_symbol, batch_size=32, number_to_decrypt=4):
    index_to_decrypt = np.random.choice(range(batch_size), number_to_decrypt)

    predictions = [np.squeeze(prediction) for prediction in predictions]
    predictions = [np.argmax(prediction, axis=1) for prediction in predictions]

    for index in index_to_decrypt:
        question = "".join([idx_to_symbol[idx] for idx in questions[index, :]])
        true_answer = "".join([idx_to_symbol[idx] for idx in answers[index, :]])
        fake_answer = "".join([idx_to_symbol[prediction[index]] for prediction in predictions])

        cprint("Sample {}".format(index), color="yellow")
        cprint("Question: > {}".format(question), color="yellow")
        cprint("True answer: > {}".format(true_answer), color="green")
        cprint("Fake answer: > {}".format(fake_answer), color="red")


def decrypt_single(sentence, idx_to_symbol):
    return "".join([idx_to_symbol[idx] for idx in sentence])


def encrypt_single(string, symbol_to_idx):
    return np.array([symbol_to_idx[char] for char in string.lower()])
