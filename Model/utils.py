import matplotlib.pyplot as plt
from termcolor import cprint
import numpy as np


def get_random_bucket_id_pkl(bucket_sizes):
    # Fix problem.
    bucket_sizes = [float(bucket_size) for bucket_size in bucket_sizes]
    odds = bucket_sizes / np.sum(bucket_sizes)
    bucket_id = np.argmax(np.random.multinomial(1, odds, 1))
    return bucket_id


def get_batch(data, buckets, bucket_id, batch_size, indices=None):
    if indices is None:
        indices = np.random.choice(len(data[bucket_id]), size=batch_size)
    pairs = np.array(data[bucket_id])[indices]

    q_pads = np.zeros([batch_size, buckets[bucket_id][0]])
    a_pads = np.zeros([batch_size, buckets[bucket_id][1]])

    for i, (q, a) in enumerate(pairs):
        q_pads[i][:q.shape[0]] = q
        a_pads[i][:a.shape[0]] = a
    return q_pads, a_pads


def get_mix_batch(data_chars, data_words, buckets_char, buckets_words, is_char_encoder, is_char_decode, bucket_id,
                  batch_size):
    assert (len(data_chars[bucket_id]) == len(data_words[bucket_id])), "Different size between words and char dataset"
    indices = np.random.choice(len(data_chars[bucket_id]), size=batch_size)

    q_c, a_c = get_batch(data_chars, buckets_char, bucket_id, batch_size, indices)
    q_w, a_w = get_batch(data_words, buckets_words, bucket_id, batch_size, indices)

    return q_c if is_char_encoder else q_w, a_c if is_char_decode else a_w

def decrypt_single(sentence, idx_to_symbol, words=False):
    if words==True:
        decrypted = " ".join([idx_to_symbol[idx] for idx in sentence])
    else :
        decrypted = "".join([idx_to_symbol[idx] for idx in sentence])
    return decrypted

# The argument idx_to_symbol can be both idx_to_char or idx_to_word dictionary
def decrypt(questions, answers, predictions, idx_to_char, idx_to_word, batch_size,
            char_encoder=True, char_decoder=True, number_to_decrypt=4):
    index_to_decrypt = np.random.choice(range(batch_size), number_to_decrypt)
    predictions = [np.squeeze(prediction) for prediction in predictions]
    predictions = [np.argmax(prediction, axis=1) for prediction in predictions]
    predictions = np.transpose(np.asarray(predictions))

    for index in index_to_decrypt:
        if char_encoder:
            question = decrypt_single(questions[index], idx_to_char, words=False)
        else:
            question = decrypt_single(questions[index], idx_to_word, words=True)

        if char_decoder:
            true_answer = decrypt_single(answers[index], idx_to_char, words=False)
            fake_answer = decrypt_single(predictions[index], idx_to_char, words=False)
        else:
            true_answer = decrypt_single(answers[index], idx_to_word, words=True)
            fake_answer = decrypt_single(predictions[index], idx_to_word, words=True)

        cprint("Sample {}".format(index), color="yellow")
        cprint("Question: > {}".format(question), color="yellow")
        cprint("True answer: > {}".format(true_answer), color="green")
        cprint("Fake answer: > {}".format(fake_answer), color="red")


def plot_attention(questions, attentions, predictions, idx_to_char, idx_to_word, batch_size,
            char_encoder=True, char_decoder=True, nb_figures=4):
    fig, (tuples) = plt.subplots(1, nb_figures)
    for i in range(nb_figures):
        index = np.random.choice(range(batch_size), 1)[0]

        pred = [np.squeeze(prediction) for prediction in predictions]
        pred = [np.argmax(prediction, axis=1) for prediction in pred]

        if char_encoder:
            question = [idx_to_char[idx] for idx in questions[index, :]]
        else:
            question = [idx_to_word[idx]+" " for idx in questions[index, :]]

        if char_decoder:
            answer = [idx_to_char[prediction[index]] for prediction in pred]
        else:
            answer = [idx_to_word[prediction[index]]+" " for prediction in pred]

        # List of attention given the encoder inputs
        attention = [att[index] for att in attentions]

        # per rows: question attention
        # per cols: answer generation
        data = np.stack(attention, axis=1)
        tuples[i].imshow(data, vmin=0, vmax=1, cmap='Greys', interpolation="none")
        tuples[i].set_xticks(range(len(answer)))
        tuples[i].set_xticklabels(answer, fontsize='xx-small')

        tuples[i].set_yticks(range(len(question)), minor=False)
        tuples[i].set_yticklabels(question, fontsize='xx-small')
    plt.axis('off')
    #plt.savefig("attention_matrix.png")
    fig.savefig("attention_matrix.png", dpi=300)
    plt.close()

def encrypt_single(string, symbol_to_idx):
    return np.array([symbol_to_idx[char] for char in string.lower()])
