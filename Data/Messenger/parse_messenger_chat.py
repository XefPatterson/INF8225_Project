"""
Prerequis
---------
Discussion facebook dans le folder: messages.htm

To run
------
git clone https://github.com/ownaginatious/fbchat-archive-parser.git
cd fbchat-archive-parser
python setup.py install
fbcap messages.htm -r -f json  > file.json

    -r: utile pour remplacer les ids par les vrais noms. Vu que c'est open source,
        je pense pas qu'on va se faire voler nos credentials

En plus
-------
Pour avoir des affichages des fichiers json lisibles
python -m json.tool file.json > pretty-file.json
        # python -m json.tool all_answers.json > all_answer2.json
        # python -m json.tool all_questions.json > all_question2.json
"""
import pickle
import numpy as np
import json
import copy
from datetime import datetime
from termcolor import cprint
from tqdm import tqdm

bucket_lengths_chars = [(50, 50), (100, 100), (150, 150), (300, 300), (500, 500)]
# Nom de l'utilisateur
main_user = "Louis-Henri Franc"
# Restreindre les réponses à main_user
restrict_answer_to_main_user = False


def parse_conversation(messages):
    questions, answers = [], []
    last_question = {
        "user": None,
        "message": None,
        "date": None
    }
    last_answer = {
        "user": None,
        "message": None,
        "date": None
    }

    def update(object, new_value):

        object["user"] = new_value["sender"]
        object["message"] = new_value["message"]

    previous_date = None
    for i, message in enumerate(messages["messages"]):
        # Make sure the last message and the current one have not more than 2 days of difference
        date = datetime.strptime(message["date"], "%Y-%m-%dT%H:%M-%S:00")

        # Init the first "question"
        if last_question["message"] is None:
            update(last_question, message)
        # Init the first "answer"
        elif last_answer["message"] is None and message["sender"] != last_question["user"]:
            update(last_answer, message)
        else:
            # Extend the first question while no other users are talking in the conversation
            if last_answer["message"] is None:
                last_question["message"] += ". " + message["message"]
            elif previous_date is not None and (date - previous_date).days > 22222:
                # Erase the conversation if the last two message have been send at different epoch
                last_question = copy.deepcopy(last_answer)
                update(last_answer, message)
            elif last_answer["user"] != message["sender"]:
                # last_question and last_answer now contains a owl sentences
                if (restrict_answer_to_main_user and main_user == last_answer["user"]) or (
                        not restrict_answer_to_main_user):
                    questions.append(last_question["message"])
                    answers.append(last_answer["message"])

                # Answer become a question
                last_question = copy.deepcopy(last_answer)
                update(last_answer, message)
            else:
                # Extend the answer
                last_answer["message"] += message["message"]
        # Remember the previous date
        previous_date = date
    return questions, answers


def parse_json():
    with open('fbchat-archive-parser/file.json') as data_file:
        data = json.load(data_file)
    all_questions, all_answers = [], []
    for messages in data["threads"]:
        q, a = parse_conversation(messages)
        all_questions.append(q)
        all_answers.append(a)

        json.dump(all_questions, open("all_questions.json", "w"))
        json.dump(all_answers, open("all_answers.json", "w"))


def parse_messenger_into_chars():
    chars = ['<PAD>', '<UNK>', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
             'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0',
             '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ',
             ',', '.', ':', ';', "'", '!', '?', '$', '%', '&',
             '(', ')', '=', '+', '-', '<EOS>']

    idx_to_chars = {i: c for i, c in enumerate(chars)}
    chars_to_idx = {c: i for i, c in enumerate(chars)}

    with open("idx_to_chars.pkl", 'wb') as f:
        pickle.dump(idx_to_chars, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("chars_to_idx.pkl", 'wb') as f:
        pickle.dump(chars_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    def stringToIndices(s, chars_to_idx, lower=True):
        if lower:
            s = s.lower()

        v_seq = np.zeros(shape=(len(s) + 1), dtype=np.int32)
        for i in range(len(s)):
            v_seq[i] = chars_to_idx.get(s[i], 1)
        v_seq[-1] = chars_to_idx['<EOS>']
        return v_seq

    # PROCESS THE DATA
    with open('fbchat-archive-parser/file.json') as data_file:
        data = json.load(data_file)

    all_questions, all_answers = [], []
    for messages in data["threads"]:
        q, a = parse_conversation(messages)
        all_questions.extend(q)
        all_answers.extend(a)

    # change to lower case (just for en)
    questions = [stringToIndices(line, chars_to_idx) for line in all_questions]
    answers = [stringToIndices(line, chars_to_idx) for line in all_answers]

    qa_pairs_chars = zip(questions, answers)

    return qa_pairs_chars


def create_buckets(qa_pairs_chars):
    """
    Creates a dict of buckets of format bucket_id : list of tuples
    :param qa_pairs:
    :return: Dictionary of buckets
    """
    # Init buckets:
    chars_buckets = {}
    for i in range(len(bucket_lengths_chars)):
        chars_buckets[i] = []

    # Fill buckets :
    for i in tqdm(range(len(qa_pairs_chars)), desc="Creating buckets"):
        for j in range(len(bucket_lengths_chars)):
            # Q and A are shorter than bucket size
            if len(qa_pairs_chars[i][0]) <= bucket_lengths_chars[j][0] and \
                            len(qa_pairs_chars[i][1]) <= bucket_lengths_chars[j][1]:
                chars_buckets[j].append(qa_pairs_chars[i])
                break

    return chars_buckets


if __name__ == '__main__':
    cprint("Parsing Dataset", color="green")
    qa_pairs_chars = parse_messenger_into_chars()
    qa_pairs_chars = create_buckets(qa_pairs_chars)

    bucket_sizes_chars = []
    for k, v in qa_pairs_chars.items():
        bucket_sizes_chars.append(len(v))

    print("Saving file")
    with open('QA_Pairs_Chars_Buckets.pkl', 'wb') as f:
        pickle.dump({"qa_pairs": qa_pairs_chars,
                     "bucket_sizes": bucket_sizes_chars,
                     "bucket_lengths": bucket_lengths_chars}, f, protocol=pickle.HIGHEST_PROTOCOL)
