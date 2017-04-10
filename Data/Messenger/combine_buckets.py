import os
import sys
import numpy as np
import cPickle as pickle

to_combine = ["QA_Pairs_Chars_Buckets_felix.pkl", "QA_Pairs_Chars_Buckets_julien.pkl"]
bucket_lengths_chars = [(50, 50), (100, 100), (150, 150), (300, 300), (500, 500)]

if __name__ == "__main__" :

    data = []

    for file_name in to_combine :
        with open(file_name, 'rb') as f:
            data.append(pickle.load(f))

    n_buckets = np.min([len(dataset['bucket_sizes']) for dataset in data])
    shorter_dataset = np.argmin([len(dataset['bucket_sizes']) for dataset in data])

    # Init
    bucket_sizes = data[shorter_dataset]['bucket_sizes']
    bucket_lengths = data[shorter_dataset]['bucket_lengths']
    buckets =  data[shorter_dataset]['qa_pairs']

    del data[shorter_dataset]

    for i in xrange(len(data)):
        for j in xrange(n_buckets):
            bucket_sizes[j] += data[i]['bucket_sizes'][j]
            buckets[j] += data[i]['qa_pairs'][j]

    with open("QA_Pairs_Chars_Buckets_FJ.pkl", 'wb') as f:
        pickle.dump({"qa_pairs": buckets,
                     "bucket_sizes": bucket_sizes,
                     "bucket_lengths": bucket_lengths}, f, protocol=pickle.HIGHEST_PROTOCOL)
