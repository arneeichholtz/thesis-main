import tensorflow_datasets as tfds
import logging
import tensorflow as tf
import numpy as np
import pickle
import re
from bitarray import bitarray

# prints occurences of different sequence lengths in the movie review databases
def print_seq_length(data):
    data_lst = list(data)
    count_1k, count_s_2k, count_l_2k = 0, 0, 0
    for example in data_lst:
        text = example['Source'].numpy()
        l = len(text)
        if l <= 1000:
            count_1k += 1
        elif l <= 2048:
            count_s_2k += 1
        elif l > 2048:
            count_l_2k += 1

    num_ex = len(data_lst)
    print("Num examples: ", num_ex)
    print(f"len <= 1000: {count_1k} -- {(count_1k/num_ex*100):.2f}%")
    print(f"1000 < len <= 2048: {count_s_2k} -- {(count_s_2k/num_ex*100):.2f}%")
    print(f"len > 2048: {count_l_2k} -- {(count_l_2k/num_ex*100):.2f}%")

def get_imdb_dataset():
    if sample_size != 1:
        percent = int(sample_size * 100)
        train_raw, test_raw = tfds.load('imdb_reviews', split=[f'train[:{percent}%]', f'test[:{percent}%]'])
        valid_raw = test_raw
    else:
        train_raw, test_raw = tfds.load('imdb_reviews', split=['train', 'test'])
        valid_raw = test_raw
  
    def adapt_example(example):
        return {'Source': example['text'], 'Target': example['label']}

    train = train_raw.map(adapt_example)
    valid = valid_raw.map(adapt_example)
    test = test_raw.map(adapt_example)

    return train, valid, test

def to_binary_old(source):
    try:
        str = source.numpy().decode('ascii', errors='ignore')
        bin_str = [format(x, 'b').zfill(8) for x in bytearray(str, 'utf-8')] # converts the text (type str) to 8-bit binary representation of the characters
        return tf.constant([[int(b) for b in [*s]] for s in bin_str]) # casts each bit to int, so shape: (seq_length, 8)
    except:
        pass

def to_binary(source):
    arr = bitarray()
    arr.frombytes(source.numpy()) # bitarray
    arr_2d = np.zeros(shape=(len(arr)//8, 8), dtype=np.int8) # 2d np array with bits -- storing bits rather than int8 is not possible
    
    for i in range(len(arr)//8):
       for j in range(8):
           arr_2d[i, j] = arr[8*i+j]
    return arr_2d

def tokenize(x):
    result = {
        'input': to_binary(x['Source'])[:max_length], # to_binary is called with str tensor of a single movie review, output is bitarray
        'target': x['Target']
    }
    return result

def get_bin_datasets(train, valid, test):
    train_bin = map(tokenize, list(train))
    valid_bin = map(tokenize, list(valid))
    test_bin = map(tokenize, list(test))
    return train_bin, valid_bin, test_bin

# max_length = 4000 # this value is used in Nystromformer paper and LRA
max_length = 2048 # this value is used in Sparse Sinkhorn Attention (and ~85% of examples have len < 2048)
sample_size = 1/100 # change to 1 for full dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":

    train, valid, test = get_imdb_dataset() # train, valid, test (type MapDataset) consists of tensor tuples with string and int -- valid and test are the same
    # print_seq_length(train)

    train_bin, valid_bin, test_bin = get_bin_datasets(train, valid, test)

    mapping = {"train":train_bin, "valid": valid_bin, "test":test_bin}
    for component in mapping:
        ds_list = []
        for idx, data_point in enumerate(iter(mapping[component])):
            # arr = np.array(data_point['input']) # this does the same as packbits
            padded = np.concatenate((data_point['input'], np.zeros(shape=(max_length - data_point['input'].shape[0], 8), dtype=np.int8)))
            ds_list.append({
                'input_bin': padded,
                'target': data_point['target'].numpy()
            })
            if idx % 100 == 0:
                print(f"{idx}\t\t", end = "\r")
        with open(f"text_bin8_2048_100.{component}.pickle", "wb") as f:
            pickle.dump(ds_list, f)



