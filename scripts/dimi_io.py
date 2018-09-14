#!/usr/bin/env python3.4

import logging
import numpy as np
import os.path
import pickle
import shutil
import sys, linecache
import gzip

## This method reads a "tagwords" file which is space-delimited, and each
## token is formatted as POS/token.
def read_input_file(filename):
    pos_seqs = list()
    token_seqs = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        pos_seq = list()
        token_seq = list()
        for token in line.split():
            if "/" in token:
                (pos, token) = token.split("/")
            else:
                pos = 0

            pos_seq.append(int(pos))
            token_seq.append(int(token))

        pos_seqs.append(pos_seq)
        token_seqs.append(token_seq)

    return (pos_seqs, token_seqs)

def read_word_vector_file(filename, word_dict):
    f = open(filename, 'r', encoding='utf-8')
    dim = -1
    inv_dict = {}
    for key,val in word_dict.items():
        inv_dict[val] = key

    for line in f:
        parts = line.split()
        if len(parts) == 2:
            dim = int(parts[1])
            word_matrix = np.zeros((len(word_dict)+1, dim))
            print("Creating word matrix with size %d, %d" % (word_matrix.shape[0], word_matrix.shape[1]))
            continue

        if len(parts) > dim+1:
            logging.warn("Found line in vectors file with incompatible length %d" % len(parts))
            continue

        word = parts[0]
        if not word in inv_dict:
            continue

        word_ind = inv_dict[word]
        vec = []
        for ind in range(1,dim+1):
            try:
                vec.append(float(parts[ind]))
            except:
                print("Encountered a problem with line %s with length %d" % (line, len(parts)))
                raise Exception

        np_vec = np.array(vec, dtype='float16')
        word_matrix[word_ind] += np_vec

    f.close()
    return word_matrix

def read_sample_file(filename):
    pos_seqs = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        pos_seq = list()
        line = line.replace('[', '').replace(']', '').replace("'", "")
        for token in line.split(', '):
            (junk, pos) = token.split(':')
            pos_seq.append(int(pos))

        pos_seqs.append(pos_seq)

    f.close()
    return pos_seqs

def read_serialized_sample(pickle_filename):
    pickle_file = open(pickle_filename, 'rb')
    return pickle.load(pickle_file)

def write_serialized_models(model_list, pickle_file):
    pickle.dump(model_list, pickle_file)

def read_serialized_models(pickle_filename):
    pickle_file = open(pickle_filename, 'rb')
    return pickle.load(pickle_file)

def read_dict_file(dict_file):
    word_dict = dict()
    f = open(dict_file, 'r', encoding='utf-8')
    for line in f:
        #pdb.set_trace()
        (word, index) = line.rstrip().split(" ")
        word_dict[int(index)] = word

    return word_dict

def extract_pos(sample):
    pos_seqs = list()
    for sent_state in sample.hid_seqs:
        pos_seqs.append(list(map(lambda x: x.g, sent_state)))

    return pos_seqs

def printException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

class ParsingError(Exception):
    def __init__(self, cause):
        self.cause = cause

    def __str__(self):
        return self.cause

def write_linetrees_file(trees, word_dict, fn, pprint=False):
    with gzip.open(fn+'.gz', 'wt', encoding='utf8') as of:
        if pprint:
            pprint_fn = fn.replace('.linetrees', '.pptrees.gz')
            pprint_fh = gzip.open(pprint_fn, 'wt', encoding='utf8')
        index = 0
        for t in trees:
            if t is None:
                continue
            for lposition in t.treepositions(order='leaves'):
                t[lposition] = word_dict[int(t[lposition])]
            # print('Tree', str(index), file=of)
            print(t.pformat(margin=100000), file=of)
            if pprint:
                print('##############Tree', str(index), file=pprint_fh)
                t.pretty_print(stream=pprint_fh)
            index += 1
        if pprint:
            pprint_fh.close()

def read_gold_pcfg_file(pcfg_file, word_dict):
    seqs = []
    import nltk
    reverse_dict = {x[1] : x[0] for x in word_dict.items()}
    with open(pcfg_file) as gold_trees:
        for ttree in gold_trees:
            ttree = ttree.strip()
            nltktree = nltk.Tree.fromstring(ttree)
            for leaf in nltktree.treepositions(order='leaves'):
                nltktree[leaf] = str(reverse_dict[nltktree[leaf]])
            seqs.append(nltktree)
    return seqs