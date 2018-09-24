import sys

import itertools

if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()

import scripts.dimi_io as io
import configparser
import scripts.dimi as dimi
import os
from random import randint, random
import time
import multiprocessing

def main(argv):
    if len(argv) < 1:
        sys.stderr.write("One required argument: <Config file|Resume directory>\n")
        sys.exit(-1)

    path = argv[0]
    D, K, init_alpha = 0, 0, 0
    if len(argv) == 3:
        D, K = argv[1], argv[2]
    elif len(argv) == 4:
        D, K, init_alpha = argv[1], argv[2], argv[3]
    if not os.path.exists(path):
        sys.stderr.write("Input file/dir does not exist!\n")
        sys.exit(-1)

    config = configparser.ConfigParser()
    input_seqs_file = None

    time.sleep(random() * 10)
    if os.path.isdir(path):
        ## Resume mode
        config.read(path + "/config.ini")
        out_dir = config.get('io', 'output_dir')
        resume = True
    else:
        config.read(argv[0])
        input_seqs_file = config.get('io', 'init_seqs', fallback=None)
        if not input_seqs_file is None:
            del config['io']['init_seqs']
        out_dir = config.get('io', 'output_dir')
        if not D and not K:
            D = config.get('params', 'd')
            K = config.get('params', 'k')
        if not init_alpha:
            init_alpha = config.get('params', 'init_alpha')
        init_alpha = str(float(init_alpha))
        config['params']['d'] = D
        config['params']['k'] = K
        if init_alpha:
            config['params']['init_alpha'] = init_alpha
        out_dir += '_D'+D+'K'+K+'A'+init_alpha

        counter = itertools.count()
        for i in counter:
            new_out_dir = out_dir + '_{}'.format(i)
            if not os.path.exists(new_out_dir):
                os.makedirs(new_out_dir)
                out_dir = new_out_dir
                config['io']['output_dir'] = out_dir
                sys.stderr.write("The output directory for this run is {}.\n".format(out_dir))
                break
        resume = False


        with open(out_dir + "/config.ini", 'w') as configfile:
            config.write(configfile)

    ## Write git hash of current branch to out directory
    os.system('git rev-parse HEAD > %s/git-rev.txt' % (out_dir))

    input_file = config.get('io', 'input_file')
    working_dir = config.get('io', 'working_dir', fallback=out_dir)
    dict_file = config.get('io', 'dict_file')

    ## Read in input file to get sequence for X
    (pos_seq, word_seq) = io.read_input_file(input_file)

    params = read_params(config)
    params['output_dir'] = out_dir

    ## Store tag sequences of gold tagged sentences
    gold_seq = dict()
    if 'num_gold_sents' in params and params['num_gold_sents'] == 'all':
        for i in range(0, len(pos_seq)):
            gold_seq[i] = pos_seq[i]
    else:
        while len(gold_seq) < int(params.get('num_gold_sents', 0)) and len(gold_seq) < len(word_seq):
            rand = randint(0, len(word_seq) - 1)
            if rand not in gold_seq.keys():
                gold_seq[rand] = pos_seq[rand]

    word_vecs = None
    if 'word_vecs_file' in params:
        #sys.stderr.write("This functionality is at alpha stage and disabled in master.\n")
        #sys.exit(-1)
        word_vecs = io.read_word_vector_file(params.get('word_vecs_file'), io.read_dict_file(dict_file))
    dimi.wrapped_sample_beam(word_seq, params, working_dir, gold_seqs=gold_seq,
                             word_vecs=word_vecs,
                             word_dict_file = dict_file, resume=resume)


def read_params(config):
    params = {}
    for (key, val) in config.items('io'):
        params[key] = val
    for (key, val) in config.items('params'):
        params[key] = val

    return params


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except:
        ctx = multiprocessing.get_start_method()
        print(ctx)
    main(sys.argv[1:])
