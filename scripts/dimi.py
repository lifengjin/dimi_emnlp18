#!/usr/bin/env python3.4

import copy
import logging
import os
import pickle
import signal
import sys
import copy
import torch
import multiprocessing
from .WorkDistributerServer import WorkDistributerServer
from .bounded_pcfg_model import Bounded_PCFG_Model, UnBounded_PCFG_Model
from .init_pcfg_strategies import *
from .pcfg_model import PCFG_model
from .pcfg_translator import *
from .workers import start_local_workers_with_distributer, start_cluster_workers
from .dimi_io import write_linetrees_file, read_gold_pcfg_file
from collections import Counter, defaultdict
# Has a state for every word in the corpus
# What's the state of the system at one Gibbs sampling iteration?
class Sample:
    def __init__(self):
        self.hid_seqs = []
        self.models = None
        self.log_prob = 0


def wrapped_sample_beam(*args, **kwargs):
    try:
        sample_beam(*args, **kwargs)
    except Exception as e:
        # print(e)
        logging.info('Sampling beam function has errored out!')
        raise e
        exit(0)

# This is the main entry point for this module.
# Arg 1: ev_seqs : a list of lists of integers, representing
# the EVidence SEQuenceS seen by the user (e.g., words in a sentence
# mapped to ints).
def sample_beam(ev_seqs, params, working_dir, gold_seqs=None,
                word_dict_file=None, word_vecs=None, resume=False):
    global K
    K = int(params.get('k'))
    sent_lens = list(map(len, ev_seqs))

    max_len = max(map(len, ev_seqs))
    # vocab_size = max(map(max, ev_seqs)) # vocab_size, which is the max index of the word indices

    f = open(word_dict_file, 'r', encoding='utf-8')
    word_dict = {}
    for line in f:
        (word, index) = line.rstrip().split(" ")
        word_dict[int(index)] = word

    vocab_size = len(word_dict)

    num_sents = len(ev_seqs)
    num_tokens = np.sum(sent_lens)

    num_samples = 0
    ## Set debug first so we can use it during config setting:
    debug = params.get('debug', 'INFO')
    logfile = params.get('logfile', 'log.txt')

    filehandler = logging.FileHandler(os.path.join(working_dir, 'log.txt'))
    streamhandler = logging.StreamHandler(sys.stdout)
    handler_list = [filehandler, streamhandler]
    logging.basicConfig(level=getattr(logging, debug), format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', handlers=handler_list)
    logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)

    D = int(params.get('d', 1))
    iters = int(params.get('iters'))
    try:
        num_cpu_workers = int(params.get('cpu_workers', 0))
    except ValueError as err:
        if params.get('cpu_workers', 0)=='auto':
            # import multiprocessing
            num_cpu_workers = multiprocessing.cpu_count()
        else:
            raise err
    num_gpu_workers = int(params.get('gpu_workers', 0))

    cluster_cmd = params.get('cluster_cmd', None)
    batch_per_update = min(num_sents, int(params.get('batch_per_update', num_sents)))
    batch_per_worker = min(num_sents, int(params.get('batch_per_worker', 1)))
    gpu = bool(int(params.get('gpu', 0)))
    if gpu and num_gpu_workers < 1 and num_cpu_workers > 0:
        logging.warning("Inconsistent config: gpu flag set with %d gpu workers; setting gpu=False"
                      % (num_gpu_workers))
        gpu = False

    resume_iter = int(params.get("resume_iter", -1))

    init_strategy = params.get("init_strategy", '')
    gold_pos_dict_file = params.get("gold_pos_dict_file", '')

    init_alpha = float(params.get("init_alpha", 1))
    # output settings:
    print_out_first_n_sents = int(params.get('first_n_sents', -1))

    if gold_pos_dict_file:
        gold_pos_dict = {}
        with open(gold_pos_dict_file) as g:
            for line in g:
                line = line.strip().split(' = ')
                gold_pos_dict[int(line[0])] = int(line[1])


    if (gold_seqs != None and 'num_gold_sents' in params):
        logging.info('Using gold tags for %s sentences.' % str(params['num_gold_sents']))

    seed = int(params.get('seed', -1))
    if seed > 0:
        logging.info("Using seed %d for random number generator." % (seed))
        np.random.seed(seed)
    else:
        logging.info("Using default seed for random number generator.")

    logging.info("Total number of tokens: {}, number of nodes: {}".format(sum(sent_lens),
                                                                          sum(sent_lens)*2))

    samples = []
    start_ind = 0
    end_ind = min(num_sents, batch_per_update)

    logging.info("Initializing state: K is {}; D is {}; MaxLen is {}".format(K, D, max_len))

    rnn_model_file = os.path.join(working_dir, 'rnn_model.pkl')

    pcfg_model = PCFG_model(K, D, vocab_size, num_sents, num_tokens, log_dir=working_dir,
                            word_dict_file=word_dict_file)
    pcfg_model.set_alpha(alpha=init_alpha)

    if D != -1:
        bounded_pcfg_model = Bounded_PCFG_Model(K, D)
    else:
        bounded_pcfg_model = UnBounded_PCFG_Model(K)

    word_dict = pcfg_model.word_dict
    # print(bounded_pcfg_model.K)

    if not resume:

        dnn_obs_model = None

        hid_seqs = [None] * len(ev_seqs)

        pcfg_model.start_logging()
        # initialization: a few controls:
        pcfg_replace_model(hid_seqs, ev_seqs, bounded_pcfg_model, pcfg_model)

        cur_iter = 0

    else:
        try:
            if resume_iter > 0:
                num_iter = resume_iter
            else:
                pcfg_runtime_stats = open(os.path.join(working_dir, 'pcfg_hypparams.txt'))
                num_iter = int(pcfg_runtime_stats.readlines()[-1].split('\t')[0])
            pcfg_model, dnn_obs_model = torch.load(open(os.path.join(working_dir, 'pcfg_model_'+str(
                num_iter)+'.pkl'), 'rb'))
        except:
            pcfg_model, dnn_obs_model = torch.load(open(os.path.join(working_dir, 'pcfg_model_'+str(
                num_iter-1)+'.pkl'), 'rb'))

        dnn_obs_model = None
        logging.info("Conitinuing from iteration {}".format(num_iter))
        pcfg_model.set_log_mode('a')
        pcfg_model.start_logging()

        pcfg_replace_model(None, None, bounded_pcfg_model, pcfg_model, resume=True, dnn=dnn_obs_model)

        hid_seqs = [None] * num_sents

        cur_iter = pcfg_model.iter
    workDistributer = WorkDistributerServer(ev_seqs, working_dir)
    logging.info("GPU is %s with %d workers and batch size %d" % (gpu, num_gpu_workers, batch_per_worker))
    logging.info("Start a new worker with python3 scripts/workers.py %s %d %d %d %d %d %d" % (
    workDistributer.host, workDistributer.jobs_port, workDistributer.results_port, workDistributer.models_port,
    max_len + 1, int(gpu), batch_per_worker))


    ## Initialize all the sub-processes with their input-output queues
    ## and dimensions of matrix they'll need
    logging.info("Starting workers")

    if num_cpu_workers + num_gpu_workers > 0:
        inf_procs = start_local_workers_with_distributer(workDistributer, max_len, num_cpu_workers, num_gpu_workers, gpu,
                                                         batch_per_worker, K=K, D=D)

    elif cluster_cmd != None:
        start_cluster_workers(workDistributer, cluster_cmd, max_len, gpu, K=K, D=D, batch_size=batch_per_worker)
    else:
        master_config_file = os.path.join(working_dir, 'masterConfig.txt')
        with open(master_config_file, 'w') as c:
            print({'host':workDistributer.host, 'jobs_port':workDistributer.jobs_port,
                   'results_port':workDistributer.results_port,
                    'models_port':workDistributer.models_port,
                   'max_len':max_len + 1,
                   'gpu':int(gpu),
                    'batch_size':batch_per_worker, 'K':K, 'D':D}, file=c)
            print('OK', file=c)

    signal.signal(signal.SIGINT, lambda x, y: handle_sigint(x, y, inf_procs, workDistributer))

    max_loglikelihood = -np.inf
    best_init_model = None
    best_anneal_model = None
    best_anneal_likelihood = -np.inf
    prev_anneal_coeff = -np.inf
    total_logprob = 0
    warming_period = False
    ### Start doing actual sampling:
    while cur_iter < iters:
        sent_list = []
        pcfg_model.iter = cur_iter
        logging.info('Parsing started Now. Sink loading the sentences.')

        if not sent_list:
            workDistributer.submitSentenceJobs(start_ind, end_ind)
        else:
            workDistributer.submitSentenceJobs(sent_index_list=sent_list)

        num_processed = 0
        parses = workDistributer.get_parses()
        # print(len(parses))
        logging.info("Parsing is done!")

        assert len(parses) == end_ind - start_ind or len(parses) == len(sent_list), 'wrong number of parses received!'

        total_logprobs = 0
        state_list = []
        state_indices = []
        productions = defaultdict(Counter)
        p0_counter = Counter()
        l_branches = 0
        r_branches = 0

        for parse in parses:
            num_processed += 1

            if parse.success:
                try:
                    state_list.append(parse.state_list)
                    state_indices.append(ev_seqs[parse.index])
                    if parse.productions is not None:
                        this_productions, this_p0 = parse.productions
                        for parent in this_productions:
                            productions[parent].update(this_productions[parent])

                        p0_counter.update(this_p0)
                        lr_branches = parse.lr_branches
                        l_branches += lr_branches[0]
                        r_branches += lr_branches[1]
                except:
                    logging.error('This parse is bad:')
                    logging.error('The sentence is ' + ' '.join([str(x) for x in ev_seqs[parse.index]]))
                    logging.error(' '.join([x.str() for x in parse.state_list]))
                    logging.error('The index is %d' % parse.index)
                    raise
                total_logprobs += parse.log_prob
            hid_seqs[parse.index] = parse.state_list
        pcfg_model.log_probs = total_logprobs
        pcfg_model.right_branching_tendency = r_branches / (l_branches + r_branches)
        logging.info("iter {} has a right branching tendency score of {:.2f}".format(cur_iter,
                                                                                 pcfg_model.right_branching_tendency))
        linetrees_fn = 'iter_' + str(cur_iter) + '.linetrees'
        full_fn = os.path.join(working_dir, linetrees_fn)
        if print_out_first_n_sents != -1:
            trees = hid_seqs[: print_out_first_n_sents]
        else:
            trees = hid_seqs
        hid_seqs = [None] * len(ev_seqs)
        if cur_iter % 100 == 0 and cur_iter != 0:
            pprint_bool = True
        else:
            pprint_bool = False
        p = multiprocessing.Process(target=write_linetrees_file, args=(trees,
                                                                       pcfg_model.word_dict,
                                                                       full_fn, pprint_bool))
        p.daemon = True
        p.start()


        logging.info("The log prob for this iter is {}".format(total_logprobs))
        pcfg_replace_model(hid_seqs, ev_seqs, bounded_pcfg_model, pcfg_model, dnn=dnn_obs_model,
                           productions=(productions, p0_counter))


        ## Update sentence indices for next batch:
        if batch_per_update < num_sents:
            if end_ind == len(ev_seqs):
                start_ind = 0
                end_ind = min(len(ev_seqs), batch_per_update)
            else:
                start_ind = end_ind
                end_ind = start_ind + min(len(ev_seqs), batch_per_update)
                if end_ind > num_sents:
                    end_ind = num_sents

        # if dnn_obs_model:
        #     with open(rnn_model_file, 'wb') as rfn:
        #         pickle.dump(dnn_obs_model, rfn)
        #
        cur_iter += 1
        p.join()

    logging.debug("Ending sampling")
    workDistributer.stop()

    for cur_proc in range(0, num_cpu_workers+num_gpu_workers):
        logging.info("Sending terminate signal to worker {} ...".format(cur_proc))
        inf_procs[cur_proc].terminate()

    for cur_proc in range(0, num_cpu_workers+num_gpu_workers):
        logging.info("Waiting to join worker {} ...".format(cur_proc))
        inf_procs[cur_proc].join()
        inf_procs[cur_proc] = None

    logging.info("Sampling complete.")
    # return samples


def handle_sigint(signum, frame, workers, work_server):
    logging.info("Master received quit signal... will terminate after cleaning up.")
    for ind, worker in enumerate(workers):
        logging.info("Terminating worker %d" % (ind))
        worker.terminate()
        logging.info("Joining worker %d" % (ind))
        worker.join(1)
    logging.info("Workers terminated successfully.")
    work_server.stop()
    logging.info("Master existing now.")
    raise SystemExit
