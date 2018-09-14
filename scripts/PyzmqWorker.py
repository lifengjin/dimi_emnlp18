#!/usr/bin/env python3

import zmq
from PyzmqMessage import get_file_signature, resource_current, ModelWrapper, PyzmqJob, SentenceJob, CompileJob, CompletedJob, PyzmqParse, CompiledRow, SentenceRequest, RowRequest
import logging
import os.path
import pickle
from cky_sampler_inner import CKY_sampler
import sys

import tempfile

import os
import time
from uhhmm_io import printException, ParsingError

from WorkDistributerServer import get_local_ip
from collections import Counter, defaultdict


class PyzmqWorker:
    def __init__(self, host, jobs_port, results_port, models_port, maxLen, K=0, D=0, out_freq=1000,
                 tid=0,
                 gpu=False, batch_size=1, seed=0, level=logging.INFO):
        #Process.__init__(self)
        # logging.info("Thread created with id %s" % (threading.get_ident()))
        self.host = host
        self.jobs_port = jobs_port
        self.results_port = results_port
        self.models_port = models_port
        self.max_len = maxLen
        self.K = K
        self.D = D
        self.out_freq = out_freq
        self.tid = tid
        self.quit = False
        self.seed = seed
        self.debug_level = level
        self.model_file_sig = None
        self.indexer = None
        self.gpu = gpu
        self.batch_size = batch_size
        self.my_ip = get_local_ip()
        # logging.info('worker {} at init: D is {} and my K is {}'.format(self.tid, self.D, self.K))


    def __reduce__(self):
        return (PyzmqWorker, (self.host, self.jobs_port, self.results_port, self.models_port,
                              self.max_len, self.out_freq, self.tid, self.gpu, self.batch_size,
                              self.seed, self.debug_level), None)

    def run(self, gpu_num=None):
        if not gpu_num is None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
        logging.basicConfig(level=self.debug_level)
        # logging.info("Thread starting run() method with id=%s" % (threading.get_ident()))
        context = zmq.Context()
        models_socket = context.socket(zmq.REQ)
        url = "tcp://%s:%d" % (self.host, self.models_port)
        logging.debug("Connecting to models socket at url %s" % (url) )
        models_socket.connect(url)

        logging.debug("Worker %d connecting to work distribution server..." % self.tid)
        jobs_socket = context.socket(zmq.REQ)
        jobs_socket.connect("tcp://%s:%d" % (self.host, self.jobs_port))

        results_socket = context.socket(zmq.PUSH)
        results_socket.connect("tcp://%s:%d" % (self.host, self.results_port))

        logging.debug("Worker %d connected to all three endpoints" % self.tid)
        # logging.info('starting sampler: my D is {} and my K is {}'.format(self.D, self.K))
        self.cky_sampler = CKY_sampler(K=self.K, D=self.D, max_len=self.max_len, gpu=self.gpu)
        while True:
            if self.quit:
                break
            # logging.info("GPU for worker is %s" % self.gpu)
            #  Socket to talk to server
            logging.debug("Worker %d sending request for new model" % self.tid)
            models_socket.send(b'0')
            logging.debug("Worker %d waiting for new model location from server." % self.tid)
            msg = models_socket.recv_pyobj()
            logging.debug("Worker %d received new model location from server." % self.tid)
            # if self.gpu:
            #     msg = msg + '.gpu' # use gpu model for model
            # if self.gpu:
            #     logging.info('before get model')
            try:
                models, self.model_file_sig = self.get_model(msg)
            except Exception as e:

                printException()
                raise e
            # if self.gpu:
            #     logging.info('after get model')
            logging.debug("Worker %d preparing to process new model" % self.tid)


            logging.debug("Worker %d doing finite model inference" % (self.tid))
                    #print("Observation model type is %s" % (type(model_wrapper.model[0].lex)))
            self.cky_sampler.set_models(*models)
            self.processSentences(jobs_socket, results_socket)

            logging.debug("Worker %d received new models..." % self.tid)


        logging.debug("Worker %d disconnecting sockets and finishing up" % self.tid)
        jobs_socket.close()
        results_socket.close()
        models_socket.close()

    def processSentences(self, jobs_socket, results_socket):
        # print('5 init dynprog with batch_size %d and maxlen %d' % (self.batch_size, self.maxLen))

        sents_processed = 0

        if self.quit:
            return

        longest_time = 10
        sent_batch = []

        while True:
            logging.log(logging.DEBUG-1, "Worker %d waiting for job" % self.tid)
            try:
                # print(self.model_file_sig, self.batch_size)
                ret_val = jobs_socket.send_pyobj(SentenceRequest(self.model_file_sig, self.batch_size))
                jobs = jobs_socket.recv_pyobj()
                if self.batch_size == 1:
                    job = jobs
                else:
                    job = jobs[0]
            except Exception as e:
                ## Timeout in the job socket probably means we're done -- quit
                logging.info("Exception raised while waiting for sentence: %s" % (e) )
                self.quit = True
                break
            if job.type == PyzmqJob.SENTENCE:
                if self.batch_size > 1:
                    sent_index = job.resource.index
                    for job in jobs:
                        sentence_job = job.resource
                        sent = sentence_job.ev_seq
                        sent_batch.append(sent)
                else:
                    sentence_job = job.resource
                    sent_index = sentence_job.index
                    sent = sentence_job.ev_seq
                    sent_batch.append(sent)
                    # print(sent_index, sent)

            elif job.type == PyzmqJob.QUIT:
                logging.debug('Worker %d received signal from job server to check for new model' %
                          self.tid)
                epoch_done = True
                if len(sent_batch) == 0:
                    ## We got the epoch done signal with no sentences to process
                    break

            logging.log(logging.DEBUG-1, "Worker %d has received sentence %d" % (self.tid, sent_index))

            t0 = time.time()

            sent_samples = []
            log_probs = []
            productions = defaultdict(Counter)
            p0_counter = Counter()
            l_branch = 0
            r_branch = 0
            if True: # len(sent_batch) >= self.batch_size or epoch_done:
                #if self.batch_size > 1:
                    #logging.info("Batch now has %d sentences and size is %d so starting to process" % (len(sent_batch), self.batch_size) )

                success = False
                t0 = time.time()
                tries = 0
                #
                # for sent in sent_batch:
                #     sampled_tree, log_prob = self.cky_sampler.inside_sample(sent)
                #     sent_samples.append(sampled_tree)
                #     log_probs.append(log_prob)

                while not success and tries < 1:
                    # print('parsing now')
                    try:
                        if tries > 0:
                            logging.info("Error in previous sampling attempt. Retrying batch")
                        for sent in sent_batch:
                            # NLTK tree, large negative float (logprob of string given grammar; marginal prob of sentence), counter object (like a dictionary of counts) of production rules (how many times was each rule used in this sampled tree), a tuple of number of times a left-child is expanded vs number of times a right-child is expanded
                            sampled_tree, log_prob, this_productions_counters, lr_branches = \
                                self.cky_sampler.inside_sample(
                                sent)
                            # print(sampled_tree, log_prob)
                            sent_samples.append(sampled_tree)
                            log_probs.append(log_prob)
                            this_production_counter, this_p0_counter = this_productions_counters
                            for parent in this_production_counter:
                                productions[parent].update(this_production_counter[parent])
                            p0_counter.update(this_p0_counter)
                            l_branch += lr_branches[0]
                            r_branch += lr_branches[1]
                            # logging.debug('sentence {} is done'.format(sent_index))

                        success = True

                    except Exception as e:
                        logging.warning("Warning: Sentence %d had a parsing error %s." % (sent_index, e))
                        tries += 1
                        sent_sample = None
                        log_prob = 0
                        raise

                if (self.batch_size == 1 and sent_index % self.out_freq == 0) or (self.batch_size > 1 and (sent_index // self.out_freq != (sent_index+len(sent_batch)) // self.out_freq)):
                    logging.info("Processed sentence {0} (Worker {1})".format(sent_index, self.tid))

                t1 = time.time()

                if not success:
                    logging.info("Worker %d was unsuccessful in attempt to parse sentence %d" % (self.tid, sent_index) )

                if self.batch_size == 1 and (t1-t0) > longest_time:
                    longest_time = t1-t0
                    logging.warning("Sentence %d was my slowest sentence to parse so far at %d s" % (sent_index, longest_time) )

                # Send results back one-by-one
                for ind, sent_sample in enumerate(sent_samples):
                    if ind != len(sent_samples) -1 :
                        parse = PyzmqParse(sent_index + ind, sent_sample, log_probs[ind], success)
                    else:
                        parse = PyzmqParse(sent_index+ind, sent_sample, log_probs[ind], success,
                                       productions=(productions, p0_counter), lr_branches=(
                                l_branch, r_branch))
                    # print(sent_index+ind, log_probs[ind], self.batch_size)
                    sents_processed +=1
                    results_socket.send_pyobj(CompletedJob(PyzmqJob.SENTENCE, parse, parse.success))

                sent_batch = []

            if self.quit:
                break

            if log_prob > 0:
                logging.error('Sentence %d had positive log probability %f' % (sent_index, log_prob))

        logging.debug("Worker %d processed %d sentences this iteration" % (self.tid, sents_processed))

    def get_model(self, model_loc):
        ip = model_loc.ip_addr
        if ip == self.my_ip or ip.startswith('10.'):
            in_file = open(model_loc.file_path, 'rb')
            file_sig = get_file_signature(model_loc.file_path)
        else:
            dir = tempfile.mkdtemp()
            local_path = os.path.join(dir, os.path.basename(model_loc.file_path))
            logging.info("Model location is remote... ssh-ing into server to get model file %s and saving to %s" % (model_loc.file_path, local_path))
            os.system("scp -p %s:%s %s" % (model_loc.ip_addr, model_loc.file_path, local_path))
            in_file = open(local_path, 'rb')
            file_sig = get_file_signature(local_path)
        while True:
            try:
                model = pickle.load(in_file)
                assert len(model) == 3
                break
            except EOFError:
                logging.warning("EOF error encounter at model loading")
            time.sleep(5)

        in_file.close()
        return model, file_sig

    def handle_sigint(self, signum, frame):
        logging.info("Worker %d received interrupt signal... terminating immediately." % (self.tid))
        self.quit = True
        sys.exit(0)

    def handle_sigterm(self, signum, frame):
        logging.info("Worker %d received terminate signal... will terminate after cleaning up." % (self.tid))
        self.quit = True

    def handle_sigalarm(self, signum, frame):
        logging.warning("Worker %d received alarm while trying to process sentence... will raise exception" % (self.tid))
        raise ParsingError("Worker hung while parsing sentence")
