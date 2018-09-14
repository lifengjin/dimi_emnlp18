import logging
import os.path
import gzip
import nltk
import numpy as np
import time
from scipy.stats import dirichlet
import collections
from .cky_utils import compute_Q
import pickle
import torch
import bidict

EPSILON = 1e-200

def normalize_a_tensor(tensor):
    return tensor / (np.sum(tensor, axis=-1, keepdims=True) + 1e-20)  # to supress zero division warning

class PCFG_model:
    def __init__(self, K, D, len_vocab, num_sents, num_words, log_dir='.', iter=0,
                 word_dict_file=None, autocorr_lags=(50,100)):
        self.autocorr_lags = autocorr_lags
        self.prev_models = collections.deque([], max(self.autocorr_lags))
        self.iter_autocorrs = []
        self.K = K
        self.K2 = self.K ** 2
        self.Q = compute_Q(K, D)
        self.len_vocab = len_vocab
        self.dist_size = self.K**2 + self.len_vocab # must use utils/make_ints_file
        self.num_sents = num_sents
        self.num_words = num_words
        self.non_root_nonterm_mask = []
        self.nonterms = list(range(self.K))
        self.alpha_range = []
        self.alpha = 0
        self.alpha_scale = 0
        self.nonterm_alpha, self.term_alpha = 0, 0
        self.alpha_array_flag = False
        self.counts = {}
        self.p0_counts = np.empty((1,))
        self.p0 = None
        self.right_branching_tendency = 0.0
        self.dnn_emit_likelihood = 0
        # self.init_counts()
        self.unannealed_dists = {}
        self.log_dir = log_dir
        self.log_mode = 'w'
        self.iter = iter
        self.hypparams_log_path = os.path.join(log_dir, 'pcfg_hypparams.txt')
        self.counts_log_path = os.path.join(log_dir, 'pcfg_counts_info.txt')
        self.word_dict = self._read_word_dict_file(word_dict_file)
        self.log_probs = 0
        self.annealed_counts = {}
        self.constraints = {}

    def set_log_mode(self, mode):
        self.log_mode = mode  # decides whether append to log or restart log

    def start_logging(self):

        self.counts_log = open(self.counts_log_path, self.log_mode)
        if self.log_mode == 'w':

            counts_header = ['iter', ]
            for lhs in self.nonterms:
                counts_header.append(str(lhs))

            self.counts_log.write('\t'.join(counts_header + counts_header[1:]) + '\n')
        self.hypparam_log = open(self.hypparams_log_path, self.log_mode)
        if self.log_mode == 'w':
            self.hypparam_log.write('iter\tlogprob\talpha\tac\tRB\tdnn_L\n')

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        state = self.__dict__.copy()

        del state['counts_log']
        del state['hypparam_log']
        return state

    def init_counts(self):
        self.counts = {k: np.zeros(self.dist_size) + self.alpha
                       for k in range(self.K)}
        self.p0_counts = np.zeros(self.Q)
        self.p0_counts[:self.K] = self.alpha

    def set_alpha(self, alpha=0.0):
        self.alpha, self.nonterm_alpha, self.term_alpha = alpha, alpha, alpha
        self.init_counts()

    def sample(self, pcfg_counts=None, p0_counts=None, annealing_coeff=1.0,
               sample_alpha_flag=False, resume=False, dnn=None):  # used as
        #  the normal sampling procedure
        # import pdb; pdb.set_trace()
        if not resume:
            # self.right_branching_tendency = right_branching_tendency
            self._reset_counts()
            self._update_counts(pcfg_counts, p0_counts) # Convert dict to numpy.array. This will be used later for sampling
        else:
            self.log_probs = 0
            self.iter += 1
        sampled_pcfg = self._sample_model(annealing_coeff, resume=resume, dnn=dnn)
        # self._calc_autocorr()
        sampled_pcfg = self._translate_model_to_pcfg(sampled_pcfg)
        # self.nonterm_log.flush()
        # self.term_log.flush()
        if self.log_probs != 0:
            self.hypparam_log.flush()
            self.counts_log.flush()
        return sampled_pcfg, self.p0

    def _reset_counts(self, use_alpha=True):
        self.p0_counts.fill(0)
        if use_alpha:
            self.p0_counts[:self.K] = 1
        for parent in self.counts:
            if use_alpha:
                self.p0_counts[:self.K] = self.alpha
                self.counts[parent].fill(self.alpha)
            else:
                self.counts[parent].fill(0)

    def _update_counts(self, pcfg_counts, p0_counts):
        self.nonterm_total_counts = {}
        self.nonterm_non_total_counts = {}
        # print(p0_counts)
        for k in p0_counts:
            self.p0_counts[int(k)] += p0_counts[k]
        for parent in pcfg_counts:
            lhs = int(parent.symbol())
            self.nonterm_total_counts[lhs] = sum(pcfg_counts[parent].values())
            self.nonterm_non_total_counts[lhs] = 0
            for children in pcfg_counts[parent]:
                if len(children) == 2:
                    rhs1, rhs2 = children
                    index = int(rhs1.symbol()) * self.K + int(rhs2.symbol())
                elif len(children) == 1:
                    term = children[0]
                    index = int(term) + self.K2
                else:
                    raise ValueError('Children more than 2 or less than 1.')
                self.counts[lhs][index] += pcfg_counts[parent][
                    children]
                if index < self.K2:
                    self.nonterm_non_total_counts[lhs] += pcfg_counts[parent][children]

    def _sample_model(self, annealing_coeff=1.0, resume=False, dnn=None):
        if not resume:
            self.save(dnn)
            logging.info("resample the pcfg model with nonterm alpha {}, term alpha {} and annealing "
                     "coeff {}.".format(self.nonterm_alpha, self.term_alpha, annealing_coeff))
        if self.log_probs != 0: # If we have not just initialized...

            self.hypparam_log.write('\t'.join([str(x) for x in [self.iter, self.log_probs,
                                                                (self.nonterm_alpha,
                                                                 self.term_alpha),
                                                                annealing_coeff,
                                                                self.right_branching_tendency,
                                                                self.dnn_emit_likelihood
                                                                ]])+'\n')

            self.counts_log.write('\t'.join([str(self.iter),] + [str(
                round(self.nonterm_total_counts.get(p, 0), 1)) for p in self.nonterms] + [
                str(self.nonterm_non_total_counts.get(x, 0)) for x in self.nonterms ]) +'\n')

        self.p0 = np.zeros_like(self.p0_counts)

        self.anneal_counts = self.counts
        self.unannealed_dists = {x: np.random.dirichlet(self.counts[x]) for x in self.counts}
        self.p0[:self.K] = np.random.dirichlet(self.p0_counts[:self.K])
        dists = self.unannealed_dists
        self.p0 = self.p0.astype(np.float32)
        # print(dists)
        return dists # This is G in Eq 26 of ACL 2018 submission

    def _translate_model_to_pcfg(self, dists):
        pcfg = {x: {} for x in dists}
        for parent in pcfg:
            # print(parent)
            for index, value in enumerate(dists[parent]):
                if index < self.K2:
                    rhs = (index // self.K, index % self.K)
                else:
                    rhs = index - self.K2
                pcfg[parent][rhs] = value
        return pcfg

    def get_current_pcfg(self):
        pcfg = self._translate_model_to_pcfg(self.unannealed_dists)
        return pcfg, self.p0

    def save(self, dnn):
        t0 = time.time()
        log_dir = self.log_dir
        save_model_fn = 'pcfg_model_' + str(self.iter) + '.pkl'
        past_three = os.path.join(log_dir, 'pcfg_model_' + str(self.iter - 3) + '.pkl')
        if os.path.exists(past_three) and (self.iter - 3) % 100:
            os.remove(past_three)
        this_f = os.path.join(log_dir, save_model_fn)
        torch.save((self,dnn), open(this_f, 'wb'))
        t1 = time.time()
        logging.info('Dumping out the pcfg model takes {:.3f} secs.'.format(t1 - t0))

    def _read_word_dict_file(self, word_dict_file):
        f = open(word_dict_file, 'r', encoding='utf-8')
        word_dict = bidict.bidict()
        for line in f:
            (word, index) = line.rstrip().split(" ")
            word_dict[int(index)] = word
        return word_dict
