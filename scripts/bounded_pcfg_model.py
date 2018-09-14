from scipy import sparse
import numpy as np
from .cky_utils import  *
import pickle

class Bounded_PCFG_Indexer:

    def __init__(self, D, K):
        self.K = K
        self.D = D
        self.D_pcfg = self.D + 1
        self.Q = compute_Q(K, D)
        self.lhs_dims = (2, self.D_pcfg, self.K)
        self.rhs_dims = (2, self.D_pcfg, self.K, 2, self.D_pcfg, self.K)


    def ravel_lhs_index(self, mutli_index):
        return np.ravel_multi_index(mutli_index, self.lhs_dims)

    def ravel_rhs_index(self, mutli_index):
        return np.ravel_multi_index(mutli_index, self.rhs_dims)

    def unravel_lhs_index(self, lhs_index):
        return np.unravel_index(lhs_index, self.lhs_dims)

    def unravel_rhs_index(self, rhs_index):
        return np.unravel_index(rhs_index, self.rhs_dims)

class UnBounded_PCFG_Model:
    def __init__(self, K):
        self.K = K
        self.D = -1
        self.K2 = self.K ** 2
        self.gammas = sparse.dok_matrix((self.K, self.K2), dtype=np.float32)
        self.raw_gammas = None
        self.p0 = np.zeros((self.K, ), dtype=np.float32)
        self.sparse_grammar = None

    def set_gammas(self, gammas):
        assert len(gammas) > 2
        # print(gammas)
        for lhs in gammas:
            lhs_sym = lhs
            for rhs, val in gammas[lhs].items():
                if isinstance(rhs, tuple):
                    rhs_1, rhs_2 = rhs[0], rhs[1]

                    lhs_index = lhs_sym
                    rhs_index = rhs_1 * self.K + rhs_2
                    # print(lhs, rhs, val,  lhs_mat, rhs_1_mat, rhs_2_mat, side, d)
                    # print(lhs, rhs, lhs, lhs_mat, rhs_1_mat, rhs_2_mat, side, d, val)
                    self.gammas[lhs_index, rhs_index] = val
        # print(self.gammas)
        self.sparse_grammar = self.gammas.tocsr()

    def set_lexis(self, pcfg_model):
        # num_words = pcfg_model.len_vocab
        # print(num_words)
        lexis = []
        for k in range(self.K):
            lexis.append(pcfg_model.unannealed_dists[k][self.K2:])
        lexis = np.vstack(lexis)
        lexis = lexis.T
        # print(lexis.shape)
        self.lexis = lexis.astype(np.float32)

    def set_p0(self, p0):
        # print(p0)
        self.p0 = p0

    def dump_out_models(self, fn):
        pickle.dump((self.sparse_grammar, self.p0, self.lexis), fn)


class Bounded_PCFG_Model:
    def __init__(self, K, D):
        self.K = K
        self.D = D
        self.Q = compute_Q(K, D)
        self.Q2 = self.Q ** 2
        self.K2 = self.K ** 2
        self.gammas = sparse.dok_matrix((self.Q, self.Q2), dtype=np.float32)
        self.raw_gammas = None
        self.p0 = np.zeros((self.Q, ), dtype=np.float32)
        self.indexer = Bounded_PCFG_Indexer(self.D, self.K)
        self.sparse_grammar = None
        # self.lexis_scale = self.Q
        # _, _, self.size, _ = compute_hd_size(self.D, self.K)

    def set_gammas(self, gammas):
        # side -> D -> K
        self.raw_gammas = gammas
        assert len(gammas ) == 2
        for side in range(2):
            for d in range(self.D):
                if gammas:
                    for lhs in gammas[side][d]:
                        lhs_sym = lhs
                        for rhs, val in gammas[side][d][lhs].items():
                            rhs_1, rhs_2 = rhs[0], rhs[1]
                            if side == 1:
                                depth = d + 1
                            else:
                                depth = d
                            lhs_index = self.indexer.ravel_lhs_index((side, d, lhs_sym))
                            rhs_index = self.indexer.ravel_rhs_index((0, depth, rhs_1, 1, d, rhs_2))
                            # print(lhs, rhs, val,  lhs_mat, rhs_1_mat, rhs_2_mat, side, d)
                            # print(lhs, rhs, lhs, lhs_mat, rhs_1_mat, rhs_2_mat, side, d, val)
                            self.gammas[lhs_index, rhs_index] = val
        self.sparse_grammar = self.gammas.tocsr()

    def set_lexis(self, pcfg_model):
        # num_words = pcfg_model.len_vocab
        # print(num_words)
        lexis = []
        for k in range(self.K):
            lexis.append(pcfg_model.unannealed_dists[k][self.K2:])
        lexis = np.vstack(lexis)
        lexis = lexis.T
        # print(lexis.shape)
        lexis = np.tile(lexis, (1, (self.D + 1) * 2))
        self.lexis = lexis.astype(np.float32)

    def set_p0(self, p0):
        self.p0 = p0

    def dump_out_models(self, fn):
        pickle.dump((self.sparse_grammar, self.p0, self.lexis), fn)
