import itertools
from .cky_utils import *
from scipy import sparse as scisparse
from numba import cuda
from .cky_utils import *
from .treenode import Node, Rule, nodes_to_tree
import logging
class CKY_sampler:
    def __init__(self, K=0, D=0, max_len=40, gpu=False):
        assert D != 0 and K != 0, 'Sampler initialization error: K {}, D {}'.format(K, D)
        if gpu:
            _temp = __import__('pyculib', fromlist=['sparse', 'blas', 'rand'])
            self.cusparse_p = _temp.sparse
            self.blas_p = _temp.blas
            self.rand_p = _temp.rand
            self.kernels = __import__('kernels')
        # logging.info("sampler: getting K {} and D {}".format(K, D))
        self.K = K
        self.D = D
        self.lexis = None # Preterminal expansion part of the grammar (this will be dense)
        self.G = None     # Nonterminal expansion part of the grammar (usually be a sparse matrix representation)
        self.p0 = None
        self.gpu = gpu
        self.max_len = max_len
        self.num_points = max_len + 1
        self.chart = np.zeros((max_len+1, max_len+1), dtype=object) # split points = len + 1
        self.Q = self.calc_Q(self.K, self.D)
        # logging.info("sampler: getting Q {}".format(self.Q))
        self.this_sent_len = -1
        self.U = 0 # the random numbers used for sampling
        self.counter = 0
        if self.gpu:
            self.num_streams = 10
            self._init_streams()
            self.cusparse = self.cusparse_p.Sparse()
            self.cublas = self.blas_p.Blas()
            self.standard_scalar = cuda.device_array((1,), dtype=np.float32)
            self.standard_biscalar = cuda.device_array((2,), dtype=np.float32)
            self.U = cuda.device_array(((self.max_len - 1) * self.Q**2,), dtype=np.float32)
            self.random_generator = self.rand_p.PRNG(stream=self.streams[-1])

        self._init_chart()

    def set_models(self, G, p0, lexis):
        self.G = G
        self.p0 = p0
        self._scale_lexis(lexis)
        # print(self.scaler)
        self.lexis = lexis
        if self.gpu:
            self.lexis_cpu = self.lexis
            flat_lexis_cpu = self.lexis_cpu.ravel()
            self.lexis_flat = cuda.to_device(flat_lexis_cpu)
            self.lexis = self.lexis_flat.reshape(*self.lexis_cpu.shape)
            self.G_cpu = self.G
            self.G = self.cusparse_p.csr_matrix(self.G_cpu)
            self.p0_cpu = self.p0
            self.p0 = cuda.to_device(self.p0.reshape(1, -1))
        # print(self.p0_cpu)
        # print(self.G_cpu)
        # print(self.lexis_cpu)
        # exit()

    def _init_chart(self):
        # init the  temp arrays
        self.standard_q_array = np.zeros((self.Q,), dtype=np.float32)
        self.kron_vec_q2 = np.zeros((self.Q ** 2,), dtype=np.float32)
        self.dot_vec_q2 = np.zeros_like(self.kron_vec_q2)
        num_chart_cells = compute_decr_sum(self.num_points)
        self.num_ele_in_chart = self.Q * num_chart_cells
        self.incr_chart = np.zeros((self.Q, num_chart_cells), dtype=np.float32, order='F')
        # self.incr_chart_1d = np.zeros((self.num_ele_in_chart,), dtype=np.float32, order='F')

        # self.decr_chart_1d = np.zeros_like(self.incr_chart_1d)
        self.decr_chart = np.zeros_like(self.incr_chart)


        standard_q_zero_array = np.zeros((self.Q,), dtype=np.float32)
        if self.gpu:
            # send the temp arrays to GPU
            self.standard_q_array = cuda.to_device(self.standard_q_array)
            self.kron_vec_q2 = cuda.to_device(self.kron_vec_q2)
            self.dot_vec_q2 = cuda.to_device(self.dot_vec_q2)

            self.incr_chart = cuda.to_device(self.incr_chart)
            self.decr_chart = cuda.to_device(self.decr_chart)
            self.decr_chart_flat = self.decr_chart.ravel(order='F')
            # self.incr_chart = self.incr_chart_1d.reshape(self.Q, num_chart_cells)
            # self.decr_chart = self.decr_chart_1d.reshape(self.Q, num_chart_cells)

            self.shadow_chart = np.zeros_like(self.chart)
            self.incr_ptrs = np.zeros((self.num_points, self.num_points), dtype=object)
            self.decr_ptrs = np.zeros((self.num_points, self.num_points), dtype=object)
            incr_counter = 0
        for i in range(0, self.max_len+1):
            for j in range(i+1, self.max_len+1):
                if i < j:
                    if self.gpu:
                        self.chart[i, j] = self.incr_chart[:, incr_counter].reshape(self.Q)
                        self.shadow_chart[i, j] = self.decr_chart[:, compute_decr_sum(j) + i]
                        self.incr_ptrs[i, j] = self.incr_chart[:, compute_incr_sum(
                            self.num_points, self.num_points-i) : compute_incr_sum(
                            self.num_points, self.num_points-i) + j - i - 1]
                        self.decr_ptrs[i, j] = self.decr_chart[:, compute_decr_sum( j)+i+1 :
                        compute_decr_sum(j+1)]
                        # print('i:',i, 'j:', j, incr_counter, compute_decr_sum(j) + i, compute_incr_sum(
                        #     self.num_points, self.num_points-i), ':',
                        #       compute_incr_sum(self.num_points,
                        #                        self.num_points - i)
                        #       + j - i- 1, '||',
                        #       compute_decr_sum( j) + i + 1, ':', compute_decr_sum( j+1))
                        incr_counter += 1
                    else:
                        self.chart[i, j] = np.copy(standard_q_zero_array)
        if self.gpu:
            self.sync_all_streams()
        self.kron_vec_view_qq = self.kron_vec_q2.reshape((self.Q, self.Q))
        self.dot_vec_view_qq = self.dot_vec_q2.reshape((self.Q, self.Q))


    def _init_streams(self):
        if self.num_streams <= 1:
            return
        else:
            self.streams = []

            for i in range(self.num_streams):
                this_stream = cuda.stream()
                self.streams.append(this_stream)

            self.stream_generator = self._stream_gen()


    def inside_sample(self, sent):
        # print(sent)
        # there is a bug where the sampler is first loaded onto a worker, the vector views are
        # not linked to the original vectors, which causes the first sentence to underflow. it is
        #  not an issue here because the workers call init_chart directly, but in other cases
        # where the init_chart is called by master, it will have weird consequences

        success = 0
        tries = 0
        while not success:
            if tries > 10:
                raise Exception('overflowing/underflowing problem unsolvable!')
            tries += 1
            try:
                if self.gpu:
                    self.cublas.scal(0., self.decr_chart_flat)
                self.compute_inside(sent)

                # for i in range(0, len(sent)+1):
                #     for j in range(0, len(sent)+1):
                #         if type(self.chart[i,j]) is not int:
                #             if not self.gpu:
                #                 print(i, j, self.chart[i,j].sum())
                #             else:
                #                 print(i, j, np.sum(self.chart[i,j].copy_to_host()))
                # if np.sum(self.chart[i,j].copy_to_host()) == 0:
                #     print(sent[i], self.lexis_cpu[sent[i]])

                # if len(sent) == self.max_len:
                #     print('Sent')
                #
                #     for i in range(0, len(sent)+1):
                #         for j in range(0, len(sent)+1):
                #             if type(self.chart[i,j]) is not int:
                #                 if not self.gpu:
                #                     print(i, j, self.chart[i,j].sum())
                #                 else:
                #                     print(i, j, np.sum(self.chart[i,j].copy_to_host()))
                #                     if j == self.max_len and i == j - 2:
                #                         x1 = self.chart[i,i+1].copy_to_host()
                #                         x2 = self.chart[i+1, i+2].copy_to_host()
                #                         b = self.incr_ptrs[i, j]
                #                         c = self.decr_ptrs[i, j]
                #                         print(x1)
                #                         print(self.incr_ptrs[i, j].copy_to_host())
                #                         print(x2)
                #                         print(self.decr_ptrs[i, j].copy_to_host())
                #                         print(np.kron(x1, x2))
                #                         num_ijs = b.shape[1]
                #                         self.cublas.gemm('N', 'T', self.Q, self.Q, num_ijs, 1.0, c, b, 0.,
                #                                          self.dot_vec_view_qq)
                #                         print('gpu sum', self.dot_vec_view_qq.copy_to_host())
                #                         print('cpu dot', self.G_cpu.dot(np.kron(x1, x2)))
                #                         y = self.chart[i, j]
                #                         shadow_y = self.shadow_chart[i, j]
                #                         print('y', y.copy_to_host())
                #                         print('shadow', shadow_y.copy_to_host())
                #                 if np.sum(self.chart[i,j].copy_to_host()) == 0:
                #                     print(sent[i], self.lexis_cpu[sent[i]])
                # exit()
                nodes, logprob = self.sample_tree(sent)
                success = True
            except OverflowException as oe:
                rescaler = 1e-1
                logging.warning("overflow detected. curent scaler is {}, rescaler is {}. "
                                "number of tries {}".format(
                    self.scaler, rescaler, tries))
                self._init_chart() # flush the temp vectors and chart vectors
                self._scale_lexis(self.lexis, rescaler)
                logging.warning(oe)
            except UnderflowException as ue:
                rescaler = 1e1
                logging.warning("underflow detected. curent scaler is {}, rescaler is {}. "
                                "number of tries {}".format(
                    self.scaler, rescaler, tries))
                self._init_chart() # flush the temp vectors and chart vectors
                self._scale_lexis(self.lexis, rescaler)
                logging.warning(ue)
            except:
                raise
        self.this_sent_len = -1
        assert logprob < 0, 'weird logprob {}! {}'.format(logprob, nodes)
        this_tree, production_counter_dict, lr_branches = nodes_to_tree(nodes, sent)
        # print(this_tree)
        # print(self.counter)
        self.counter+=1
        return this_tree, logprob, production_counter_dict, lr_branches

    # @profile
    def compute_inside(self, sent): #sparse
        self.this_sent_len = len(sent)
        sent_len = self.this_sent_len
        if self.gpu:
            self.random_generator.uniform(self.U)

        num_points = len(sent) + 1
        # print('lex')
        for i in range(0, len(sent)):
            w = sent[i]
            if self.gpu:
                this_stream = next(self.stream_generator)

                self.kernels.get_gpu_mat_row(self.lexis_flat, w, self.Q, self.chart[i, i+1],
                                  self.shadow_chart[ \
                    i, i+1], this_stream)
                # assert np.array_equal(self.chart[i, i+1].copy_to_host(), self.lexis_cpu[w]), "{}, {}".format(self.chart[i, i+1].copy_to_host(), self.lexis_cpu[w])
            else:
                # logging.info(self.chart[i, i+1].shape)
                # logging.info(self.lexis[w].shape)
                # logging.info("{}, {}, {}, {}".format(self.Q, self.K, self.D, self.max_len))
                np.copyto(self.chart[i, i+1], self.lexis[w])

        if self.gpu:
            nnz = self.G.nnz
            self.sync_all_streams()
        # print('sync')
        kron_temp_vector = self.kron_vec_q2
        dot_temp_vector = self.dot_vec_q2
        kron_temp_vector_2d_view = self.kron_vec_view_qq
        for ij_diff in range(2, num_points):
            # LANE: If cell (i,j) should be zero, don't calculate things: fill it in with all zeros
            # print('ijdiff', ij_diff)
            for i in range(0, num_points):
                j = i + ij_diff
                if j >= num_points and i < j:
                    continue
                if self.gpu:
                    self.cublas.scal(0.0, dot_temp_vector)

                    y = self.chart[i, j]
                else:
                    dot_temp_vector.fill(0)
                if self.gpu:

                    b = self.incr_ptrs[i, j]
                    c = self.decr_ptrs[i, j]

                    num_ijs = b.shape[1]
                    self.cublas.gemm('N', 'T', self.Q, self.Q, num_ijs, 1.0, c, b, 0.,
                                     self.dot_vec_view_qq)

                else:
                    for k in range(i+1, j):
                        #kron sum
                            np.outer(self.chart[i,k], self.chart[k,j], out=kron_temp_vector_2d_view)
                            dot_temp_vector += kron_temp_vector

                if self.gpu:

                    y = self.chart[i, j]
                    shadow_y = self.shadow_chart[i, j]
                    self.cusparse.csrmv('N', self.Q, self.Q**2, self.G.nnz, 1.0,
                                        self.cusparse.matdescr(),
                                   self.G.data, self.G.indptr, self.G.indices,
                                   self.dot_vec_q2, 0., y)
                    self.cublas.axpy(1., y, shadow_y)

                    # if j - i == 2:
                    #     left_sum = np.sum(self.chart[i, i+1].copy_to_host())
                    #     right_sum = np.sum(self.chart[i+1, i+2].copy_to_host())
                    #     if np.sum(y.copy_to_host()) > left_sum and np.sum(y.copy_to_host()) > \
                    #             right_sum:
                    #         print('weird sum!!!')
                    #         print(self.chart[i, i+1].copy_to_host())
                    #         print(self.chart[i+1, i+2].copy_to_host())
                    #         print(y.copy_to_host())
                    #         exit()

                else:
                    y = self.G.dot(dot_temp_vector)

                    self.chart[i, j] = y

    # @profile
    def sample_tree(self, sent):
        expanding_nodes = []
        expanded_nodes = []
        # rules = []
        assert self.this_sent_len > 0, "must call inside pass first!"
        sent_len = self.this_sent_len
        topnode_pdf = self.chart[0, self.this_sent_len]
        logprob = 0

        # draw the top node
        if not self.gpu:
            p_topnode = (topnode_pdf * self.p0).astype(np.float64)
            norm_term = np.linalg.norm(p_topnode,1)
            logprob = np.log10(norm_term) - sent_len * np.log10(self.scaler)
            normed_p_topnode = p_topnode / norm_term
            top_A = np.random.multinomial(1, normed_p_topnode)
            A_cat = np.nonzero(top_A)[0][0]
        else:
            # print(tt.shape, self.p0.shape)
            # print(scisparse.dok_matrix(tt.reshape(1, -1)))
            tmp_gpu_a0_vec = self.standard_q_array
            # print(scisparse.dok_matrix(self.p0.copy_to_host().reshape(1, -1)))
            self.cublas.sbmv('L', self.Q, 0, 1.0, self.p0, topnode_pdf,  0., tmp_gpu_a0_vec)
            a0_vec = tmp_gpu_a0_vec.copy_to_host()[:self.K].astype(np.float64)
            norm_term = np.sum(a0_vec)
            # print('raw prob', sent_len, norm_term)
            logprob = np.log10(norm_term) - sent_len * np.log10(self.scaler)
            normed_a0_vec = a0_vec / norm_term
            # print(scisparse.dok_matrix(a0_vec.reshape(1, -1)))
            top_A = np.random.multinomial(1, normed_a0_vec)
            A_cat = np.nonzero(top_A)[0][0]
        if np.isnan(norm_term) or np.isinf(norm_term) or norm_term == 0:
            # for i in range(0, len(sent)+1):
            #     for j in range(0, len(sent)+1):
            #         if type(self.chart[i,j]) is not int:
            #             if not self.gpu:
            #                 print(i, j, self.chart[i,j].sum())
            #             else:
            #                 print(i, j, np.sum(self.chart[i,j].copy_to_host()))
                            # if np.sum(self.chart[i,j].copy_to_host()) == 0:
                            #     print(sent[i], self.lexis_cpu[sent[i]])
            totals = []
            for i in range(0, len(sent) + 1):
                for j in range(len(sent), 0, -1):
                    if self.gpu:
                        total = np.sum(self.chart[i,j].copy_to_host())
                    else:
                        total = self.chart[i,j].sum()
                    totals.append((i, j, total))
                    if not np.isnan(total) and not np.isinf(total) and total != 0:
                        if total < 1:
                            raise UnderflowException(totals)
                        else:
                            raise OverflowException(totals)
            # for i in range(0, len(sent)+1):
            #     for j in range(0, len(sent)+1):
            #         if type(self.chart[i,j]) is not int:
            #             if not self.gpu:
            #                 print(i, j, self.chart[i,j].sum())
            #             else:
            #                 print(i, j, np.sum(self.chart[i,j].copy_to_host()))
            #                 if np.sum(self.chart[i,j].copy_to_host()) == 0:
            #                     print(sent[i], self.lexis_cpu[sent[i]])
            # raise Exception('top node likelihood is {}!'.format(norm_term))
        # prepare the downward sampling pass
        top_node = Node(A_cat, 0, sent_len, self.D, self.K)
        if sent_len > 1:
            expanding_nodes.append(top_node)
        else:
            expanded_nodes.append(top_node)
        # rules.append(Rule(None, A_cat))
        temp_kron_vector = self.kron_vec_q2
        temp_multiply_vector = self.dot_vec_q2
        if self.gpu:
            temp_a_likelihood = self.standard_scalar
            temp_biscalar = self.standard_biscalar
        kron_temp_vector_2d_view = self.kron_vec_view_qq
        kth_node = -1

        while expanding_nodes:
            # print(sent_len, expanding_nodes)
            working_node = expanding_nodes.pop()

            k_dart = 1
            while 1 - k_dart < 1e-3:
                k_dart = np.random.random()

            if not self.gpu:
                a_likelihood = self.chart[working_node.i, working_node.j][working_node.cat]
                cur_G_row = self.G[working_node.cat]
            else:
                chart_vec = self.chart[working_node.i, working_node.j]

                a_likelihood = chart_vec.copy_to_host()[working_node.cat]


                start_ptr = self.G_cpu.indptr[working_node.cat]
                nnz = self.G_cpu.indptr[working_node.cat + 1] - start_ptr
                num_blocks_row = np.math.ceil(nnz / self.kernels.NUM_THREADS_PER_BLOCK)
            k_marginal = 0
            likelihoods = []
            for k in range(working_node.i + 1, working_node.j):
                if not self.gpu:
                    np.outer(self.chart[working_node.i, k], self.chart[k, working_node.j],
                             out=kron_temp_vector_2d_view)

                    joint_k_B_C = cur_G_row.multiply(temp_kron_vector)
                    joint_k_B_C = joint_k_B_C.astype(np.float64)
                    total_likelihood_k = np.sum(joint_k_B_C)
                    k_marginal += total_likelihood_k / a_likelihood
                else:

                    b_vec_view = self.chart[working_node.i, k]
                    c_vec_view = self.chart[k, working_node.j]
                    self.cublas.scal(0., temp_kron_vector)
                    self.cublas.ger(self.Q, self.Q, 1.0, c_vec_view, b_vec_view,
                                    kron_temp_vector_2d_view)

                    self.cublas.scal(0, temp_multiply_vector)
                    try:
                        self.kernels.pointwise_sp_mat[num_blocks_row,
                                                      self.kernels.NUM_THREADS_PER_BLOCK](
                            self.G.data,
                                                                                self.G.indices, nnz,
                                                                            start_ptr,
                                                                            temp_kron_vector, temp_multiply_vector)
                    except:
                        print(nnz, start_ptr, working_node, k, expanded_nodes)
                        raise
                    joint_k_B_C = temp_multiply_vector
                    total_likelihood_k = self.cublas.asum(joint_k_B_C)
                    likelihoods.append(total_likelihood_k)
                    k_marginal += total_likelihood_k / a_likelihood

                    # test1 = self.G_cpu[working_node.cat].toarray().flatten()
                    # print(scisparse.dok_matrix(test1.reshape(1, -1)))
                    # test2 = temp_kron_vector.copy_to_host().flatten()
                    # print(scisparse.dok_matrix(test2.reshape(1, -1)))
                    # print( test1 * test2)

                if k_marginal > k_dart:
                    # print(kth_node, k_dart, k_marginal, working_node.i, working_node.j, k)
                    kth_node += 1
                    if not self.gpu:
                        p_bc = joint_k_B_C / total_likelihood_k
                        bc = np.random.multinomial(1, p_bc.data)
                        cat_bc = p_bc.col[np.nonzero(bc)[0][0]]
                    else:
                        # test_sp = scisparse.dok_matrix(joint_k_B_C.copy_to_host().reshape(1, -1))
                        # print(test_sp)
                        # for row, col in test_sp.keys():
                        #     print(self.U[row*self.Q + col])
                        cat_bc, u = self.kernels.gumbel_draw_sample(joint_k_B_C, self.cublas,
                                                                    self.U,
                                                       kth_node, self.Q)
                    # print(cat_bc, scisparse.dok_matrix(u.copy_to_host().reshape(1, -1)))
                    b_cat = cat_bc // self.Q
                    c_cat = cat_bc % self.Q
                    # print(b_cat, c_cat, cat_bc)
                    expanded_nodes.append(working_node)
                    node_b = Node(b_cat, working_node.i, k, self.D, self.K, parent=working_node)
                    node_c = Node(c_cat, k, working_node.j, self.D, self.K, parent=working_node)
                    # print(node_b, node_c)
                    if node_b.d == self.D and node_b.j - node_b.i != 1:
                        print(node_b)
                        print(joint_k_B_C.copy_to_host()[cat_bc-2:cat_bc+3], u.copy_to_host()[
                            cat_bc-2:cat_bc+3])
                        raise Exception
                    if node_b.s != 0 and node_c.s != 1:
                        raise Exception("{}, {}".format(node_b, node_c))
                    if node_b.is_terminal():
                        expanded_nodes.append(node_b)
                        # rules.append(Rule(node_b.k, sent[working_node.i]))
                    else:
                        expanding_nodes.append(node_b)
                    if node_c.is_terminal():
                        expanded_nodes.append(node_c)
                        # rules.append(Rule(node_c.k, sent[k]))
                    else:
                        expanding_nodes.append(node_c)
                    # rules.append(Rule(working_node.cat, node_b.k, node_c.k))
                    break
            else:
                print('No split point is found for {} : {}'.format(working_node.i, working_node.j))
                print('Dart is {}, k marginal {}; a likelihood {}'.format(k_dart, k_marginal, a_likelihood))
                print(likelihoods)
                raise UnderflowException

        return expanded_nodes, logprob #, rules

    @staticmethod
    def calc_Q(K=0, D=0):
        if D == -1:
            return K
        return (D+1)*(K)*2

    def _stream_gen(self):
       yield from itertools.cycle(self.streams)

    def sync_all_streams(self):
        for stream in self.streams:
            stream.synchronize()

    def _scale_lexis(self, lexis, rescaler=0.):
        # median = np.min(np.max(lexis,axis=1))
        # median = 1 # remove this TODO
        if rescaler == 0:
            median = np.mean(lexis) * np.sqrt(np.mean(self.G.data)) * self.Q * 3
            if median == 0:
                self.scaler = 1
                median = 1
            lexis *= 1 / median
            self.scaler = 1 / median
        else:
            self.scaler *= rescaler
            if self.gpu:
                self.cublas.scal(rescaler, self.lexis_flat)
            else:
                self.lexis *= rescaler

class OverflowException(Exception):
    def __init__(self, *args):
        super(OverflowException, self).__init__(*args)

class UnderflowException(Exception):
    def __init__(self, *args):
        super(UnderflowException, self).__init__(*args)
