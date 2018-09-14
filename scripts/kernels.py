import math

import numpy as np
from numba import cuda, float32

from tmp.prefix_scan import _prefix_scan, _inc_scan

NUM_THREADS_PER_BLOCK = 128
MAX_IJK_NUM = 700 # around 52 word sentence
@cuda.jit
def get_mat_row(mat, row_index, Q, vec1, vec2):
    tx = cuda.grid(1)
    if tx < Q:
        valptr = row_index * Q + tx
        vec2[tx] = mat[valptr]
        vec1[tx] = mat[valptr]


@cuda.jit
def index_vec(vec, index, scalar):
    scalar[0] = vec[index]

@cuda.jit
def kron(vec1, Q, vec2, vec3):
    location = cuda.grid(1)
    vec_1_x = location // Q
    vec_2_x = location % Q
    if vec_1_x < Q:

        vec3[location] = vec1[vec_1_x] * vec2[vec_2_x]

@cuda.jit
def pointwise(vec1, Q, vec2, vec3):
    x = cuda.grid(1)
    if x >= Q:
        return
    vec3[x] = vec1[x] * vec2[x]

# @cuda.jit
# def pointwise_mat(mat1, mat_idx, Q, vec2, vec3):
#     x = cuda.grid(1)
#     if x >= Q:
#         return
#     vec3[x] = mat1[mat_idx, x] * vec2[x]

@cuda.jit
def pointwise_sp_mat(spmat_data, spmat_index, nnz, start_ptr, vec2, vec3):
    # vec3 must be zeroed first!
    tx = cuda.grid(1)
    if tx < nnz:
        loc = spmat_index[tx + start_ptr]
        val = spmat_data[tx + start_ptr]
        vec3[loc] = vec2[loc] * val

@cuda.jit
def get_nnz_sparse_row(spmat_indptr, row_index, nnz):
    nnz[0] = spmat_indptr[row_index+1] - spmat_indptr[row_index]
    nnz[1] = spmat_indptr[row_index]

@cuda.jit
def get_sparse_row(spmat_data, spmat_index, nnz, start_ptr, y_data, y_index):
    tx = cuda.grid(1)
    if tx < nnz:
        y_data[tx] = spmat_data[tx + start_ptr]
        y_index[tx] = spmat_index[tx + start_ptr]


def accumulate(in_vec, out_vec, n):
    assert n < NUM_THREADS_PER_BLOCK * 2 * NUM_THREADS_PER_BLOCK * 2, "the array for prefix sum " \
                                                                      "is too large."
    num_blocks = np.math.ceil(n / NUM_THREADS_PER_BLOCK)
    num_threads = NUM_THREADS_PER_BLOCK // 2
    sums = cuda.device_array(num_blocks, np.float32)
    sums_of_sums = cuda.device_array(num_blocks, np.float32)
    sums_of_sums_sums = cuda.device_array(num_blocks, np.float32)
    _prefix_scan[num_blocks, num_threads](in_vec, out_vec, sums, n)
    # print(out_vec.copy_to_host()[:130])
    num_blocks_sums = np.math.ceil(num_blocks / NUM_THREADS_PER_BLOCK)
    # print(num_blocks, num_threads)
    _prefix_scan[num_blocks_sums, num_threads](sums, sums_of_sums, sums_of_sums_sums, num_blocks)

    # print(sums_of_sums.copy_to_host()[-10:])
    if num_blocks > 1:
        _inc_scan[num_blocks-1, num_threads](out_vec, sums_of_sums, n)


@cuda.jit
def draw_sample(vec1, dart, target):
    n = vec1.shape[0]
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x

    shared_array = cuda.shared.array(128, float32)

    location = bx * bw + tx
    if location < n:
        shared_array[tx] = vec1[location]
    else:
        shared_array[tx] = 0
    cuda.syncthreads()
    if dart <= shared_array[tx]:
        if location == 0:
            target[0] = location
        elif tx == 0 and dart > vec1[location - 1]:
            target[0] = location
        elif tx != 0 and dart > shared_array[tx - 1]:
            target[0] = location

@cuda.jit
def n_minuslogu(vec1, U, Q2, kth):
    tx = cuda.grid(1)
    if tx < Q2:
        loc = kth * Q2 + tx
        n_hat = vec1[tx] / (- math.log(U[loc]))
        vec1[tx] = n_hat

@cuda.jit
def compute_kron_sum(chart, delta, num_ijks, Q, kron_mat):
    ijk_idx = cuda.threadIdx.x
    raw_BC = cuda.blockIdx.x
    num_blocks = cuda.blockDim.x

    reps = delta - 1
    i = ijk_idx // reps
    j = i + delta
    k = ijk_idx % reps + i + 1

    ijk_given_BC1 = cuda.shared.array(1024, float32)
    ijk_given_BC2 = cuda.shared.array(1024, float32)

    for sec in range(2):
        BC = sec * num_blocks + raw_BC
        B = BC // Q
        C = BC % Q

        res = chart[k-i, i, B] * chart[j-k, k, C]
        if sec == 0:
            ijk_given_BC1[ijk_idx] = res
        else:
            ijk_given_BC2[ijk_idx] = res

    cuda.syncthreads()

    if ijk_idx == 0:
        BC = ijk_idx * num_blocks + raw_BC
        i_num = num_ijks // reps
        i_given_BC = cuda.local.array(50, float32)
        for ii in range(i_num):
            i_given_BC[ii] = 0
        for ijk in range(num_ijks):
            this_i = ijk // reps
            i_given_BC[this_i] += ijk_given_BC1[ijk]
        for this_i in range(i_num):
            kron_mat[BC, this_i] = i_given_BC[this_i]
    elif ijk_idx == 1:
        BC = ijk_idx * num_blocks + raw_BC
        i_num = num_ijks // reps
        i_given_BC = cuda.local.array(50, float32)
        for ii in range(i_num):
            i_given_BC[ii] = 0
        for ijk in range(num_ijks):
            this_i = ijk // reps
            i_given_BC[this_i] += ijk_given_BC2[ijk]
        for this_i in range(i_num):
            kron_mat[BC, this_i] = i_given_BC[this_i]

# @profile
def gumbel_draw_sample(vec1, cublas, U, kth_node, Q):
    num_blocks = np.math.ceil(vec1.shape[0] / NUM_THREADS_PER_BLOCK)
    Q2 = Q**2
    n_minuslogu[num_blocks, NUM_THREADS_PER_BLOCK](vec1, U, Q2, kth_node)
    target = cublas.amax(vec1)
    return target, vec1

def get_gpu_mat_row(mat, w, Q, out_vec, out_vec2, stream):
    num_blocks = math.ceil(Q / NUM_THREADS_PER_BLOCK)
    get_mat_row[num_blocks, NUM_THREADS_PER_BLOCK, stream](mat, w, Q, out_vec, out_vec2)
    return True

