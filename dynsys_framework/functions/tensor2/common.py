import numpy as np
# import cupy as cp
xp = np

def rct(x):
    return xp.maximum(x, 0)

def sdot(x, A):
    """
    Stream wise dot product.
    To each stream is combined with its respective matrix:
    l, j: \Sum_{i, k} w^{k=l,l}_{i,j} x^{k=l}_i = y^l_j .
    The size of source-stream and sink-stream stays the same (K=L).

    Example:
    x = [
    [[1, 3],[2, 4],[3, 5]]
    ]
    x.shape = (1,3,2)
    A = [
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]]
        ]
    A.shape = (3, 2, 2)
    tdot(x, A) = [[[6, 24],[12, 36]]] , shape = (1, 2, 2)
    :param x: stream, (1, input_vector_dim, stream_size)
    :param A: shortened weight of shape (input_vector_dim, output_vector_dim, stream_size)
    :return: stream of shape (1, output_vector_dim, stream_size)
    """
    d = xp.zeros((1, A.shape[1], A.shape[2]))
    for i in range(A.shape[0]):
        d[0, :, :] += x[0, i, :] * A[i, :, :]

    return d


def bdot(x, A):
    """
    Stream broadcasting.
    One source stream is transformed by multiple matricies, each forming a new sink stream.
    l, j: \Sum_{i, k} w^{k=1,l}_{i,j} x^{k=1}_i = y^l_j .

    x = [
    [[1],[2],[3]]
        ]
    x.shape = (1, 3, 1)
    A = [
    [[1,2],[2,3]],
    [[1,2],[2,3]]
    [[1,2],[2,3]]
        ]
    A.shape = (3, 2, 2)
    bdot(x, A) = [[[6, 12],[12, 18]]] , shape= (1, 2, 2)
    :param x: input of shape (1, input_vector_dim, 1)
    :param A: weights of shape (input_vector_dim, output_vector_dim, sink_stream_size)
    :return: of shape (1, output_vector_dim, sink_stream_size)
    """

    d = xp.zeros((1, A.shape[1], A.shape[2]))
    for i in range(A.shape[0]):
        d[0, :, :] += x[0, i, 0] * A[i, :, :]

    return d


def cadd(A, x):
    """
    Combination addition:
    Add stream into shortened weight, where the output_vector_dim matches source_stream_size.

    i, j, l, k=j : w^{k=j, l}_{i, j} + x^{k=j}_i = m^{k=j, l}_{i, j}

    This is usually used when the source stream scalary values are combined into one vector contained within one
    sink stream.

    x = [
    [[1, 3],[2, 4],[3, 5]]
    ]
    x.shape = (1,3,2)
    A = [
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]]
        ]
    A.shape = (3, 2, 2)
    cadd(A, x) = [
         [[2, 3], [5, 6]],
         [[3, 4], [6, 7]],
         [[4, 5], [7, 8]],
    ] , shape= (3, 2, 2)

    :param x: stream of shape (1, input_vector_dim, source_stream_size)
    :param A: shortened weight of shape (input_vector_dim, source_stream_size, sink_stream_size)
    :return: shortened weight of shape (input_vector_dim, source_stream_size, sink_stream_size)
    """

    return A + xp.expand_dims(x[0, :, :], axis=2)


def sadd(A, x):
    """
    Stream-wise addition:
    Each stream has a matrix to which the stream's vector is added.
    i, j, l: w^{k=l,l}_{i,j} + b^{l}_i

    x = [
    [[1, 3],[2, 4],[3, 5]]
    ]
    x.shape = (1,3,2)
    A = [
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]]
        ]
    A.shape = (3, 2, 2)
    sadd(A, x) = [
        [[2, 5],[3, 6]]
        [[3, 6],[4, 7]]
        [[4, 7],[5, 8]]
    ] , shape= (3, 2, 2)
    :param x: stream of shape = (1, input_vector_dim, stream_size)
    :param A: shortened weight, shape = (input_vector_dim, output_vector_dim, stream_size)
    :return: weight of shape = (input_vector_dim, output_vector_dim, stream_size)
    """

    return A + xp.swapaxes(x, 0, 1)


def vec_to_scalar_stream(x):
    return x.reshape((1, 1, -1))


def t_anti_reflexivity(A):
    """
    Diagonal between second and third dimension is set to zero.
    :param A: shape = (input_vector_dim, ?, sink_stream_size)
    :return: shape = (input_vector_dim, ?, sink_stream_size)
    """
    A_res = xp.zeros(A.shape) + A
    for i in range(A.shape[0]):
        A_res[i, :, i] = 0
    return A_res

#
# if __name__ == '__main__':
#     import time
#     _im = 4
#     _jm = 2
#     _km = 10000
#     _lm = 10000
#
#     ## TDOT - experiments
#     x = np.random.rand(_im * _km).reshape((1, _im, _km))
#     A = np.random.rand(_im*_jm*_km).reshape((_im, _jm, _km))
#
#     y_1 = np.zeros((1, _jm, _km))
#     start = time.time()
#     for k in range(_km):
#         y_1[:, :, k] = x[:, :, k].dot(A[:, :, k])
#     end = time.time()
#     print("tdot: time 1 {}".format(end - start))
#
#
#     y_2 = np.zeros((1, _jm, _km))
#     start = time.time()
#     for i in range(_im):
#         y_2[0, :, :] += x[0, i, :] * A[i, :, :]
#     end = time.time()
#     print("tdot: time 2 {}".format(end - start))
#
#     print("output difference: {}".format(np.max(y_1 - y_2)))
#
#
#     x = np.random.rand(_im * 1).reshape((1, _im, 1))
#     A = np.random.rand(_im*_jm*_km).reshape((_im, _jm, _km))
#     y_1 = np.zeros((1, _jm, _km))
#     start = time.time()
#     for k in range(_km):
#         y_1[:, :, k] = x[:, :, 0].dot(A[:, :, k])
#     end = time.time()
#     print("bdot: time 1 {}".format(end - start))
#
#     y_2 = np.zeros((1, _jm, _km))
#     start = time.time()
#     for i in range(_im):
#         y_2[0, :, :] += x[0, i, 0] * A[i, :, :]
#     end = time.time()
#     print("bdot: time 2 {}".format(end - start))
#
#     print("output difference: {}, y1{}, y2{}".format(np.max(y_1 - y_2), y_1.shape, y_2.shape))
