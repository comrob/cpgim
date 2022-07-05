import numpy as np
# import cupy as cp
xp = np

"""
3 dim tensor logic.
Usually ~ rule of thumb of tensor shapes:

SIGNAL TYPE (the forwarded variable):
Signals have dimension A and can be generated by B different sources and each modified for C different streams.
Thus it has shape (B, A, C). If B > 1 it is refered to as multi-source.
Usually (B, A) or deep (B, A, 1) are used for signals that broadcast to multiple streams.
However!, the outputs of functions have shapes (A, B, C).

WEIGHT TYPE (the learned variable):
Weights combine the components of A dimensional signal which can have B different sources modified
for C different streams.
Thus it has shape (A, B, C). 

(Since this mini manifesto was written after writing the code this is not the rule.)
"""


def tdot(x, A):
    """
    Tensor dot product.
    Dot product of first two dimensions for all #streams (which equals to number of cpgs).
    x = [
    [[1, 3],[2, 4],[3, 5]]
    ]
    x.shape = (1,3,2) (input vector of size 3 comming from(going to) 2 streams)
    A = [
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]]
        ]
    A.shape = (3, 2, 2)
    tdot(x, A) = [[[6, 24],[12, 36]]] , shape= (1, 2, 2)
    :param x: input of shape (1, input_size, #streams)
    :param A: weights of shape (input_size, output_size, #streams)
    :return: of shape (1, output_size, #streams)
    """
    input_size = A.shape[0]
    output_size = A.shape[1]
    num_of_streams = A.shape[2]
    d = xp.zeros((1, output_size, num_of_streams))
    # for k in range(num_of_streams):
    #     for j in range(output_size):
    #         for i in range(input_size):
    #             d[0, j, k] += x[0, i, k] * A[i, j, k]

    for k in range(num_of_streams):
        d[:, :, k] = x[:, :, k].dot(A[:, :, k])

    return d


def bdot(x, A):
    """
    Broadcasted dot product.
    Dot product of a vector x that is broadcested to matricies of A #streams-times.
    x = [
    [1,2,3]
        ]
    x.shape = (1, 3)
    A = [
    [[1,2],[2,3]],
    [[1,2],[2,3]]
    [[1,2],[2,3]]
        ]
    A.shape = (3, 2, 2)
    bdot(x, A) = [[[6, 12],[12, 18]]] , shape= (1, 2, 2)
    :param x: input of shape (1, input_size)
    :param A: weights of shape (input_size, output_size, #streams)
    :return: of shape (1, output_size, #streams)
    """
    input_size = A.shape[0]
    output_size = A.shape[1]
    num_of_streams = A.shape[2]
    d = xp.zeros((1, output_size, num_of_streams))
    # for k in range(num_of_streams):
    #     for j in range(output_size):
    #         for i in range(input_size):
    #             d[0, j, k] += x[0, i] * A[i, j, k]

    for k in range(num_of_streams):
        d[:, :, k] = x[:, :].dot(A[:, :, k])
    return d


def tadd(x, A):
    """
    Addition of mult-vector and mult-parametric-matrix.
    x = [
    [[1, 3],[2, 4],[3, 5]]
    ]
    x.shape = (1,3,2) (input vector of size 3 coming from(going to) 2 streams)
    A = [
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]]
        ]
    A.shape = (3, 2, 2)
    tadd(x, A) = [
        [[2, 5],[3, 6]]
        [[3, 6],[4, 7]]
        [[4, 7],[5, 8]]
    ] , shape= (3, 2, 2)
    :param x: shape = (1, input_size, #streams)
    :param A: shape = (input_size, output_size, #streams)
    :return: shape = (input_size, output_size, #streams)
    """
    input_size = A.shape[0]
    output_size = A.shape[1]
    num_of_streams = A.shape[2]
    d = xp.zeros((input_size, output_size, num_of_streams))
    # for k in range(num_of_streams):
    #     for j in range(output_size):
    #         for i in range(input_size):
    #             d[i, j, k] = x[0, i, k] + A[i, j, k]

    for k in range(num_of_streams):
        d[:, :, k] = x[:, :, k].T + A[:, :, k]
    return d


def badd(x, A):
    """
    Addition of vector to mult-parametric-matrix.
    x = [
    [1,2,3]
    ]
    x.shape = (1,3) (input vector of size 3 coming from(going to) 2 streams)
    A = [
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]],
         [[1, 2], [2, 3]]
        ]
    A.shape = (3, 2, 2)
    badd(x, A) = [
        [[2, 3],[3, 4]]
        [[3, 4],[4, 5]]
        [[4, 5],[5, 6]]
    ] , shape= (3, 2, 2)
    :param x: shape = (1, input_size) or (output_size, input_size) in case of multi-source or
    (output_size, input_size, 1) in case of deep multi-source.
    :param A: shape = (input_size, output_size, #streams)
    :return: shape = (input_size, output_size, #streams)
    """
    input_size = A.shape[0]
    output_size = A.shape[1]
    num_of_streams = A.shape[2]
    d = xp.zeros((input_size, output_size, num_of_streams))
    # for k in range(num_of_streams):
    #     for j in range(output_size):
    #         for i in range(input_size):
    #             d[i, j, k] = x[0, i] + A[i, j, k]
    _x = x.T
    for k in range(num_of_streams):
        d[:, :, k] = _x + A[:, :, k]
    return d


def unstream(A):
    """
        Transposes first and third dim in matrix. Usually for SIGNAL TYPE where we change stream to source and v.v.
        A = [
             [[1, 2, 4], [2, 3, 4], [3, 4, 4], [4, 5, 4]],
            ]
        A.shape = (1, 4, 3)
        ttrans(A) = [
                 [[1], [2], [3], [4]],
                 [[2], [3], [4], [5]],
                 [[4], [4], [4], [4]]
                ], shape = (3, 4, 1)
        :param A: shape = (#target, input_size, #source)
        :return: shape = (#source, input_size, #target)
        """
    # target_num = A.shape[0]
    # input_size = A.shape[1]
    # source_num = A.shape[2]
    # At = xp.zeros((source_num, input_size, target_num))
    # for k in range(source_num):
    #     for j in range(target_num):
    #         At[k, :, j] = A[j, :, k]
    At = xp.swapaxes(A, 0, 2)
    return At


def ttrans(A):
    """
    Transposes second and third dim in matrix. Usually for WEIGHT TYPE where we change stream to source and v.v.
    A = [
         [[1, 2, 4], [2, 3, 4]],
         [[1, 2, 4], [2, 3, 4]],
         [[1, 2, 4], [2, 3, 4]]
        ]
    A.shape = (3, 2, 3)
    ttrans(A) = [
             [[1, 2], [2, 3], [4, 4]],
             [[1, 2], [2, 3], [4, 4]],
             [[1, 2], [2, 3], [4, 4]]
            ], shape = (3, 3, 2)
    :param A: shape = (input_size, #target, #source)
    :return: shape = (input_size, #source, #target)
    """
    # input_size = A.shape[0]
    # target_num = A.shape[1]
    # source_num = A.shape[2]
    # At = xp.zeros((input_size, source_num, target_num))
    # for k in range(source_num):
    #     for j in range(target_num):
    #         At[:, k, j] = A[:, j, k]
    At = xp.swapaxes(A, 1, 2)
    return At


def typeswap(A):
    """
    Transposes first and second dim in matrix. Changes the type of variable fro WEIGHT TYPE to SIGNAL TYPE and v.v.
    A = TODO
    A.shape = (1, 2, 3)
    typeswap(A) = TODO typeswap(A).shape = (2, 1, 3)
    :param A: shape = (first, second, #stream)
    :return: shape = (second, first, #stream)
    TODO unittest
    """
    At = xp.swapaxes(A, 0, 1)
    return At


def t_anti_reflexivity(A):
    """
    Diagonal between second and third dimension is set to zero.
    :param A: shape = (#streams, ?, #streams)
    :return: shape = (#streams, ?, #streams)
    """
    A_res = xp.zeros(A.shape) + A
    for i in range(A.shape[0]):
        A_res[i, :, i] = 0
    return A_res


def tvectorify(*args):
    return xp.asarray(list(args)).flatten().reshape((1, len(args), 1))


def streamify(*args):
    return xp.asarray(list(args)).flatten().reshape((1, 1, len(args)))


def gaussian_rbf(eps, broadcasted=True):
    if broadcasted:
        def dyn(centers, inputs):
            """
            Radial basis function with euclidean distance.

            :param centers: shape = (input_size, output_size, #streams)
            :param inputs: shape = (output_size, input_size, 1)
            :return: shape = (1, output_size, #streams)
            """
            dist = xp.abs(xp.sum(xp.square(badd(inputs, -centers)), axis=0, keepdims=True))
            return xp.exp(-eps * dist),

    else:
        def dyn(centers, inputs):
            """
            Radial basis function with euclidean distance.

            :param centers: shape = (input_size, output_size, #streams)
            :param inputs: shape = (1, input_size, #streams)
            :return: shape = (1, output_size, #streams)
            """
            dist = xp.abs(xp.sum(xp.square(tadd(inputs, -centers)), axis=0, keepdims=True))
            return xp.exp(-eps * dist),

    return dyn


def gaussian_double_rbf(eps_input, eps_pivot):

    def dyn(centers, pivots, inputs):
        """
        Radial basis function with euclidean distance of centers from the input and pivot.
        An input signal of dimension input_dimension generated by respective #stream is compared
        with center of respective #stream and output. Each #stream has its own pivot to which is
        the center of respective #stream compared.

        :param centers: WEIGHT TYPE, shape = (input_dimension, #sources, #streams)
        :param pivots: WEIGHT TYPE, shape = (input_dimension, #sources, 1)
        :param inputs: SIGNAL TYPE, shape = (#sources, input_dimension, 1)
        :return: shape = (1, #sources, #streams)
        """
        dist_input = xp.abs(xp.sum(xp.square(badd(inputs, -centers)), axis=0, keepdims=True))
        dist_pivot = xp.abs(xp.sum(xp.square(badd(typeswap(pivots), -centers)), axis=0, keepdims=True))
        return xp.exp(- eps_input * dist_input - eps_pivot * dist_pivot),

    return dyn