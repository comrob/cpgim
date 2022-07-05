from dynsys_framework.functions.tensor2.common import *
from dynsys_framework.functions.tensor2.dynamics.learning_dynamics import *
from dynsys_framework.functions.tensor2.dynamics.matsuoka import *
from dynsys_framework.functions.tensor2.specialized_units import *
from dynsys_framework.functions.tensor2.expressions.activation_functions import *

"""
Tensor computation:
Use for speeding up the linear algebra calculations.

When considering the architecture there are two base variable types:
1) STREAM and 2) WEIGHT

Stream:
Each stream of vectors is produced by a UNIT (e.g. Matsuoka CPG produces a 4-dim vector stream).
The units are organized into unit LAYERS, and thus the layer outputs a multi-source stream which is encoded
as a tensor of shape (1, vector_dimensionality, stream_source_size);
e.g. stream from three Matsuoka CPGs has shape (1, 4, 3).

Weight:
The streams are linearly combined, where the parameter of linear combination is called WEIGHT.
The stream linear combination recombines the input_vector_dimension and stream_source_size into a new
stream with output_vector_dimension and stream_sink_size.
Therefore, the full WEIGHT is a 4-dimensional tensor of shape:
(input_vector_dimension, output_vector_dimension, stream_sink_size, stream_source_size)

Notation norm:
Follow this notation when reading/creating the documentation

i, input_vector_dim - index of (source-)stream vector element
j, output_vector_dim - index of sink-stream vector element
k, source_stream_size - index of the source-stream
l, sink_stream_size - index of the sink-stream

I, J, K, L are sizes of the respective indexes.

STREAM SHAPE is (1, I, K)
WEIGHT SHAPE is (I, J, L, K) for the FULL WEIGHT, 
             and (I, J, L) for the SHORTENED WEIGHT (used for stream-wise operations or source broadcasting)

The full linear combination (fdot):
Takes stream x of shape (1, I, K) and outputs stream y of shape (1, J, L) where the elements are
l, j: \Sum_{i, k} w^{k,l}_{i,j} x^k_i = y^l_j ,
the w^{k,l}_{i,j} are elements of weight parameter W of shape (I, J, L, K).
"""

if __name__ == '__main__':
    import time
    """
    Calc time experiments
    input/output vector dimensionalities (I, J) are expected to be low and constant
    while the source/stream dimensions (K, L) should be easily scalable (as number of units will grow).
    """
    _im = 4
    _jm = 2
    _km = 100000
    _lm = 10000

    # original indexing
    x = np.random.rand(_im * _km).reshape((1, _im, _km))
    A = np.random.rand(_im*_jm*_km*_lm).reshape((_im, _jm, _lm, _km))
    y = np.zeros((1, _jm, _lm))
    start = time.time()
    for l in range(_lm):
        for k in range(_km):
            y[0, :, l] += x[0, :, k].dot(A[:,:, l, k])
    end = time.time()
    print("time 1: {}".format(end - start))

    # reverse indexing
    x = np.random.rand(_im * _km).reshape((1, _km, _im))
    A = np.random.rand(_im*_jm*_km*_lm).reshape((_lm, _km, _im, _jm))
    y = np.zeros((1, _lm, _jm))
    start = time.time()
    for l in range(_lm):
        for k in range(_km):
            y[0, l, :] += x[0, k, :].dot(A[l, k, :, :])
    end = time.time()
    print("time 2: {}".format(end - start))


    # original indexing - different calculation
    x = np.random.rand(_im * _km).reshape((1, _im, _km))
    A = np.random.rand(_im*_jm*_km*_lm).reshape((_im, _jm, _km,  _lm))
    y = np.zeros((1, _jm, _lm))
    start = time.time()
    for i in range(_im):
        for j in range(_jm):
            y[0, j, :] += x[0, i, :].dot(A[i, j, :, :])
    end = time.time()
    print("time 3: {}".format(end - start))
