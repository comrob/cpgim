import unittest
import numpy as np
from dynsys_framework.functions.tensor2.common import sdot, bdot, sadd, cadd, t_anti_reflexivity


class TestTensorFunctions(unittest.TestCase):

    def test_sdot(self):
        x = np.asarray([
            [[1, 3], [2, 4], [3, 5]]
        ])
        A = np.asarray([
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]]
        ])
        expected = np.asarray([[[6, 24], [12, 36]]])
        res = sdot(x, A)
        self.assertListEqual([1, 2, 2], list(res.shape))
        np.testing.assert_equal(expected, res)

    def test_sdot_random(self):
        def naive_sdot(stream, weight):
            """l, j: \Sum_{i, k} w^{k=l,l}_{i,j} x^{k=l}_i = y^l_j"""
            d = np.zeros((1, weight.shape[1], weight.shape[2]))
            for l in range(A.shape[2]):
                for j in range(A.shape[1]):
                    for i in range(A.shape[0]):
                        d[0, j, l] += stream[0, i, l] * weight[i, j, l]
            return d
        x = np.random.rand(4*20).reshape((1, 4, 20))
        A = np.random.rand(4*10*20).reshape((4, 10, 20))

        res = sdot(x, A)
        expected = naive_sdot(x, A)
        np.testing.assert_equal(expected, res)

    def test_bdot(self):
        x = np.asarray([
            [[1], [2], [3]]
        ])
        A = np.asarray([
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]]
        ])
        expected = np.asarray([[[6, 12], [12, 18]]])
        res = bdot(x, A)
        self.assertListEqual([1, 2, 2], list(res.shape))
        np.testing.assert_equal(expected, res)

    def test_bdot_random(self):

        def naive_bdot(stream, weight):
            """ l, j: \Sum_{i, k} w^{k=1,l}_{i,j} x^{k=1}_i = y^l_j  """
            d = np.zeros((1, weight.shape[1], weight.shape[2]))
            for l in range(weight.shape[2]):
                for j in range(weight.shape[1]):
                    for i in range(weight.shape[0]):
                        d[0, j, l] += stream[0, i, 0] * weight[i, j, l]
            return d
        x = np.random.rand(4*1).reshape((1, 4, 1))
        A = np.random.rand(4*10*20).reshape((4, 10, 20))

        res = bdot(x, A)
        expected = naive_bdot(x, A)
        np.testing.assert_equal(expected, res)

    def test_cadd_simple(self):
        x = np.asarray(
            [
            [[1, 3], [2, 4], [3, 5]]
            ]
        )

        A = np.asarray(
            [
                [[1, 2], [2, 3]],
                [[1, 2], [2, 3]],
                [[1, 2], [2, 3]]
            ]
        )

        expected = np.asarray(
            [
                [[2, 3], [5, 6]],
                [[3, 4], [6, 7]],
                [[4, 5], [7, 8]]
            ]
        )
        res = cadd(A, x)
        np.testing.assert_equal(expected, res)

    def test_cadd_random(self):
        def naive_cadd(weight, stream):
            """ i, j, l, k=j : w^{k=j, l}_{i, j} + x^{k=j}_i = m^{k=j, l}_{i, j}"""
            new_weight = np.zeros(weight.shape)
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    for l in range(weight.shape[2]):
                        new_weight[i, j, l] = weight[i, j, l] + stream[0, i, j]
            return new_weight
        x = np.random.rand(4*20).reshape((1, 4, 20))
        A = np.random.rand(4*20*30).reshape((4, 20, 30))

        res = cadd(A, x)
        expected = naive_cadd(A, x)
        np.testing.assert_equal(expected, res)

    def test_sadd_simple(self):
        x = np.asarray(
            [
            [[1, 3], [2, 4], [3, 5]]
            ]
        )

        A = np.asarray(
            [
                [[1, 2], [2, 3]],
                [[1, 2], [2, 3]],
                [[1, 2], [2, 3]]
            ]
        )

        expected = np.asarray(
            [
                [[2, 5], [3, 6]],
                [[3, 6], [4, 7]],
                [[4, 7], [5, 8]]
            ]
        )
        res = sadd(A, x)
        np.testing.assert_equal(expected, res)

    def test_sadd_random(self):
        def naive_sadd(weight, stream):
            """ i, j, l: w^{k=l,l}_{i,j} + b^{l}_i  = m^{k=l,l}_{i,j}"""
            new_weight = np.zeros(weight.shape)
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    for l in range(weight.shape[2]):
                        new_weight[i, j, l] = weight[i, j, l] + stream[0, i, l]
            return new_weight
        x = np.random.rand(4*20).reshape((1, 4, 20))
        A = np.random.rand(4*10*20).reshape((4, 10, 20))

        res = sadd(A, x)
        expected = naive_sadd(A, x)
        np.testing.assert_equal(expected, res)

    def test_t_anti_reflexivity(self):
        A = np.asarray([
            [[1, 2, 4], [2, 3, 4], [5, 6, 7]],
            [[1, 2, 4], [2, 3, 4], [5, 6, 7]],
            [[1, 2, 4], [2, 3, 4], [5, 6, 7]]
        ])
        expected = np.asarray([
            [[0, 2, 4], [0, 3, 4], [0, 6, 7]],
            [[1, 0, 4], [2, 0, 4], [5, 0, 7]],
            [[1, 2, 0], [2, 3, 0], [5, 6, 0]]
        ])
        res = t_anti_reflexivity(A)

        np.testing.assert_equal(expected, res)