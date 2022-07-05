import unittest
import numpy as np
from dynsys_framework.functions.tensor.common import badd, tadd, tdot, ttrans, bdot, t_anti_reflexivity, unstream,\
    gaussian_double_rbf


class TestTensorFunctions(unittest.TestCase):

    def test_tadd(self):
        A = np.asarray([1, 2, 2, 3, 1, 2, 2, 3, 1, 2, 2, 3]).reshape((3, 2, 2))
        x = np.asarray([1, 3, 2, 4, 3, 5]).reshape((1, 3, 2))
        expected = np.asarray([2, 5, 3, 6, 3, 6, 4, 7, 4, 7, 5, 8]).reshape((3, 2, 2))
        res = tadd(x, A)
        self.assertListEqual([3, 2, 2], list(res.shape))
        np.testing.assert_equal(expected, res)

    def test_badd(self):
        A = np.asarray([1, 2, 2, 3, 1, 2, 2, 3, 1, 2, 2, 3]).reshape((3, 2, 2))
        x = np.asarray([1, 2, 3]).reshape((1, 3))
        expected = np.asarray([[[2, 3], [3, 4]], [[3, 4], [4, 5]], [[4, 5], [5, 6]]])
        res = badd(x, A)
        self.assertListEqual([3, 2, 2], list(res.shape))
        np.testing.assert_equal(expected, res)

    def test_badd_multisource(self):
        A = np.asarray([
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]]
        ])
        x = np.asarray([
            [1, 2, 3],
            [4, 5, 6]
        ])
        expected = np.asarray([
            [[2, 3], [6, 7]],
            [[3, 4], [7, 8]],
            [[4, 5], [8, 9]]
        ])
        res = badd(x, A)
        self.assertListEqual([3, 2, 2], list(res.shape))
        np.testing.assert_equal(expected, res)

    def test_badd_deep_multisource(self):
        A = np.asarray([
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]]
        ])
        x = np.asarray([
            [[1], [2], [3]],
            [[4], [5], [6]]
        ])
        expected = np.asarray([
            [[2, 3], [6, 7]],
            [[3, 4], [7, 8]],
            [[4, 5], [8, 9]]
        ])
        res = badd(x, A)
        self.assertListEqual([3, 2, 2], list(res.shape))
        np.testing.assert_equal(expected, res)

    def test_ttrans(self):
        A = np.asarray([
            [[1, 2, 4], [2, 3, 4]],
            [[1, 2, 4], [2, 3, 4]],
            [[1, 2, 4], [2, 3, 4]]
        ])
        expected = np.asarray([
            [[1, 2], [2, 3], [4, 4]],
            [[1, 2], [2, 3], [4, 4]],
            [[1, 2], [2, 3], [4, 4]]
        ])
        res = ttrans(A)
        self.assertListEqual([3, 3, 2], list(res.shape))
        np.testing.assert_equal(expected, res)

    def test_tdot(self):
        x = np.asarray([
            [[1, 3], [2, 4], [3, 5]]
        ])
        A = np.asarray([
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]]
        ])
        expected = np.asarray([[[6, 24], [12, 36]]])
        res = tdot(x, A)
        self.assertListEqual([1, 2, 2], list(res.shape))
        np.testing.assert_equal(expected, res)

    def test_bdot(self):
        x = np.asarray([
            [1, 2, 3]
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

    def test_unstream(self):
        A = np.asarray([
            [[1, 2, 4], [2, 3, 4], [3, 4, 4], [4, 5, 4]],
        ])
        expected = np.asarray([
            [[1], [2], [3], [4]],
            [[2], [3], [4], [5]],
            [[4], [4], [4], [4]]
        ])
        res = unstream(A)

        np.testing.assert_equal(expected, res)

