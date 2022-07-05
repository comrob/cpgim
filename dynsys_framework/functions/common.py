import numpy as np


def rct(x):
    return np.maximum(x, 0)


def wsum(weights, outputs, b=0):
    return weights.dot(outputs) + b,


def gaussian_rbf(eps):
    def dyn(centers, inputs):
        dist = np.sqrt(np.sum(np.square(centers - inputs)))  # TODO will work on just vectors
        return np.exp(-np.square(eps * dist)),

    return dyn


def concat(*args):
    return np.asarray(list(args)).flatten(),


def constant(constant_value):
    def dyn():
        return constant_value,

    return dyn
