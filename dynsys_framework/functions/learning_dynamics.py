import numpy as np

def grossberg(learning_rate):
    """

    :param learning_rate:
    :return:
    """
    def dyn(a, w, o):
        """
        Learning should find such weights 'w' that is similar to (mean of) output 'o'.
        :param a: activation exciting the learning (usually activation of the post-synaptic neuron)
        :param o: output that is being imitated (usually output of pre-synaptic neuron)
        :param w: updated connection weight
        :return: weight gradient dw
        """
        dw = learning_rate * a * (o - w)
        return dw,
    return dyn


def oja(decay=0):
    """

    :param learning_rate:
    :return:
    """
    def dyn(y, w, x, learning_rate):
        """
        Oja's rule.
        :param x: pre-synaptic input
        :param y: activation (scalary)
        :param w: approaches first PCA component
        :return: weight gradient dw
        """
        dw = learning_rate * y * (x - w * y) - learning_rate * decay * w
        return dw,
    return dyn