from dynsys_framework.functions.tensor2.common import *


def grossberg(learning_rate, stream_wise=False):
    """

    :param learning_rate:
    :return:
    """
    _add = cadd
    if stream_wise:
        _add = sadd

    def dyn(y, W, x, llearning_rate=learning_rate):
        """
        Learning should find such weights 'w' that is similar to (mean of) input 'x'.

        In this case we have multiple sources (#streams) of presynaptic signal of size (#inputs)
        which are learned and processed in #neurons independently.

        :param y: shape = (1, #neurons, 1),
        activation exciting the learning (usually activation of the post-synaptic neuron)
        Also shape (1, #neurons, #streams) works (depends on the * operator)
        See:
        y = np.asarray([1,2,3,4,5,6]).reshape((1,2,3))
        A = np.ones((4,2,3))
        X = y * A
        X[:,:,0] has shape (4,2) has only 1s and 4s

        :param x: shape = (1, #inputs, #streams),
        input that is being imitated (usually output of pre-synaptic neuron)
        :param W: shape = (#inputs, #neurons, #streams), updated connection weight
        :return: shape as W, weight gradient dw
        """
        dW = llearning_rate * y * _add(-W, x)
        return dW,

    return dyn


def oja(learning_rate=0.1, stream_wise=False):
    """

    :param learning_rate:
    :return:
    """
    _add = cadd
    if stream_wise:
        _add = sadd

    def dyn(y, W, x, llearning_rate=learning_rate):
        """
        Oja's rule.

        In this case we have multiple (#streams) presynaptic vectors of size input_size which are being combined
        in groups of neurons, each of size #neurons.

        :param x: shape = (1, input_size, #streams), pre-synaptic input
        :param y: shape = (1, 1, #streams), activation
        :param W: shape = (input_size, #neurons, #streams), approaches first PCA component
        :return: weight gradient dw
        """
        dw = llearning_rate * y * _add(- W * y, x)
        return dw,
    return dyn


def anti_oja(llearning_rate=0.1):
    """

    :param llearning_rate:
    :return:
    """

    def dyn(t, W, x, learning_rate=llearning_rate):
        """
        Anti-Oja's rule.

        Rule to learn such weights W that will amplify x so the W*x will approach t.

        :param x: shape = (1, input_size, #streams), pre-synaptic input
        :param t: shape = (1, input_size, #streams), target presynaptic net input
        :param W: shape = (input_size, #neurons, #streams), net operator
        :return: weight gradient dw
        """
        xt = np.swapaxes(x, 0, 1)
        tt = np.swapaxes(t, 0, 1)
        dW = learning_rate * xt * (tt - W * xt)
        return dW,

    return dyn


def adaptive_firing_threshold(alpha, beta, epsilon, theta_min, learning_rate):
    """

    :param alpha: rate of ascension
    :param beta: rate of descend
    :param epsilon: margin of activation
    :param theta_min: minimum threshold
    :param learning_rate:
    :return:
    """
    def dyn(theta, activation, llearning_rate=learning_rate):
        """
        Adaptation of firing threshold so the neuron will not fire (that much) and yet the threshold is kept low.
        Inspired by https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full

        :param theta: (1, input_size, #streams), neuron firing threshold
        :param activation: (1, input_size, #streams), neuron activation
        :param llearning_rate: learning rate
        :return:
        """
        return (alpha * rct(activation + epsilon - theta) + beta * (theta_min - theta)) * llearning_rate

    return dyn
