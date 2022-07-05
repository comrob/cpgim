from dynsys_framework.functions.tensor.common import *


def grossberg(learning_rate, broad_casted=False):
    """

    :param learning_rate:
    :return:
    """
    if not broad_casted:
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
            dW = llearning_rate * y * tadd(x, -W)
            return dW,
    else:
        def dyn(y, W, x, llearning_rate=learning_rate):
            """
            Learning should find such weights 'w' that is similar to (mean of) input 'x'.

            In this case we have only one presynaptic vector of size #inputs which is broadcasted into each group
            (#streams) of neurons (#neurons). Groups are independent on each other.

            :param y: shape = (1, 1, #neurons),
            activation exciting the learning (usually activation of the post-synaptic neuron)
            :param x: shape = (1, #inputs),
            input that is being imitated (usually output of pre-synaptic neuron)
            :param W: shape = (#inputs, #neurons, #streams), updated connection weight
            :return: shape as W, weight gradient dw
            """
            dW = llearning_rate * y * badd(x, -W)
            return dW,
    return dyn


def oja(llearning_rate=0.1, broadcasted=False):
    """

    :param learning_rate:
    :return:
    """
    if broadcasted:
        def dyn(y, W, x, learning_rate=llearning_rate):
            """
            Oja's rule.

            In this case we have multiple (#streams) presynaptic vectors of size input_size which are being combined
            in groups of neurons, each of size #neurons.

            :param x: shape = (1, input_size, #streams), pre-synaptic input
            :param y: shape = (1, 1, #streams), activation
            :param W: shape = (input_size, #neurons, #streams), approaches first PCA component
            :return: weight gradient dw
            """
            dw = learning_rate * y * badd(x, - W * y)
            return dw,
    else:
        def dyn(y, W, x, learning_rate=llearning_rate):
            """
            Oja's rule.

            In this case we have multiple (#streams) presynaptic vectors of size input_size which are being combined
            in groups of neurons, each of size #neurons.

            :param x: shape = (1, input_size, #streams), pre-synaptic input
            :param y: shape = (1, 1, #streams), activation
            :param W: shape = (input_size, #neurons, #streams), approaches first PCA component
            :return: weight gradient dw
            """
            dw = learning_rate * y * tadd(x, - W * y)
            return dw,

    return dyn


def anti_oja(llearning_rate=0.1):
    """

    :param learning_rate:
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