from dynsys_framework.functions.common import *


def weighted_matsu_neuron(beta, tau, gamma):
    def dyn(u, v, w, c_in, *args):
        if len(args) > 0:
            c_in = np.asarray([c_in] + list(args))
        c = w.dot(c_in)

        dv = rct(u) - v
        dv /= tau

        du = - u - v * beta + c
        du /= gamma

        return du, dv

    return dyn
