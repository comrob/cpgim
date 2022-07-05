import numpy as np
# import cupy as cp
xp = np

def rct(x):
    return xp.maximum(x, 0)


def matsuoka_oscillator(alpha, beta, tau, gamma):
    def dyn(uv, c_e, c_f):
        u_e = uv[:, 0:1, :]
        u_f = uv[:, 1:2, :]
        rct_u_e = rct(u_e)
        rct_u_f = rct(u_f)

        v_e = uv[:, 2:3, :]
        v_f = uv[:, 3:4, :]

        duv = xp.zeros((1, 4, uv.shape[2]))

        duv[:, 2:3, :] = (rct_u_e - v_e) / tau

        duv[:, 3:4, :] = (rct_u_f - v_f) / tau

        duv[:, 0:1, :] = (- u_e - rct_u_f * alpha - v_e * beta + c_e) / gamma

        duv[:, 1:2, :] = (- u_f - rct_u_e * alpha - v_f * beta + c_f) / gamma

        return duv,

    return dyn


def matsuoka_neuron(beta, tau, gamma):
    def dyn(uv, c):
        u = uv[:, 0:1, :]
        v = uv[:, 1:2, :]

        duv = xp.zeros((1, 2, uv.shape[2]))

        duv[:, 1:2, :] = (rct(u) - v) / tau
        duv[:, 0:1, :] = (- u - v * beta + c) / gamma

        return duv,

    return dyn

def matsuoka_firing_neuron(beta, tau, gamma, fire_u_thr, fire_c_thr, fire_ampl):
    def dyn(uv, c):
        u = uv[:, 0:1, :]
        v = uv[:, 1:2, :]

        duv = xp.zeros((1, 2, uv.shape[2]))
        exc = np.bitwise_and(u > fire_u_thr, c > fire_c_thr)
        _c = ~exc*c + exc*fire_ampl
        duv[:, 1:2, :] = (rct(u) - v) / tau
        duv[:, 0:1, :] = (- u - v * beta + _c) / gamma

        return duv,

    return dyn
