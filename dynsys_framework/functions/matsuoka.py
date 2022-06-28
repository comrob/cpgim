from dynsys_framework.functions.common import rct


def matsuoka_oscillator(alpha, beta, tau, gamma):
    def dyn(u_e, u_f, v_e, v_f, c_e, c_f):
        dv_e = rct(u_e) - v_e
        dv_e /= tau

        dv_f = rct(u_f) - v_f
        dv_f /= tau

        du_e = - u_e - rct(u_f) * alpha - v_e * beta + c_e
        du_e /= gamma

        du_f = - u_f - rct(u_e) * alpha - v_f * beta + c_f
        du_f /= gamma

        return du_e, du_f, dv_e, dv_f

    return dyn


def matsuoka_neuron(beta, tau, gamma):
    def dyn(u, v, c):
        dv = rct(u) - v
        dv /= tau

        du = - u - v * beta + c
        du /= gamma

        return du, dv
    return dyn