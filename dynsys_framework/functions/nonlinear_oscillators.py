def van_der_pol(beta):
    def dyn(u, v):
        dv = u
        du = beta*(1 - v*v)*u - v

        return du, dv

    return dyn


def ijspeert(beta, E, tau):
    def dyn(u, v):
        dv = u
        du = -beta*(v*v + u*u - E)/E*u - v
        return du/tau, dv/tau
    return dyn
