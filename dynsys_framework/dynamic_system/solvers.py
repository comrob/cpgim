
def euler_solver(step):
    def solver(model, state, d_to_int):
        """
        Euler numerical solver for differential equations.
        :param model: callable that takes state and returns new state containing derivative variables
        :param state: dictionary variable_name -> value, value must implement scaling and addition
        :param d_to_int: dictionary derivative_variable_name -> integrated_variable_name,
        serves also as a list of derivative variables
        :return: new state with approximated integrations of variables in d_to_int
        """
        xy = state
        ydx = model(xy)
        for dv in d_to_int:
            ydx[d_to_int[dv]] = xy[d_to_int[dv]] + step * ydx[dv]
        return ydx

    return solver


def runge_kutta4_solver(step):
    # @njit
    def solver(model, state, d_to_int):
        """
        Runge-Kutta numerical solver for differential equations.
        :param model: callable that takes state and returns new state containing derivative variables
        :param state: dictionary variable_name -> value, value must implement scaling and addition
        :param d_to_int: dictionary derivative_variable_name -> integrated_variable_name,
        serves also as a list of derivative variables
        :return: new state with approximated integrations of variables in d_to_int
        """
        xy = state
        dtx1 = model(state)
        x1 = dict((d_to_int[k], xy[d_to_int[k]] + 0.5 * step * dtx1[k]) for k in d_to_int)
        dtx2 = model({**xy, **x1})
        x2 = dict((d_to_int[k], xy[d_to_int[k]] + 0.5 * step * dtx2[k]) for k in d_to_int)
        dtx3 = model({**xy, **x2})
        x3 = dict((d_to_int[k], xy[d_to_int[k]] + step * dtx2[k]) for k in d_to_int)
        dtx4 = model({**xy, **x3})
        for dv in d_to_int:
            dtx1[d_to_int[dv]] = xy[d_to_int[dv]] + (step / 6) * (dtx1[dv] + 2 * dtx2[dv] + 2 * dtx3[dv] + dtx4[dv])
        return dtx1

    return solver
