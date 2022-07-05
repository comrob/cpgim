from dynsys_framework.execution_helpers.model_executor_parameters import ModelExecutorParameters
import numpy as np
xp = np
MOTOR_EXPECTED_REGULARIZATION = "u_exp_reg"

#u_exp, u_ctx, ctx_ctr_sel


def model(mep: ModelExecutorParameters, optimized_u_name, base_u_name, selection_vector_name,
          leg_squared_budged=0.3,
          leg_squared_weight=1.,
          variance_maximization_weight=0.,
          neighbourhood_activity_distance_minimization_weight=1.,
          zero_mean_weight=1.,
          anti_phase_symmetry=0.
          ):

    sq_leg_reg = motor_leg_squared_regularization(leg_squared_budged)
    var_max_reg = motor_variation_regularization()
    zero_mean_reg = motor_phase_zero_mean_regularization()
    phase_neigh_min_reg = motor_phase_neighbourhood_regularization()
    symmetry_reg = motor_contralateral_joint_symmetry_regularization()

    mep.add(MOTOR_EXPECTED_REGULARIZATION, (optimized_u_name, base_u_name, selection_vector_name),
            lambda _u_exp, _u_ctx, _ctr_sel:
            sq_leg_reg(_u_exp, _u_ctx, _ctr_sel) * leg_squared_weight
            - var_max_reg(_u_exp, _u_ctx, _ctr_sel) * variance_maximization_weight
            + phase_neigh_min_reg(_u_exp, _u_ctx, _ctr_sel) * neighbourhood_activity_distance_minimization_weight
            - zero_mean_reg(_u_exp, _u_ctx, _ctr_sel) * zero_mean_weight
            - symmetry_reg(_u_exp, _u_ctx, _ctr_sel) * anti_phase_symmetry)


def motor_squared_regularization(radius):
    def reg(u_exp, u_ctx, ctx_ctr_sel):
        ret = np.zeros(u_exp.shape)
        sel_id = np.argmax(ctx_ctr_sel)
        any_sel = np.sum(ctx_ctr_sel) > 0
        u_absolute = u_exp + u_ctx
        r_diff = radius - np.sum(np.square(u_absolute), axis=(1, 3))  # sum in phase direction
        rgl = -4 * r_diff[:,None,:,None,:] * u_absolute
        if any_sel:
            ret[:,:,:,:,sel_id] = rgl[:,:,:,:,sel_id]
        return ret
    return reg


def motor_leg_squared_regularization(radius):
    A = xp.zeros((6, 12))
    A[0, [0, 2]] = 1.   # fl
    A[1, [1, 3]] = 1.   # fr
    A[2, [4, 6]] = 1.   # hl
    A[3, [5, 7]] = 1.   # hr
    A[4, [8, 10]] = 1.  # ml
    A[5, [9, 11]] = 1.  # mr

    def reg(u_exp, u_ctx, ctx_ctr_sel):
        ret = np.zeros(u_exp.shape)
        sel_id = np.argmax(ctx_ctr_sel)
        any_sel = np.sum(ctx_ctr_sel) > 0
        u_absolute = u_exp + u_ctx
        u_abs_ampl = np.sum(np.square(u_absolute), axis=3) # sum in phase direction
        leg_ampl = xp.einsum("injx,ln->lx", u_abs_ampl, A)
        r_diff = radius - leg_ampl
        rgl = -4 * xp.einsum("lx,ln,injcx->injcx", r_diff, A, u_absolute)
        if any_sel:
            ret[:, :, :, :, sel_id] = rgl[:, :, :, :, sel_id]
        return ret
    return reg


def motor_variation_regularization():
    def reg(u_exp, u_ctx, ctx_ctr_sel):
        ret = xp.zeros(u_exp.shape)
        sel_id = xp.argmax(ctx_ctr_sel)
        any_sel = xp.sum(ctx_ctr_sel) > 0
        u_absolute = u_exp + u_ctx
        u_abs_muph = xp.mean(u_absolute, axis=3)  # mean over phase
        u_dif = u_abs_muph[:, :, :, None, :] - u_absolute
        rgl = -1 / u_exp.shape[3] * u_dif
        if any_sel:
            ret[:, :, :, :, sel_id] = rgl[:, :, :, :, sel_id]
        return ret
    return reg


def motor_phase_neighbourhood_regularization():
    def reg(u_exp, u_ctx, ctx_ctr_sel):
        ret = xp.zeros(u_exp.shape)
        sel_id = xp.argmax(ctx_ctr_sel)
        any_sel = xp.sum(ctx_ctr_sel) > 0
        u_absolute = u_exp + u_ctx
        _u_abs = u_absolute[:, :, :, :, sel_id]
        ret[:, :, :, :, sel_id] = 2 * _u_abs
        ret[:, :, :, 1:-1, sel_id] -= _u_abs[:, :, :, 0:-2] + _u_abs[:, :, :, 2:]  # middle
        ret[:, :, :, 0, sel_id] -= _u_abs[:, :, :, 1] + _u_abs[:, :, :, -1]  # first
        ret[:, :, :, -1, sel_id] -= _u_abs[:, :, :, -2] + _u_abs[:, :, :, 0]  # last
        if not any_sel:
            return xp.zeros(u_exp.shape)
        return ret
    return reg


def motor_phase_zero_mean_regularization():
    def reg(u_exp, u_ctx, ctx_ctr_sel):
        ret = xp.zeros(u_exp.shape)
        sel_id = xp.argmax(ctx_ctr_sel)
        any_sel = xp.sum(ctx_ctr_sel) > 0

        u_absolute = u_exp + u_ctx
        _u_abs = u_absolute[:, :, :, :, sel_id]
        u_abs_muph = np.mean(_u_abs, axis=3)
        ret[:, :, :, :, sel_id] -= u_abs_muph[:, :, :, None]
        if not any_sel:
            return xp.zeros(u_exp.shape)
        return ret
    return reg


def motor_contralateral_joint_symmetry_regularization():
    left_ids = [0, 2, 4, 6, 8, 10]
    right_ids = [1, 3, 5, 7, 9, 11]

    def reg(u_exp, u_ctx, ctx_ctr_sel):
        ret = xp.zeros(u_exp.shape)
        _C = u_exp.shape[3]
        sel_id = xp.argmax(ctx_ctr_sel)
        any_sel = xp.sum(ctx_ctr_sel) > 0
        u_absolute = u_exp + u_ctx
        _u_abs = u_absolute[:, :, :, :, sel_id]
        reg = -_u_abs
        reg[:, left_ids, :, :_C//2] += _u_abs[:, right_ids, :, _C//2:]
        reg[:, left_ids, :, _C//2:] += _u_abs[:, right_ids, :, :_C//2]
        reg[:, right_ids, :, :_C//2] += _u_abs[:, left_ids, :, _C//2:]
        reg[:, right_ids, :, _C//2:] += _u_abs[:, left_ids, :, :_C//2]
        if any_sel:
            ret[:, :, :, :, sel_id] = 2 * reg
        return ret
    return reg