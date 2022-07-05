from dynsys_framework.execution_helpers.model_executor_parameters import ModelExecutorParameters
import numpy as np
import models.temporary_constants as TS
xp = np
import time

#RMS PROP
RMS_RO = 0.9
RMS_EPS = 0.001
RMS_LR = 10

_D = "d_"
STATE_PHASE_ESTIMATION = "ph_s"
MOTOR_PHASE_CONTROLLER = "ph_m"
MOTOR_PERTURBATION = "p_m"
STATE_PERTURBATION = "p_s"
MOTOR_PHASE_ACTIVATOR = "a_m"
STATE_PHASE_ACTIVATOR = "a_s"
STATE_CENTERS = "m_s"
MOTOR_CENTERS = "m_m"
####
SENSORY_INPUT = "y_inp"
SENSORY_REFERENCE = "y_ref"
SENSORY_REFERENCE_MASK = "y_ref_msk"
SENSORY_REFERENCE_MINMAX_MASK = "y_ref_msk_minmax"

SENSORY_ESTIMATION = "y_est"
SENSORY_EFFERENT_ESTIMATION = "y_eff"
SENSORY_MEMORY = "y_mem"
MODEL_ERROR = "e_mod"
MODEL_WEIGHTS = "W_mod"
MODEL_BIAS = "b_mod"
###
MOTOR_COMMAND_PERTURBATION = "u_cmd_p"
MOTOR_COMMAND = "u_cmd"
MOTOR_MEMORY = "u_mem"
MOTOR_EXPECTED = "u_exp"
MOTOR_DIFF = "u_dif"
MOTOR_CONTEXT = "u_ctx"
MOTOR_OUTPUT = "u_out"
MOTOR_ERROR = "e_mot"
MOTOR_ERROR_FREE_ENERGY = "e_mot_fe"
MOTOR_ERROR_FILTERED = "e_mot_flt"
MOTOR_PRIOR_ERROR_FREE_ENERGY = "e_mot_pr_fe"
###
MODEL_LEARNING_RATE = "lr_mod"
MOTOR_LEARNING_RATE = "lr_mot"
MEMORY_LEARNING_RATE = "lr_mem"
CONTEXT_LEARNING_RATE = "lr_ctx"
###
EPICYCLE_PHASE = "ph_ep"
EPICYCLE_PHASE_ACTIVATOR = "a_ep"
EPICYCLE_CENTERS = "m_ep"
EPICYCLE_PERTURBATION = "p_ep"
###
CONTEXT_MODEL_ERROR = "ctx_e_mod"

CONTEXT_SENSORY_ESTIMATION = "ctx_y_est"
CONTEXT_SENSORY_PREV_OBSERVAITON = "ctx_y_prev"

CONTEXT_MODEL_QUALITY = "ctx_mod_q"
CONTEXT_MODEL_QUALITY_MEAN = "ctx_mod_q_mu"
CONTEXT_MODEL_QUALITY_VARIANCE = "ctx_mod_q_var"
CONTEXT_ACTIVABLE = "ctx_act"

CONTEXT_IS_BEST_FREEZED = "ctx_opt_freeze"
CONTEXT_IS_BEST_HOT = "ctx_opt_hot"

CONTEXT_LEARNING_STATE = "ctx_state_lrn"
CONTEXT_LEARNING_SELECTION = "ctx_sel_lrn"
CONTEXT_CONTROL_SELECTION = "ctx_sel_ctr"
CONTEXT_COMMANDING_SELECTION = "ctx_sel_cmd"

CONTEXT_QUALITY_INNER_THRESHOLD_DYN = "ctx_q_th_in"
CONTEXT_QUALITY_OUTER_THRESHOLD_DYN = "ctx_q_th_out"
REAL_PERFORMANCE_QUALITY = "q_real"
###
PROB_VAR_SENSORY_MOTOR = "sigm_yu"
PROB_JOINT_SENSORYMOTOR = "pdf_yu"

###
RMS_MODEL_WEIGHTS = "W_mod_rms"
RMS_MODEL_BIAS = "b_mod_rms"
RMS_MOTOR_EXPECTED = "u_exp_rms"
RMS_D = "dd_"

def model(mep: ModelExecutorParameters,
          epicycle_size, phase_velocity,
          motor_dim, motor_segments_num, sensory_dim, sensory_segments_num,
          context_num,
          model_quality_lower_bound, model_quality_upper_bound, performance_quality_upper_bound,
          sensory_input_name=SENSORY_INPUT,
                                     sensory_reference_name=SENSORY_REFERENCE, use_motor_perturbation=True,
                                     use_state_perturbation=True, is_control_learned=False, is_model_learned=True,
                                     perturbation_probability=0.02, motor_command_override=False,
                                     is_model_lr_external=False, is_control_lr_external=False,
                                     dynamic_outer_threshold=False, regularization_name=None,
                                     epicycle_perturbation_eps=1., no_external_perturbation=True,
                                     q_mean_up_lr=0.0005, q_mean_down_lr=0.02, u_exp_act_lr=1e-6
          ):
    if is_model_learned:
        model_learning_rate = RMS_LR
    else:
        model_learning_rate = 0

    if is_control_learned:
        control_learning_rate = RMS_LR
    else:
        control_learning_rate = 0

    motor_error_input = MOTOR_ERROR_FREE_ENERGY



    ## PHASE CONTROL-ESTIMATION ---------------------------------------------------------------
    mep.add(STATE_PHASE_ESTIMATION, EPICYCLE_PHASE, lambda ep_ph: ep_ph * epicycle_size - xp.pi - 8 * (xp.pi/epicycle_size))
    mep.add(STATE_PHASE_ACTIVATOR, (STATE_PHASE_ESTIMATION, STATE_CENTERS), parameter_rbf(2, max_norm=True))

    mep.add(MOTOR_PHASE_CONTROLLER, EPICYCLE_PHASE, lambda ep_ph: ep_ph * epicycle_size - xp.pi - 8 * (xp.pi/epicycle_size))
    mep.add(MOTOR_PHASE_ACTIVATOR, (MOTOR_PHASE_CONTROLLER, MOTOR_CENTERS), parameter_rbf(2, max_norm=True))

    # Epicycle
    mep.add(_D + EPICYCLE_PHASE, (EPICYCLE_PHASE, EPICYCLE_PERTURBATION),
            parameter_phase_syncing_estimator(phase_velocity/epicycle_size, pert_eps=epicycle_perturbation_eps,
                                              sync_speed=epicycle_size),
            default_initial_values={EPICYCLE_PHASE: xp.zeros((1, 1))})
    mep.add(EPICYCLE_PHASE_ACTIVATOR, (EPICYCLE_PHASE, EPICYCLE_CENTERS), parameter_rbf(2, max_norm=True))

    # MEMORY -----------------------------------------------------------------------------------
    mep.add(_D + MOTOR_MEMORY,
            (MOTOR_MEMORY, MOTOR_COMMAND, MOTOR_PHASE_ACTIVATOR, EPICYCLE_PHASE_ACTIVATOR, MEMORY_LEARNING_RATE),
            d_motor_memory,
            default_initial_values={MOTOR_MEMORY: xp.zeros((1, motor_dim, 1, motor_segments_num, epicycle_size))})

    mep.add(MOTOR_DIFF, (MOTOR_MEMORY, MOTOR_CONTEXT), motor_diff)

    mep.add(_D + SENSORY_MEMORY,
            (SENSORY_MEMORY, sensory_input_name, STATE_PHASE_ACTIVATOR, EPICYCLE_PHASE_ACTIVATOR, MEMORY_LEARNING_RATE),
            d_sensory_memory,
            default_initial_values={SENSORY_MEMORY: xp.zeros((1, sensory_dim, 1, sensory_segments_num, epicycle_size))})

    # SENSORY ESTIMATION ----------------------------------------------------------------------------
    # Estimation of entire epicycle for model learning
    mep.add(SENSORY_ESTIMATION, (MODEL_BIAS, MODEL_WEIGHTS, MOTOR_DIFF, CONTEXT_COMMANDING_SELECTION), get_epicycle_estimator(sensory_segments_num))

    # ERROR SIGNALS ----------------------------------------------------------------------------
    # Error signal of epicycle for model learning
    mep.add(MODEL_ERROR, (SENSORY_MEMORY, SENSORY_ESTIMATION, EPICYCLE_PHASE_ACTIVATOR), model_error)

    # Error signal of previous gait for motor learning
    mep.add(MOTOR_ERROR, (sensory_reference_name, SENSORY_MEMORY, EPICYCLE_PHASE_ACTIVATOR), true_motor_error)

    mep.add(_D + MOTOR_ERROR_FREE_ENERGY,
            (MOTOR_ERROR_FREE_ENERGY, MOTOR_ERROR, PROB_VAR_SENSORY_MOTOR, CONTEXT_COMMANDING_SELECTION),
            d_fe_motor_error,
            default_initial_values={MOTOR_ERROR_FREE_ENERGY: xp.random.randn(1, sensory_dim, 1, sensory_segments_num)})

    mep.add(MOTOR_ERROR_FILTERED, (motor_error_input, SENSORY_REFERENCE_MASK, SENSORY_REFERENCE_MINMAX_MASK), error_masking_clipping)

    # MODEL PARAMETERS ----------------------------------------------------------------------------
    # Bias
    mep.add(RMS_D + MODEL_BIAS, (MODEL_ERROR, STATE_PHASE_ACTIVATOR, MODEL_LEARNING_RATE, CONTEXT_LEARNING_SELECTION), d_bias)

    mep.add(_D + RMS_MODEL_BIAS, (RMS_MODEL_BIAS, RMS_D + MODEL_BIAS), rms_prop_learning_rate(RMS_RO),
            default_initial_values={RMS_MODEL_BIAS: xp.zeros((1, sensory_dim, 1, sensory_segments_num, context_num))})
    mep.add(_D + MODEL_BIAS, (RMS_MODEL_BIAS, RMS_D + MODEL_BIAS), rms_prop_grad(model_learning_rate, RMS_EPS),
            default_initial_values={MODEL_BIAS: xp.random.randn(1, sensory_dim, 1, sensory_segments_num, context_num)/5})
    # Weights
    mep.add(RMS_D + MODEL_WEIGHTS,
            (MODEL_WEIGHTS, MODEL_ERROR, MOTOR_DIFF, STATE_PHASE_ACTIVATOR, MOTOR_PHASE_ACTIVATOR, MODEL_LEARNING_RATE, CONTEXT_LEARNING_SELECTION),
            get_d_weights(sensory_segments_num))
    mep.add(_D + RMS_MODEL_WEIGHTS, (RMS_MODEL_WEIGHTS, RMS_D + MODEL_WEIGHTS), rms_prop_learning_rate(RMS_RO),
            default_initial_values={
                RMS_MODEL_WEIGHTS: xp.zeros((motor_dim, sensory_dim, motor_segments_num, sensory_segments_num, context_num))})
    mep.add(_D + MODEL_WEIGHTS, (RMS_MODEL_WEIGHTS, RMS_D + MODEL_WEIGHTS), rms_prop_grad(model_learning_rate, RMS_EPS),
            default_initial_values={
                MODEL_WEIGHTS: xp.zeros((motor_dim, sensory_dim, motor_segments_num, sensory_segments_num, context_num))}
            )
    # Expected Motor Control
    mep.add(RMS_D + MOTOR_EXPECTED,
            (MOTOR_ERROR_FILTERED, MODEL_WEIGHTS, MOTOR_LEARNING_RATE,
             CONTEXT_CONTROL_SELECTION, regularization_name),
            get_d_motor_expected(u_exp_act_lr=u_exp_act_lr))

    mep.add(_D + RMS_MOTOR_EXPECTED, (RMS_MOTOR_EXPECTED, RMS_D + MOTOR_EXPECTED), rms_prop_learning_rate(RMS_RO),
            default_initial_values={RMS_MOTOR_EXPECTED: xp.zeros((1, motor_dim, 1, motor_segments_num, context_num))})

    mep.add(_D + MOTOR_EXPECTED, (RMS_MOTOR_EXPECTED, RMS_D + MOTOR_EXPECTED),
            rms_prop_grad(control_learning_rate, RMS_EPS),
            default_initial_values={MOTOR_EXPECTED: xp.zeros((1, motor_dim, 1, motor_segments_num, context_num))})
    # Base motor control of current context
    mep.add(_D + MOTOR_CONTEXT,
            (MOTOR_CONTEXT, MOTOR_OUTPUT, CONTEXT_ACTIVABLE, CONTEXT_LEARNING_SELECTION, CONTEXT_COMMANDING_SELECTION),
            model_motor_context_follow_u_out,
            default_initial_values={MOTOR_CONTEXT: xp.zeros((1, motor_dim, 1, motor_segments_num, context_num))})
    # Measured sensory variance
    mep.add(_D + PROB_VAR_SENSORY_MOTOR,
            (PROB_VAR_SENSORY_MOTOR, MODEL_ERROR, CONTEXT_LEARNING_SELECTION, MODEL_LEARNING_RATE),
            get_d_sensory_posterior_variance(),
            default_initial_values={PROB_VAR_SENSORY_MOTOR: np.zeros((1, sensory_dim, 1, motor_segments_num, context_num)) + 0.0001})

    # MODEL PREDICTION QUALITY ----------------------------------------------------------------------------
    # Model-wise previous gait sensory estimation
    mep.add(CONTEXT_SENSORY_ESTIMATION,
            (MOTOR_MEMORY, MOTOR_CONTEXT, MODEL_WEIGHTS, MODEL_BIAS, STATE_PHASE_ACTIVATOR, EPICYCLE_PHASE_ACTIVATOR),
            prev_y_est)

    mep.add(CONTEXT_SENSORY_PREV_OBSERVAITON, (SENSORY_MEMORY, STATE_PHASE_ACTIVATOR, EPICYCLE_PHASE_ACTIVATOR),
            prev_y_obs)

    mep.add(CONTEXT_MODEL_ERROR, (CONTEXT_SENSORY_PREV_OBSERVAITON, CONTEXT_SENSORY_ESTIMATION), context_motor_error)
    # Model Quality
    mep.add(CONTEXT_MODEL_QUALITY,
            (CONTEXT_MODEL_ERROR, MODEL_ERROR, PROB_VAR_SENSORY_MOTOR, CONTEXT_COMMANDING_SELECTION, STATE_PHASE_ACTIVATOR),
            model_quality_current_normed)
    # Model Quality statistics
    mep.add(_D + CONTEXT_MODEL_QUALITY_MEAN, (CONTEXT_MODEL_QUALITY_MEAN, CONTEXT_MODEL_QUALITY),
            get_model_quality_mean(q_mean_down_lr=q_mean_down_lr, q_mean_up_lr=q_mean_up_lr),
            default_initial_values={CONTEXT_MODEL_QUALITY_MEAN: xp.zeros((1, context_num)) + model_quality_upper_bound})
    mep.add(_D + CONTEXT_MODEL_QUALITY_VARIANCE, (CONTEXT_MODEL_QUALITY_VARIANCE, CONTEXT_MODEL_QUALITY_MEAN, CONTEXT_MODEL_QUALITY),
            model_qiality_variance,
            default_initial_values={CONTEXT_MODEL_QUALITY_VARIANCE: xp.zeros((1, context_num))})

    # MODEL PERFORMANCE QUALITY ----------------------------------------------------------------------------
    mep.add(REAL_PERFORMANCE_QUALITY, MOTOR_ERROR_FILTERED, performance_quality)

    # CONTEXT MANAGEMENT SIGNALS ----------------------------------------------------------------------------
    mep.add(_D + CONTEXT_ACTIVABLE, (CONTEXT_ACTIVABLE, CONTEXT_MODEL_QUALITY_MEAN, CONTEXT_QUALITY_INNER_THRESHOLD_DYN, MODEL_LEARNING_RATE),
            context_is_freezed,
            default_initial_values={CONTEXT_ACTIVABLE: xp.zeros((1, context_num))})

    mep.add(CONTEXT_IS_BEST_FREEZED, (CONTEXT_ACTIVABLE, CONTEXT_MODEL_QUALITY_MEAN), context_is_best_freezed)
    mep.add(CONTEXT_IS_BEST_HOT, CONTEXT_ACTIVABLE, context_is_next_hot)
    mep.add(_D + CONTEXT_LEARNING_STATE,
            (CONTEXT_ACTIVABLE, CONTEXT_LEARNING_STATE, CONTEXT_MODEL_QUALITY_MEAN, CONTEXT_IS_BEST_HOT, CONTEXT_IS_BEST_FREEZED,
             CONTEXT_QUALITY_OUTER_THRESHOLD_DYN),
            d_context_is_learning,
            default_initial_values={CONTEXT_LEARNING_STATE: xp.zeros((1, context_num))})
    mep.add(CONTEXT_LEARNING_SELECTION, CONTEXT_LEARNING_STATE, context_is_learning)
    mep.add(CONTEXT_CONTROL_SELECTION, (CONTEXT_MODEL_QUALITY_MEAN, CONTEXT_LEARNING_SELECTION, CONTEXT_IS_BEST_FREEZED,
                                        CONTEXT_QUALITY_OUTER_THRESHOLD_DYN),
            context_is_active)
    mep.add(CONTEXT_COMMANDING_SELECTION, (CONTEXT_LEARNING_SELECTION, CONTEXT_CONTROL_SELECTION, CONTEXT_IS_BEST_FREEZED),
            context_is_commanding)

    # THRESHOLD CONTROL ----------------------------------------------------------------------------
    mep.add(_D + CONTEXT_QUALITY_INNER_THRESHOLD_DYN,
            (CONTEXT_QUALITY_INNER_THRESHOLD_DYN, CONTEXT_MODEL_QUALITY,
             CONTEXT_MODEL_QUALITY_MEAN, CONTEXT_MODEL_QUALITY_VARIANCE,
             CONTEXT_LEARNING_SELECTION, MODEL_LEARNING_RATE),
            get_d_inner_threshold(model_quality_lower_bound, model_quality_upper_bound, 0.001, 0.9),
            default_initial_values={CONTEXT_QUALITY_INNER_THRESHOLD_DYN: np.zeros((1,)) + model_quality_lower_bound}
            )

    if dynamic_outer_threshold:
        mep.add(_D + CONTEXT_QUALITY_OUTER_THRESHOLD_DYN,
                (CONTEXT_QUALITY_OUTER_THRESHOLD_DYN, REAL_PERFORMANCE_QUALITY,
                 CONTEXT_CONTROL_SELECTION, CONTEXT_LEARNING_SELECTION, MODEL_LEARNING_RATE),
                get_d_outer_threshold(upper_bound=model_quality_upper_bound,
                                      restart_rate=0.1,
                                      acceptable_performance=performance_quality_upper_bound,
                                      decay_rate=0.1),
                default_initial_values={CONTEXT_QUALITY_OUTER_THRESHOLD_DYN: np.zeros((1,)) + model_quality_upper_bound}
                )
    else:
        mep.add(CONTEXT_QUALITY_OUTER_THRESHOLD_DYN, (), lambda: np.zeros((1,)) + model_quality_upper_bound)

    # MOTOR CONTROL SIGNAL  ----------------------------------------------------------------------------
    mep.add(MOTOR_OUTPUT, (MOTOR_CONTEXT, MOTOR_EXPECTED), lambda ctx, exp: ctx + exp)

    # u_cmd_p (1, MOTOR_DIM, 1, NUM_MOTOR_CENTERS)
    mep.add(_D + MOTOR_COMMAND_PERTURBATION, (MOTOR_COMMAND_PERTURBATION, MOTOR_PHASE_ACTIVATOR),
            get_motor_binary_perturbation(perturbation_probability=perturbation_probability,
                                          update_rate=100, num_motor_centers=motor_segments_num),
            default_initial_values={MOTOR_COMMAND_PERTURBATION: xp.zeros((1, motor_dim, 1, motor_segments_num))})

    # u_cmd (1, MOTOR_DIM)
    if not motor_command_override:
        mep.add(MOTOR_COMMAND,
                (MOTOR_OUTPUT, MOTOR_COMMAND_PERTURBATION, MOTOR_PHASE_ACTIVATOR, CONTEXT_COMMANDING_SELECTION, CONTEXT_LEARNING_SELECTION),
                motor_command)


    # HYPER-PARAMETERS ----------------------------------------------------------------------------
    motor_centers = (2 * xp.pi * xp.arange(motor_segments_num)/motor_segments_num).reshape((1, motor_segments_num))
    mep.add(MOTOR_CENTERS, (), lambda: motor_centers)
    state_centers = (2 * xp.pi * xp.arange(sensory_segments_num)/sensory_segments_num).reshape((1, sensory_segments_num))
    mep.add(STATE_CENTERS, (), lambda: state_centers)
    epicycle_centers = (2 * xp.pi * xp.arange(epicycle_size)/epicycle_size).reshape((1, epicycle_size))
    mep.add(EPICYCLE_CENTERS, (), lambda: epicycle_centers)
    ## learning rates
    mep.add(MEMORY_LEARNING_RATE, "t", lambda t: 100.)
    if not is_control_lr_external:
        mep.add(MOTOR_LEARNING_RATE, "t", lambda t: .01)
    if not is_model_lr_external:
        mep.add(MODEL_LEARNING_RATE, "t", lambda t: .9)

    ## lc perturbations
    if not use_state_perturbation:
        mep.add(STATE_PERTURBATION, (), lambda: 0.)
    if not use_motor_perturbation:
        mep.add(MOTOR_PERTURBATION, (), lambda: 0.)

    # Epicycle synchronization
    if no_external_perturbation:
        mep.add(EPICYCLE_PERTURBATION, STATE_PERTURBATION, lambda st: st)

    # DEBUG/NONESSENTIAL SIGNALS ---------------------------------------------------------------
    # For the idea that the error should be scaled by motor_expected (further from context the more outside of prior)
    mep.add(_D + MOTOR_PRIOR_ERROR_FREE_ENERGY,
            (MOTOR_PRIOR_ERROR_FREE_ENERGY, MOTOR_EXPECTED, CONTEXT_CONTROL_SELECTION),
            get_d_fe_motor_prior_error(variance=1.),
            default_initial_values={
                MOTOR_PRIOR_ERROR_FREE_ENERGY:  xp.random.randn(1, motor_dim, 1, motor_segments_num, context_num)
            }
            )
    # Probability representation of the prediction error and the closeness to motor context. Good for analysis.
    mep.add(PROB_JOINT_SENSORYMOTOR, (PROB_VAR_SENSORY_MOTOR, SENSORY_MEMORY, SENSORY_ESTIMATION, MOTOR_EXPECTED), get_joint_norm(0.01))
    # Estimation of expected motor effect (not used anymore but good for analysis)
    mep.add(SENSORY_EFFERENT_ESTIMATION, (MODEL_BIAS, MODEL_WEIGHTS, MOTOR_EXPECTED), sensory_estimation)


# PHASE REGULATION ---------------------------------------------------

def parameter_phase_syncing_estimator(velocity, pert_eps=1., sync_speed=1.):
    def phase_estimator(phase, perturbation):
        return velocity - xp.sin(phase * sync_speed) * pert_eps * perturbation
    return phase_estimator


def parameter_rbf(epsilon, max_norm=False):
    def rbf(phase, center):
        dist = xp.abs(xp.exp(-1j*phase)-xp.exp(-1j*center))
        return xp.exp(-xp.square(epsilon * dist))

    if max_norm:
        def rbf_max(phase, center):
            rbf_signal = rbf(phase, center)
            ret = xp.zeros(rbf_signal.shape)
            ret[0, xp.argmax(rbf_signal)] = 1
            return ret
        return rbf_max
    return rbf


# ESTIMATION ---------------------------------------------------

def sensory_estimation(bias, weights, motor_diff):
    return bias + xp.einsum("nmcdx,injcx->imjdx", weights, motor_diff)


def _motor_diff_doubling(motor_diff):
    # injck -> injqk
    doubled_diff = xp.zeros(
        (motor_diff.shape[0], motor_diff.shape[1], motor_diff.shape[2],
         motor_diff.shape[3] * 2, motor_diff.shape[4]))

    # copy to lower half
    doubled_diff[:, :, :, motor_diff.shape[3]:, :] += motor_diff
    # upper half contains past epicycle this shifted to left
    doubled_diff[:, :, :, :motor_diff.shape[3], 1:] += motor_diff[:, :, :, :, :motor_diff.shape[4] - 1]
    # first epicycle is consequence of the last one
    doubled_diff[:, :, :, :motor_diff.shape[3], 0] += motor_diff[:, :, :, :, motor_diff.shape[4] - 1]
    return doubled_diff


def get_epicycle_estimator(state_num):
    shift_matricies = xp.zeros((state_num, state_num, state_num * 2))
    for i in range(state_num):
        shifted_identity = xp.roll(xp.identity(state_num), -(i + 1), axis=1)
        shift_matricies[i, :, i + 1:i+state_num+1] += shifted_identity

    def _ep_estimator_dyn(bias, weights, motor_diff, ctx_lr):
        y = xp.zeros((
            1, weights.shape[1], 1, weights.shape[3], motor_diff.shape[4], weights.shape[4]
        ))
        win_idx = xp.argmax(ctx_lr[0])  # winner. If no win then the computation must be done anyways.
        _motor_diff = motor_diff[:, :, :, :, :, win_idx]
        _weights = weights[:, :, :, :, win_idx]
        _bias = bias[:, :, :, :, win_idx]
        # ---------------double motor_diff
        doubled_diff = _motor_diff_doubling(_motor_diff)

        # ------------------shift weights
        shifted_weight = xp.einsum("nmcd,dcq->nmqd", _weights, shift_matricies, optimize=False)

        #compute estimations
        y[:, :, :, :, :, win_idx] = xp.einsum("nmqd,injqk->imjdk", shifted_weight, doubled_diff, optimize=False)
        y[:, :, :, :, :, win_idx] += _bias[:, :, :, :, None]
        return y
    return _ep_estimator_dyn


# ERRORS ---------------------------------------------------


def true_motor_error(sensory_reference, _sensory_memory, epicycle_activator):
    """
    Returns motor error of previous gait.
    This is because the current gait memory is not valid.
    :param sensory_reference:
    :param _sensory_memory:
    :param epicycle_activator:
    :return:
    """
    epicycle_id = np.argmax(epicycle_activator)
    if epicycle_id == 0:
        prev_epicycle_id = epicycle_activator.shape[-1] - 1
    else:
        prev_epicycle_id = epicycle_id - 1

    return sensory_reference - _sensory_memory[:,:,:,:, prev_epicycle_id]


def d_fe_motor_error(fe_motor_error, motor_error, sensory_posterior_variances, cmd_sel):
    cmd_id = np.argmax(cmd_sel)
    return np.clip((motor_error - fe_motor_error * np.maximum(sensory_posterior_variances[:, :, :, :, cmd_id], 0.00001))*100, a_min=-50., a_max=50.)


def get_d_fe_motor_prior_error(variance=1.):
    def dyn(fe_motor_prior_error, u_exp, ctx_ctr_sel):
        sel_id = np.argmax(ctx_ctr_sel)
        any_id = np.sum(sel_id) > 0
        ret = xp.zeros(fe_motor_prior_error.shape)
        diff = u_exp[:, :, :, :, sel_id] - fe_motor_prior_error[:, :, :, :, sel_id] * variance
        if any_id:
            ret[:, :, :, :, sel_id] = diff
        else:
            ret[:, :, :, :, sel_id] += 0.
        return ret
    return dyn


def model_error(_sensory_memory, _sensory_estimation, epicycle_activator):
    """
    Model error but filters the model error of the previous gait.
    This is because the current gait memory is not valid, thus it could introduce inconsistencies.
    :param _sensory_memory:
    :param _sensory_estimation:
    :param epicycle_activator:
    :return:
    """
    return xp.einsum("imjdkx,ik->imjdkx", _sensory_memory[:, :, :, :, :, None] - _sensory_estimation,
                     1 - np.roll(epicycle_activator, 1, axis=1))


# MEMORY ---------------------------------------------------


def d_motor_memory(_motor_memory, motor_command, motor_phase_activator, epicycle_activator, memory_learning_rate):
    return xp.einsum("injck,ic,ik->injck",
                     -_motor_memory + motor_command[:, :, None, None, None], motor_phase_activator, epicycle_activator) * memory_learning_rate


def d_sensory_memory(_sensory_memory, _sensory_input, state_phase_activator, epicycle_activator, memory_learning_rate):
    return xp.einsum("imjdk,id,ik->imjdk",
                     -_sensory_memory + _sensory_input[:, :, None, None, None], state_phase_activator, epicycle_activator) * memory_learning_rate


# BIAS ---------------------------------------------------


def d_bias(model_err, state_phase_activator, model_lr, context_learning_sel):
    return xp.einsum("imjdkx,id,ix->imjdx", model_err, 1 - state_phase_activator, context_learning_sel) * model_lr / model_err.shape[-2] * 0.001,


# WEIGHT ----------------------|----------------------------
def _error_doubling(model_err, motor_diff_shape_3):
    doubled_error = xp.zeros(
        (model_err.shape[0], model_err.shape[1], model_err.shape[2],
         model_err.shape[3] * 2, model_err.shape[4]))
    # copy to upper half
    doubled_error[:, :, :, :model_err.shape[3], :] += model_err
    # lower half contains future epicycle thus shifted to the right
    doubled_error[:, :, :, model_err.shape[3]:, :model_err.shape[4] - 1] += model_err[:, :, :, :, 1:]
    # last epicycle is continues to first one
    doubled_error[:, :, :, :motor_diff_shape_3, model_err.shape[4] - 1] += model_err[:, :, :, :, 0]
    return doubled_error


def get_d_weights(state_num):
    # cQd
    shift_matricies = xp.zeros((state_num, state_num, state_num * 2))
    for c in range(state_num):
        shifted_identity = xp.roll(xp.identity(state_num), -c, axis=1)
        shift_matricies[c, :, c:c + state_num] += shifted_identity

    def _weights_dyn(w, model_err, motor_diff, state_phase_activator, motor_phase_activator, model_lr, ctx_lr):
        # nmcdx
        d_weight = xp.zeros((motor_diff.shape[1], model_err.shape[1], motor_diff.shape[3], model_err.shape[3], model_err.shape[5]))
        win_idx = xp.argmax(ctx_lr[0])  # winner. If no win then the computation must be done anyways.

        _model_err = model_err[:, :, :, :, :, win_idx]
        _motor_diff = motor_diff[:, :, :, :, :, win_idx]
        # ---------------double the error
        doubled_error = _error_doubling(_model_err, _motor_diff.shape[3])
        # ---------------product with motor-diff
        enhanced_mat = xp.einsum("imjqk,injck->nmcq", doubled_error, _motor_diff)
        # ---------------filter into matrix-shape
        # nmcq -> nmcd
        ret = xp.einsum("nmcq,cdq,id,jc->nmcd", enhanced_mat, shift_matricies,
                         1 - state_phase_activator, 1 - motor_phase_activator, optimize=True) * model_lr / model_err.shape[-2] * 0.01
        d_weight[:, :, :, :, win_idx] += xp.clip(ret, a_min=-5., a_max=5.) * ctx_lr[:, win_idx]
        return d_weight
    return _weights_dyn


# MOTOR --------------------------------------------------

def get_d_motor_expected(u_exp_act_lr):
    def d_motor_expected(motor_err, weights, motor_lr, ctx_ctr_sel, regularization, decay=0.):
        active = xp.einsum("imjd,nmcdx,ix->injcx", motor_err, weights, ctx_ctr_sel, optimize=True) * motor_lr
        if np.sum(ctx_ctr_sel) > 0:
            return active * u_exp_act_lr - regularization * 0.01 - decay * 0.00001
        else:
            return np.zeros(active.shape)
    return d_motor_expected


def motor_command(_motor_output, motor_command_perturbation, motor_phase_activator, commanding_sel, ctx_ctr_lr):
    system_is_learning = np.sum(ctx_ctr_lr) > 0
    suma = xp.einsum("imjdx,ix->imjd", _motor_output, commanding_sel)
    if system_is_learning:
        suma = xp.einsum("imjd,id->im",suma + xp.maximum(0, motor_command_perturbation) * 1.,
                         motor_phase_activator)
    else:
        suma = xp.einsum("imjd,id->im", suma, motor_phase_activator)
    return suma


def error_masking_clipping(motor_err, reference_mask, minmax_mask):
    max_mask = minmax_mask >= 0
    min_mask = minmax_mask <= 0
    active = xp.einsum("imjd,imjd->imjd", motor_err, reference_mask)
    ret = xp.einsum("imjd,imjd->imjd", np.maximum(active, 0), max_mask) + xp.einsum("imjd,imjd->imjd", np.minimum(active, 0), min_mask)
    return ret

# MISC ---------------------------------------------------------------


def get_motor_binary_perturbation(perturbation_probability, update_rate, num_motor_centers):
    shift_matrix = xp.roll(xp.identity(num_motor_centers), num_motor_centers//2, axis=1)

    def motor_binary_perturbation(_motor_binary_perturbation, motor_phase_activator):
        pert = xp.random.binomial(1, perturbation_probability, _motor_binary_perturbation.shape)
        ret = pert - perturbation_probability
        activation = motor_phase_activator.dot(shift_matrix)
        return xp.einsum("injc,ic->injc", ret - _motor_binary_perturbation, activation) * update_rate
    return motor_binary_perturbation


# OPTIMIZATION ---------------------------------------------------------
def rms_prop_learning_rate(ro):
    def d_learning_rate(nu, grad):
       return (1 - ro) * (xp.square(grad) - nu)
    return d_learning_rate


def rms_prop_grad(lr, eps):
    def d_grad(nu, grad):
        return lr/xp.sqrt(nu + eps) * grad
    return d_grad
# CONTEXT LEARNING ------------------------------------------------------


def prev_y_est(u_mem, u_ctx, w, b, state_phase, epicycle_activator):
    state_id = xp.argmax(state_phase)

    cur_epicycle_id = xp.argmax(epicycle_activator)
    prev_epicycle_id = (cur_epicycle_id - 1) % epicycle_activator.shape[-1]

    # 1, N, 1, C, period of motor memory
    motor_memory = xp.zeros((u_mem.shape[0:4]))
    motor_memory[0, :, 0, state_id:] = u_mem[0, :, 0, state_id:, prev_epicycle_id]
    motor_memory[0, :, 0, :state_id] = u_mem[0, :, 0, :state_id, cur_epicycle_id]

    motor_diff = - u_ctx + motor_memory[:, :, :, :, None]

    # selecting correct weight vector for the phase
    trg_state_id = (state_id - 1) % state_phase.shape[-1]
    _w = w[:,:,:,trg_state_id]

    y_est = xp.einsum("nmcx,injcx->imjx", _w, motor_diff)
    y_est += b[:, :, :, trg_state_id, :]
    return y_est


def prev_y_obs(y_mem, state_phase, epicycle_activator):
    """
    Gets sensor-phase from previous phase
    :param y_mem: sensor-phases memory
    :param state_phase: current state phase
    :param epicycle_activator: current epicycle
    :return:
    """
    state_id = xp.argmax(state_phase)
    epicycle_id = xp.argmax(epicycle_activator)
    if state_id == 0:
        # if it is zeroeth phase, then the previous phase comes from the end of previous epicycle
        trg_state_id = state_phase.shape[1] - 1
        trg_epicycle_id = (epicycle_id - 1) % epicycle_activator.shape[-1]
    else:
        # else we use current epicycle with previous phase.
        trg_state_id = state_id - 1
        trg_epicycle_id = epicycle_id
    return y_mem[:, :, :, trg_state_id, trg_epicycle_id]


def context_motor_error(prev_y_obs, prev_y_est):
    return prev_y_obs[:, :, :, None] - prev_y_est


def model_quality_current_normed(ctx_e_mod, learn_e_mod, sensory_posterior_variance, lrn_sel, state_phase_sel):
    """

    :param ctx_e_mod: shortened error of previous observation
    :param learn_e_mod:  full error computed from updated memory and estimations
    :param sensory_posterior_variance:
    :param lrn_sel:
    :param state_phase_sel:
    :return:
    """
    # getting id of last full epoch
    state_id = np.argmax(state_phase_sel)
    prev_state_id = (state_id - 1) % state_phase_sel.shape[-1]

    # normalized error
    e_m = xp.mean(xp.square(ctx_e_mod)/sensory_posterior_variance[:, :, :, prev_state_id], axis=(0,1,2))

    # normalized updated error for currently learned system
    lrn_e_m = xp.mean(xp.square(learn_e_mod)/sensory_posterior_variance[:, :, :, :, None, :], axis=(0,1,2,3,4))
    lrn_id = np.argmax(lrn_sel)
    e_m[lrn_id] = lrn_e_m[lrn_id]

    return e_m.reshape((1, ctx_e_mod.shape[-1]))


def get_model_quality_mean(q_mean_down_lr, q_mean_up_lr):
    def model_quality_mean(quality_mean, quality):
        goes_down = quality < quality_mean
        goes_up = quality >= quality_mean
        return (quality - quality_mean) * (goes_down * q_mean_down_lr + goes_up * q_mean_up_lr)
    return model_quality_mean


def model_qiality_variance(quality_variance, quality_mean, quality):
    delt = (quality - quality_mean)
    return (np.power(delt, 2) - quality_variance) * 0.01


def context_is_freezed(freezed, mod_quality, inner_threshold, lr):
    is_hot = (1 - freezed) * lr
    is_good = (mod_quality < inner_threshold)
    return is_hot * is_good


def context_is_best_freezed(freezed, mod_quality):
    ret = xp.zeros((1, freezed.shape[-1]))
    is_freezed = (freezed > 0.5)
    sel = xp.argmin(mod_quality + (1 - is_freezed) * 100)  # best quality + hot penalty
    ret[0, sel] = is_freezed[0, sel]  # 1 iff the best is actually freezed
    return ret


def context_is_next_hot(freezed):
    ret = xp.zeros((1, freezed.shape[-1]))
    is_not_freezed = (freezed <= 0.5)
    sel = xp.argmax(is_not_freezed[0, :])  # best quality + freezed penalty
    ret[0, sel] = is_not_freezed[0, sel]  # 1 iff the best is actually hot
    return ret


def context_is_learning(learning_state):
    return learning_state > 0.5


def d_context_is_learning(freezed, learning_state, mod_quality, best_hot, best_freezed, outer_threshold):
    is_some_freezed = np.sum(best_freezed) > 0
    is_learned = learning_state > 0.5
    is_not_freezed = freezed < 0.5

    if is_some_freezed:
        # if some context is freezed
        best_quality = np.sum(best_freezed * mod_quality)
        best_freezed_unpassable = best_quality > outer_threshold
    else:
        # if none context is freezed then it is as if the best is unpassable
        best_freezed_unpassable = 1

    if np.sum(is_learned) > 0:
        # if some context is learned, this context continues learning ubless it is in inner
        # push = (1 - learning_state) * best_hot * is_not_inner (bad logic)
        # if some context is learned, this context continues learning unless it is freezed
        push = (1 - learning_state) * best_hot * is_not_freezed
    else:
        # if no context is learning yet and best freezed is unpassable, then pick the best hot
        push = (1 - learning_state) * best_hot * best_freezed_unpassable

    # learning state push is combined with decay
    return - learning_state * 0.5 + push * 0.8


def context_is_active(mod_quality, is_learning, best_freezed, outer_threshold):
    system_is_learning = np.sum(is_learning) > 0
    # only passable freezed context can optimize its u_exp
    best_quality = np.sum(best_freezed*mod_quality)
    if best_quality <= outer_threshold:
        # best freezed is active iff is pasable and system is not learning
        return best_freezed * (1 - system_is_learning)
    else:
        return best_freezed * 0


def context_is_commanding(learning, active, best_freezed):
    priority = learning + active # either learning or active or none has nonzero vector
    is_priority = np.sum(priority) > 0
    if is_priority:
        return priority
    else:
        # if all is freezed but none is passable then use atleast the best
        return best_freezed


# MOTOR CONTEXT


def model_motor_context_follow_u_out(motor_context, u_out, freezed, learning, cmd_sel):
    cmd_id = xp.argmax(cmd_sel)
    any_command = xp.sum(cmd_sel) > 0
    is_hot = (freezed <= 0.5)

    if any_command:
        _u_out = u_out[:, :, :, :, cmd_id]
        ret = xp.einsum("injcx,ix,ix->injcx", _u_out[:, :, :, :, None] - motor_context, is_hot, 1 - learning) * 0.05
    else:
        ret = xp.einsum("injcx,ix,ix->injcx", -motor_context, is_hot, 1 - learning) * 0.001

    return ret

# ---
def motor_diff(motor_mem, motor_context):
    ret = xp.zeros((motor_mem.shape[0], motor_mem.shape[1], motor_mem.shape[2], motor_mem.shape[3],
                    motor_mem.shape[4], motor_context.shape[-1]))
    ret += motor_mem[:, :, :, :, :, None]
    ret -= motor_context[:, :, :, :, None]
    return ret


def get_d_inner_threshold(lower_bound, upper_bound, threshold_grow_rate, threshold_descend_rate):
    def sys(inner_threshold, model_quality, _model_quality_mean, model_quality_var, lrn_sel, lr):
        is_any_learning = xp.sum(lrn_sel) > 0
        lrn_sel_id = xp.argmax(lrn_sel)

        lrn_q_mean = _model_quality_mean[:, lrn_sel_id]
        lrn_q_var = model_quality_var[:, lrn_sel_id]
        upper_bnd = model_quality[:, lrn_sel_id] + xp.sqrt(lrn_q_var) / 10

        if is_any_learning and lrn_q_mean < upper_bound:
            return inner_threshold * (upper_bnd > lrn_q_mean) * lr * threshold_grow_rate
        return (lower_bound - inner_threshold) * threshold_descend_rate
    return sys


def performance_quality(e_mot_flt):
    return np.mean(np.square(e_mot_flt))


def get_d_outer_threshold(upper_bound, restart_rate, acceptable_performance, decay_rate):
    def sys(outer_threshold, q_real, ctr_sel, lrn_sel, lr):
        is_any_controlling = np.sum(ctr_sel) > 0
        is_any_learning = np.sum(lrn_sel) > 0
        is_acceptable_performance = q_real < acceptable_performance

        if is_any_controlling and not is_acceptable_performance:
            return -outer_threshold * lr * decay_rate
        if is_any_learning:
            return (upper_bound - outer_threshold) * restart_rate
        return (upper_bound - outer_threshold) * restart_rate * 0.01  # slower restart during control
    return sys


## Probabilistic scaling
def get_d_sensory_posterior_variance(learning_rate=0.1):
    def sys(var_ys, model_err, context_learning_sel, model_learning):
        # var_ys (IMJDX)
        ret = np.zeros(var_ys.shape)
        is_any_learning = np.sum(context_learning_sel) > 0
        learned_id = np.argmax(context_learning_sel)
        # to keep time consistent computation must be done atleast for one
        _model_err = model_err[:, :, :, :, :, learned_id]
        cur_var_ys = np.mean(np.square(_model_err), axis=4)
        ret[:, :, :, :, learned_id] = (cur_var_ys - var_ys[:, :, :, :, learned_id]) * model_learning * learning_rate
        if is_any_learning:
            return ret
        else:
            return ret * 0

    return sys


def get_joint_norm(u_variance):
    """
    Computes the normal distributions p(u,y) = p(u|0, u_variance) * p(y|Wu+b, var_y)
    0 - p(u,y)
    1 - p(u|0, u_variance)
    2 - p(y|Wu+b, var_y)
    :param u_variance:
    :return:
    """
    def expr(var_ys, y_true, y_pred, u_exp):
        ret = np.zeros((1,3,var_ys.shape[-1]))
        u_dist = np.sum(np.square(u_exp), axis=(0,1,2,3)) * (1/u_variance)
        y_diff = np.mean(y_true[:,:,:,:,:,None] - y_pred, axis=4)
        y_dist = np.einsum("imjdx,imjdx->x", np.square(y_diff), 1/np.clip(var_ys, a_min=0.00001, a_max=None))
        ret[0, 1, :] = np.exp(-0.5 * u_dist)
        ret[0, 2, :] = np.exp(-0.5 * y_dist)
        ret[0, 0, :] = ret[0, 1, :] * ret[0, 2, :]

        return ret
    return expr