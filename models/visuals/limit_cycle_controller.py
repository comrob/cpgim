from matplotlib.pyplot import Figure
import models.limit_cycle_controller_contextual as M
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def _get_period(array_1d):
    res = np.correlate(array_1d, array_1d, mode="full")
    peaks, _ = signal.find_peaks(res)
    avg_diff = np.average(peaks[1:] - peaks[0:-1])
    return int(avg_diff)


def _get_phase_evol_slice(x_short_form, sample_times):
    phases = x_short_form.shape[-1]
    phase_evol = []
    for t in sample_times:
        phase_evol += [x_short_form[t, :, p] for p in range(phases)]
    return phase_evol


def _get_phase_samples_from_signal(signal_slice, phase_activator, threshold=0.9):
    return [signal_slice[phase_activator[:, 0, p] > threshold, :] for p in range(phase_activator.shape[-1])]


def _get_nmcd_shape(record):
    w = record[M.MODEL_WEIGHTS]
    return w[0, :, :, :, :].shape


def _get_step(record):
    t = record["t"]
    return t[1]-t[0]


def _has_epicycle(record):
    return record[M.MODEL_ERROR].ndim == 6


def _convolved_square_error(record, hann_win=1000):
    a_s = record[M.STATE_PHASE_ACTIVATOR]
    a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR]
    y_est = record[M.SENSORY_ESTIMATION]
    y_inp = record[M.SENSORY_INPUT]
    # combine
    y_est_cmb = np.einsum("timjck,tic,tik->tim", y_est, a_s, a_ep)

    win = signal.windows.hann(hann_win)
    filtered = signal.convolve(np.average(np.square(y_inp - y_est_cmb), axis=-1)[:,0], win, mode='same') / sum(win)
    return filtered


def state_evolution(record, fig: Figure):

    fig.suptitle("Parameter evolution and estimation comparison")
    t = record["t"]
    state_phase = record[M.STATE_PHASE_ESTIMATION][:, 0]
    motor_phase = record[M.MOTOR_PHASE_CONTROLLER][:, 0]
    a_s = record[M.STATE_PHASE_ACTIVATOR][:,0,:]
    a_m = record[M.MOTOR_PHASE_ACTIVATOR][:,0,:]


    f = fig.subplots(2, 2)
    fig_evol_state = f[0][0]
    fig_evol_motor = f[1][0]
    fig_activator_state = f[0][1]
    fig_activator_motor = f[1][1]

    fig_evol_state.plot(t, state_phase[:, 0], label=M.STATE_PHASE_ESTIMATION)
    fig_evol_motor.plot(t, motor_phase[:, 0], label=M.MOTOR_PHASE_CONTROLLER)
    fig_evol_state.legend()
    fig_evol_motor.legend()

    fig_activator_state.plot(t, a_s, label=M.STATE_PHASE_ACTIVATOR)
    fig_activator_motor.plot(t, a_m, label=M.MOTOR_PHASE_ACTIVATOR)
    fig_activator_state.legend()
    fig_activator_motor.legend()


def sensory_parameters(record, fig: Figure, start_from_period=-3, num_of_periods=2):
    fig.suptitle("Sensory parameters")
    t = record["t"]
    y_ref = record[M.SENSORY_REFERENCE][:, 0, :, 0, :]
    y_inp = record[M.SENSORY_INPUT][:, 0, :]
    y_est = record[M.SENSORY_ESTIMATION][:, 0, :, 0, :]
    b_mod = record[M.MODEL_BIAS][:, 0, :, 0, :]
    a_s = record[M.STATE_PHASE_ACTIVATOR]


    shape = _get_nmcd_shape(record)
    sensor_num = shape[1]
    f = fig.subplots(sensor_num, 1)
    period = _get_period(a_s[:, 0, 1])
    start_t = period * start_from_period
    t_ints = [(start_t + i * period, start_t + ((i + 1) * period)) for i in range(num_of_periods)]

    sensory_phase_samples = []
    for ts, te in t_ints:
        sensory_phase_samples += _get_phase_samples_from_signal(y_inp[ts:te], a_s[ts:te])
    t_ends = [te for ts, te in t_ints]

    for i in range(sensor_num):
        if sensor_num == 1:
            figr = f
        else:
            figr = f[i]
        data_y_est = _get_phase_evol_slice(y_est, t_ends)
        data_b_mod = _get_phase_evol_slice(b_mod, t_ends)
        data_y_ref = _get_phase_evol_slice(y_ref, t_ends)
        data_y_inp = [np.average(samp[:, i]) for samp in sensory_phase_samples]
        figr.plot(data_y_est, '--o', label=M.SENSORY_ESTIMATION)
        figr.plot(data_b_mod, '--^', label=M.MODEL_BIAS)
        figr.plot(data_y_ref, '--s', label=M.SENSORY_REFERENCE)
        figr.plot(data_y_inp, 'kx', label=M.SENSORY_INPUT + " avg")

    if sensor_num == 1:
        f.legend()
    else:
        f[-1].legend()


def motor_parameters(record, fig: Figure, start_from_period=-3, num_of_periods=2):
    fig.suptitle("Motor parameters")
    u_mem = record[M.MOTOR_MEMORY][:, 0, :, 0, :]
    u_cmd = record[M.MOTOR_COMMAND][:, 0, :]
    u_exp = record[M.MOTOR_EXPECTED][:, 0, :, 0, :]
    a_m = record[M.MOTOR_PHASE_ACTIVATOR]

    shape = _get_nmcd_shape(record)
    motor_num = shape[0]
    f = fig.subplots(motor_num, 1)
    period = _get_period(a_m[:, 0, 1])
    start_t = period * start_from_period
    t_ints = [(start_t + i * period, start_t + ((i + 1) * period)) for i in range(num_of_periods)]

    sensory_phase_samples = []
    for ts, te in t_ints:
        sensory_phase_samples += _get_phase_samples_from_signal(u_cmd[ts:te], a_m[ts:te])
    t_ends = [te for ts, te in t_ints]

    for i in range(motor_num):
        if motor_num == 1:
            figr = f
        else:
            figr = f[i]
        data_u_mem = _get_phase_evol_slice(u_mem, t_ends)
        data_u_exp = _get_phase_evol_slice(u_exp, t_ends)
        data_u_cmd = [np.average(samp[:, i]) for samp in sensory_phase_samples]
        figr.plot(data_u_mem, '--o', label=M.MOTOR_MEMORY)
        figr.plot(data_u_exp, '--^', label=M.MOTOR_EXPECTED)
        figr.plot(data_u_cmd, 'kx', label=M.MOTOR_COMMAND + " avg")

    if motor_num == 1:
        f.legend()
    else:
        f[-1].legend()


def weights_matricies(record, fig: Figure, start_from_period=-1):
    period = _get_period(record[M.MOTOR_PHASE_ACTIVATOR][:, 0, 1])
    t = -1 + ((start_from_period + 1) * period)
    fig.suptitle("Weights from t={}".format(t))
    W_mod = record[M.MODEL_WEIGHTS][t, :, :, :, :]

    _N, _M, _C, _D = W_mod.shape
    f = fig.subplots(_N, _M)
    for n in range(_N):
        for m in range(_M):
            if _N == 1 and _M == 1:
                figr = f
            elif _N == 1:
                figr = f[n]
            elif _M == 1:
                figr = f[m]
            else:
                figr = f[n][m]

            figr.matshow(W_mod[n, m, :, :])
            figr.set_xlabel("state phase")
            figr.set_ylabel("motor phase")

            if n == (_N - 1):
                figr.set_xlabel("m:{}".format(m))
            if m == 0:
                figr.set_ylabel("n:{}".format(n))


def weight_matrix_analysis(record, fig: Figure, start_from_period=-1, sensor_m=0, motor_n=0):
    period = _get_period(record[M.MOTOR_PHASE_ACTIVATOR][:, 0, 1])
    t = -1 + ((start_from_period + 1) * period)
    fig.suptitle("Weights from t={} of M={}, N={}".format(t, sensor_m, motor_n))
    W_mod = record[M.MODEL_WEIGHTS][t, :, :, :, :]

    w = W_mod[motor_n, sensor_m, :, :]
    f = fig.subplots(2, 2)
    f[0][0].matshow(w)
    f[0][0].set_xlabel("state phase D")
    f[0][0].set_ylabel("motor phase C")
    f[1][0].matshow(np.sum(w, axis=0, keepdims=True))
    f[0][1].matshow(np.sum(w, axis=1, keepdims=True))


def bias_evolution(record, fig: Figure, sensor_m=0):
    fig.suptitle("Bias evolution of sensor {}".format(sensor_m))
    b_mod = record[M.MODEL_BIAS]
    coln = 1000
    _T,_,_M,_,_D = b_mod.shape
    rows = _T//coln + 1
    evol = np.zeros((rows * _D, coln))
    for i in range(rows):
        seq = b_mod[coln*i:coln*(i + 1), 0, sensor_m, 0, :]
        evol[_D * i:_D * (i+1), :seq.shape[0]] = seq.T
    f = fig.subplots(1, 1)
    f.matshow(evol)
    f.set_yticks([i*_D for i in range(rows)])
    f.set_yticklabels([i*coln for i in range(rows)])


def expected_motor_parameters_evolution(record, fig: Figure, motor_n=0, title_on=True):
    if title_on:
        fig.suptitle("Expected motor parameters evolution of {}".format(motor_n))
    u_exp = record[M.MOTOR_EXPECTED]
    coln = 1000
    _C = u_exp.shape[-1]
    _T = u_exp.shape[0]
    rows = _T//coln + 1
    evol = np.zeros((rows * _C, coln))
    for i in range(rows):
        seq = u_exp[coln*i:coln*(i + 1), 0, motor_n, 0, :]
        evol[_C * i:_C * (i+1), :seq.shape[0]] = seq.T
    f = fig.subplots(1, 1)
    f.matshow(evol)
    f.set_yticks([i * _C for i in range(rows)])
    f.set_yticklabels([i * coln for i in range(rows)])

def jumpy_expected_motor_parameters_evolution(record, fig: Figure, motor_n=0, title_on=False):
    if title_on:
        fig.suptitle("Expected motor parameters evolution of {}".format(motor_n))
    u_exp = record[M.MOTOR_EXPECTED]
    coln = 200
    _C = u_exp.shape[-1]
    _T = u_exp.shape[0]
    jmp = _T//coln + 1
    f = fig.subplots(1, 1)
    f.matshow(u_exp[[i*jmp for i in range(coln)], 0, motor_n, 0, :].T,  cmap=plt.cm.get_cmap("seismic"))
    f.set_yticks([])
    f.set_xticks([])
    # f.set_yticklabels([i * coln for i in range(rows)])


def expected_motor_evolution(record, fig: Figure, motor_n=0):
    fig.suptitle("Expected command evolution of motor {}".format(motor_n))
    u_exp = record[M.MOTOR_EXPECTED]
    coln = 1000
    _T,_,_N,_,_C = u_exp.shape
    rows = _T//coln + 1
    evol = np.zeros((rows * _C, coln))
    for i in range(rows):
        seq = u_exp[coln*i:coln*(i + 1), 0, motor_n, 0, :]
        evol[_C * i:_C * (i+1), :seq.shape[0]] = seq.T
    f = fig.subplots(1, 1)
    f.matshow(np.power(evol, 2))
    f.set_yticks([i*_C for i in range(rows)])
    f.set_yticklabels([i*coln for i in range(rows)])


def model_parameter_convergence(record, fig: Figure):
    fig.suptitle("Model Parameter Convergence")
    t = record["t"]

    W_mod = record[M.MODEL_WEIGHTS]
    u_exp = record[M.MOTOR_EXPECTED][:, 0, :, 0, :]
    b_mod = record[M.MODEL_BIAS][:, 0, :, 0, :]
    e_mot = record[M.MOTOR_ERROR][:, 0, :, 0, :]
    if _has_epicycle(record):
        e_mod = record[M.MODEL_ERROR][:, 0, :, 0, :, :]
    else:
        e_mod = record[M.MODEL_ERROR][:, 0, :, 0, :]

    f = fig.subplots(3, 2)
    _N, _M, _C, _D = _get_nmcd_shape(record)
    fig_W_mod = f[0][0]
    fig_b_mod = f[1][0]
    fig_e_mod = f[2][0]

    fig_u_exp = f[0][1]
    fig_e_mot = f[2][1]

    for n in range(_N):
        for m in range(_M):
            fig_W_mod.plot(t, np.sum(np.sum(np.abs(W_mod[:, n, m, :, :]), axis=1), axis=1),
                           label=M.MODEL_WEIGHTS + "[n:{},m:{}]".format(n, m))
    fig_W_mod.legend()

    for m in range(_M):
        fig_b_mod.plot(t, np.sum(b_mod[:, m, :], axis=1), label=M.MODEL_BIAS + "[m:{}]".format(m))
        fig_b_mod.legend()

    for n in range(_N):
        fig_u_exp.plot(t, np.sum(u_exp[:, n, :], axis=1), label=M.MOTOR_EXPECTED + "[n:{}]".format(n))
        fig_u_exp.legend()

    for m in range(_M):
        if _has_epicycle(record):
            # fig_e_mod.plot(t, np.sum(e_mod[:, m, :, :], axis=1), 'k', alpha=0.3)
            fig_e_mod.plot(t, np.average(np.sum(np.square(e_mod[:, m, :, :]), axis=1), axis=1), label=M.MODEL_ERROR + "[avg m:{}]".format(m))
        else:
            fig_e_mod.plot(t, np.sum(e_mod[:, m, :], axis=1), label=M.MODEL_ERROR + "[m:{}]".format(m))
        fig_e_mod.legend()

    for m in range(_M):
        fig_e_mot.plot(t, np.sum(e_mot[:, m, :], axis=1), label=M.MOTOR_ERROR + "[m:{}]".format(m))
        fig_e_mot.legend()


def weight_evolution(record, fig: Figure, weight_nmcd_list=((0, 0, 0, 0),)):
    fig.suptitle("Weight evols")
    t = record["t"]
    W_mod = record[M.MODEL_WEIGHTS]
    if _has_epicycle(record):
        e_mod = record[M.MODEL_ERROR][:, 0, :, 0, :, :]
        u_dif = record[M.MOTOR_DIFF][:, 0, :, 0, :, :]
    else:
        e_mod = record[M.MODEL_ERROR][:, 0, :, 0, :]
        u_dif = record[M.MOTOR_DIFF][:, 0, :, 0, :]
    a_s = record[M.STATE_PHASE_ACTIVATOR][:, 0, :]

    f = fig.subplots(len(weight_nmcd_list), 1)
    for i, index in enumerate(weight_nmcd_list):
        if len(weight_nmcd_list) == 1:
            figr = f
        else:
            figr = f[i]
        n, m, c, d = index
        figr.set_title("n:{},m:{},c:{},d:{}".format(n, m, c, d))
        figr.plot(t, W_mod[:, n, m, c, d], label=M.MODEL_WEIGHTS + "[n,m,c,d]")
        if _has_epicycle(record):
            jmp = 0.7/e_mod.shape[-1]
            for k in range(e_mod.shape[-1]):
                if k == e_mod.shape[-1] - 1:
                    figr.plot(t, e_mod[:, m, d, k], 'r--', label=M.MODEL_ERROR + "[m,d]")
                    figr.plot(t, u_dif[:, n, c, k], 'm--', label=M.MOTOR_DIFF + "[n,c]")

                else:
                    figr.plot(t, e_mod[:, m, d, k], 'r--', alpha=k * jmp + 0.3)
                    figr.plot(t, u_dif[:, n, c, k], 'm--', alpha=k * jmp + 0.3)

        else:
            figr.plot(t, e_mod[:, m, d], label=M.MODEL_ERROR + "[m,d]")
            figr.plot(t, u_dif[:, n, c], label=M.MOTOR_DIFF + "[n,c]")
        figr.plot(t, a_s[:, d], label=M.STATE_PHASE_ACTIVATOR + "[d]")

    if len(weight_nmcd_list) == 1:
        f.legend()
    else:
        f[-1].legend()


def estimation_evolution(record, fig: Figure, md_list=((0, 0),)):
    fig.suptitle("Estimation evols")
    t = record["t"]
    y_est = record[M.SENSORY_ESTIMATION][:, 0, :, 0, :]
    y_inp = record[M.SENSORY_INPUT][:, 0, :]
    W_mod = record[M.MODEL_WEIGHTS]
    u_diff = record[M.MOTOR_DIFF][:, 0, :, 0, :]
    b_mod = record[M.MODEL_BIAS][:, 0, :, 0, :]
    a_s = record[M.MOTOR_PHASE_ACTIVATOR][:, 0, :]
    y_mem = record[M.SENSORY_MEMORY][:, 0, :, 0, :]

    f = fig.subplots(len(md_list), 1)
    for i, index in enumerate(md_list):
        if len(md_list) == 1:
            figr = f
        else:
            figr = f[i]
        m, d, = index
        figr.set_title("m:{},d:{}".format(m, d))
        figr.plot(t, np.sum(np.sum(W_mod[:, :, m, :, d] * u_diff, axis=1), axis=1),
                  label=M.MODEL_WEIGHTS + " . " + M.MOTOR_DIFF)
        figr.plot(t, b_mod[:, m, d], label=M.MODEL_BIAS)
        figr.plot(t, y_inp[:, m], 'k', label=M.SENSORY_INPUT)
        figr.plot(t, y_est[:, m, d], label=M.SENSORY_ESTIMATION)
        figr.plot(t, a_s[:, d], label=M.STATE_PHASE_ACTIVATOR)
        figr.plot(t, y_mem[:,m, d], label=M.SENSORY_MEMORY)

    if len(md_list) == 1:
        f.legend()
    else:
        f[-1].legend()


def motor_evolution(record, fig: Figure, nc_list=((0, 0),)):
    fig.suptitle("Motor evols")
    t = record["t"]
    u_cmd = record[M.MOTOR_COMMAND][:, 0, :]
    u_exp = record[M.MOTOR_EXPECTED][:, 0, :, 0, :]
    u_cmd_p = record[M.MOTOR_COMMAND_PERTURBATION][:, 0, :, 0, :]
    e_mot = record[M.MOTOR_ERROR][:, 0, :, 0, :]
    if _has_epicycle(record):
        u_mem = record[M.MOTOR_MEMORY][:, 0, :, 0, :, :]
    else:
        u_mem = record[M.MOTOR_MEMORY][:, 0, :, 0, :]

    a_m = record[M.MOTOR_PHASE_ACTIVATOR][:, 0, :]

    f = fig.subplots(len(nc_list), 1)
    for i, index in enumerate(nc_list):
        if len(nc_list) == 1:
            figr = f
        else:
            figr = f[i]
        n, c, = index
        figr.set_title("n:{},c:{}".format(n, c))
        figr.plot(t, u_cmd[:, n], 'k', label=M.MOTOR_COMMAND)
        figr.plot(t, u_exp[:, n, c], label=M.MOTOR_EXPECTED)
        figr.plot(t, u_cmd_p[:, n, c], label=M.MOTOR_COMMAND_PERTURBATION)
        figr.plot(t, e_mot[:, n, c], 'r', label=M.MOTOR_ERROR)
        if _has_epicycle(record):
            jmp = 0.7/u_mem.shape[-1]
            for k in range(u_mem.shape[-1]):
                if k == u_mem.shape[-1] - 1:
                    figr.plot(t, u_mem[:, n, c, k], 'g--', label=M.MOTOR_MEMORY, alpha=1)
                else:
                    figr.plot(t, u_mem[:, n, c, k], 'g--', alpha=k * jmp + 0.3)

        else:
            figr.plot(t, u_mem[:, n, c], 'g', label=M.MOTOR_MEMORY)
        figr.plot(t, -a_m[:, c], label=M.MOTOR_PHASE_ACTIVATOR)

    if len(nc_list) == 1:
        f.legend()
    else:
        f[-1].legend()


def epicycle_evolution(record, fig: Figure):
    fig.suptitle("Epicycle evolution")
    t = record["t"]
    state_phase = record[M.STATE_PHASE_ESTIMATION][:, 0]
    motor_phase = record[M.MOTOR_PHASE_CONTROLLER][:, 0]
    epicycle_phase = record[M.EPICYCLE_PHASE][:, 0]
    a_s = record[M.STATE_PHASE_ACTIVATOR][:, 0, :]
    a_m = record[M.MOTOR_PHASE_ACTIVATOR][:, 0, :]
    a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR][:, 0, :]

    f = fig.subplots(1, 2)
    fig_evol = f[0]
    fig_acti = f[1]

    fig_evol.plot(t, state_phase[:, 0], label=M.STATE_PHASE_ESTIMATION)
    fig_evol.plot(t, motor_phase[:, 0], label=M.MOTOR_PHASE_CONTROLLER)
    fig_evol.plot(t, epicycle_phase[:, 0], label=M.EPICYCLE_PHASE)
    fig_evol.legend()

    fig_acti.plot(t, a_s[:,0], 'k', label=M.STATE_PHASE_ACTIVATOR)
    fig_acti.plot(t, a_m[:,0], 'r', label=M.MOTOR_PHASE_ACTIVATOR)
    fig_acti.plot(t, a_ep)
    fig_acti.legend()


def y_mem_evolution(record, fig: Figure, sensor_m=0, size=20):
    fig.suptitle("Sensory memory evol - m:{}".format(sensor_m))
    y_mem = record[M.SENSORY_MEMORY]
    b = record[M.MODEL_BIAS]
    _T,_,_M,_, _D, _K = y_mem.shape
    jump = max(_T // (size * size), 1)

    f = fig.subplots(size, size)
    cou = 0
    for i in range(size):
        for j in range(size):
            cur_t = cou * jump
            if cur_t < _T:
                f[i][j].matshow(y_mem[cur_t, 0, sensor_m, 0, :, :])
                # f[i][j].matshow(y_mem[cur_t, 0, sensor_m, 0, :, :] - b[cur_t, 0, sensor_m, 0, :, None])
                f[i][j].set_yticks([])
                f[i][j].set_xticks([])
            if j == 0:
                f[i][0].set_ylabel(cur_t)
            if i == size-1:
                f[i][j].set_xlabel(j * jump)
            cou += 1


def u_mem_evolution(record, fig: Figure, motor_n=0, size=20):
    fig.suptitle("Motor memory evol - n:{}".format(motor_n))
    u_mem = record[M.MOTOR_MEMORY]
    u_exp = record[M.MOTOR_EXPECTED]

    _T,_,_N,_, _C, _K = u_mem.shape
    jump = max(_T // (size * size), 1)

    f = fig.subplots(size, size)
    cou = 0
    for i in range(size):
        for j in range(size):
            cur_t = cou * jump
            if cur_t < _T:
                # f[i][j].matshow((u_mem[cur_t, 0, motor_n, 0, :, :].T - u_exp[cur_t, 0, motor_n, 0, :]).T)
                # f[i][j].matshow((u_mem[cur_t, 0, motor_n, 0, :, :] - u_exp[cur_t, 0, motor_n, 0, :, None]))
                f[i][j].matshow(u_mem[cur_t, 0, motor_n, 0, :, :])
                f[i][j].set_yticks([])
                f[i][j].set_xticks([])
            if j == 0:
                f[i][0].set_ylabel(cur_t)
            if i == size-1:
                f[i][j].set_xlabel(j * jump)
            cou += 1


def bias_learning(record, fig: Figure, sensor_m=0, phase_d=0):
    fig.suptitle("Bias Learning n:{}, d:{}".format(sensor_m, phase_d))
    b_mod = record[M.MODEL_BIAS][:, 0, sensor_m, 0, phase_d]
    t = record["t"]
    a_s = record[M.STATE_PHASE_ACTIVATOR][:,0,:]
    a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR][:,0,:]
    y_mem = record[M.SENSORY_MEMORY][:, 0, sensor_m, 0, phase_d, :]
    y_inp = record[M.SENSORY_INPUT][:, 0, sensor_m]
    y_eff = record[M.SENSORY_EFFERENT_ESTIMATION][:, 0, sensor_m, 0, phase_d]
    e_mod = record[M.MODEL_ERROR][:, 0, sensor_m, 0, phase_d, :]

    f = fig.subplots(1, 1)
    f.plot(t, b_mod, 'b', label=M.MODEL_BIAS)
    f.plot(t, e_mod, 'r--', alpha=0.2)
    f.plot(t, np.average(e_mod, axis=1), 'r', label=M.MODEL_ERROR)
    f.plot(t, -a_s[:, phase_d]/2, label=M.STATE_PHASE_ACTIVATOR)
    f.plot(t, y_inp, 'k', alpha=0.7, label=M.SENSORY_INPUT)
    f.plot(t, y_eff, label=M.SENSORY_EFFERENT_ESTIMATION)

    jmp = 0.7/a_ep.shape[-1]
    for k in range(a_ep.shape[-1]):
        if k == a_ep.shape[-1]-1:
            f.plot(t, -a_ep[:, k], 'g--', alpha=1, label=M.EPICYCLE_PHASE_ACTIVATOR)
            f.plot(t, y_mem[:, k], 'g', alpha=1, label=M.SENSORY_MEMORY)
        else:
            f.plot(t, -a_ep[:, k], 'g--', alpha=0.3 + jmp * k)
            f.plot(t, y_mem[:, k], 'g', alpha=0.3 + jmp * k)
    f.legend()


def weight_matrix_evolution(record, fig: Figure, motor_n=0, sensor_m=0, size=20):
    fig.suptitle("Weight evol - n:{}".format(motor_n))
    w_mod = record[M.MODEL_WEIGHTS]
    _T,_M,_N,_C, _D, = w_mod.shape
    jump = max(_T // (size * size), 1)

    f = fig.subplots(size, size)
    cou = 0
    for i in range(size):
        for j in range(size):
            cur_t = cou * jump
            if cur_t < _T:
                f[i][j].matshow(w_mod[cur_t, sensor_m, motor_n, :, :])
                f[i][j].set_yticks([])
                f[i][j].set_xticks([])
            if j == 0:
                f[i][0].set_ylabel(cur_t)
            if i == size-1:
                f[i][j].set_xlabel(j * jump)
            cou += 1


def phase_combined_sensory_evolution(record, fig: Figure):
    fig.suptitle("Combined sensory evolution")
    t = record["t"]
    a_s = record[M.STATE_PHASE_ACTIVATOR]
    a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR]
    y_eff = record[M.SENSORY_EFFERENT_ESTIMATION]
    y_est = record[M.SENSORY_ESTIMATION]
    y_inp = record[M.SENSORY_INPUT]
    y_ref = record[M.SENSORY_REFERENCE]
    # combine
    y_eff_cmb = np.einsum("timjc,tic->tim", y_eff, a_s)
    y_ref_cmb = np.einsum("timjc,tic->tim", y_ref, a_s)
    y_est_cmb = np.einsum("timjck,tic,tik->tim", y_est, a_s, a_ep)

    sensor_m = y_inp.shape[-1]
    state_phase_D = a_s.shape[-1]
    fi = fig.subplots(sensor_m, 1)
    for m in range(sensor_m):
        if m > 1:
            f = fi[m]
        else:
            f = fi
        f.plot(t, y_inp[:, 0, m], 'k', label=M.SENSORY_INPUT)
        f.plot(t, y_eff_cmb[:, 0, m], 'b', label=M.SENSORY_EFFERENT_ESTIMATION)
        f.plot(t, y_ref_cmb[:, 0, m], 'r', label=M.SENSORY_REFERENCE)
        f.plot(t, y_est_cmb[:, 0, m], 'b', alpha=0.3, label=M.SENSORY_ESTIMATION)
        jmp = 0.7/state_phase_D
        for d in range(state_phase_D):
            if d == state_phase_D - 1:
                f.plot(t, -a_s[:, 0, d], 'g--', alpha=1, label=M.STATE_PHASE_ACTIVATOR)
            else:
                f.plot(t, -a_s[:, 0, d], 'g--', alpha=0.3 + jmp * d)

    if sensor_m > 1:
        fi[sensor_m - 1].legend()
    else:
        fi.legend()


def phase_combined_motor_evolution(record, fig: Figure):
    fig.suptitle("Combined motor evolution")
    t = record["t"]
    a_n = record[M.MOTOR_PHASE_ACTIVATOR]
    u_exp = record[M.MOTOR_EXPECTED]
    u_cmd = record[M.MOTOR_COMMAND]
    e_mot = record[M.MOTOR_ERROR]
    # combine
    u_exp_cmb = np.einsum("tinjd,tid->tin", u_exp, a_n)
    e_mot_cmb = np.einsum("tinjd,tid->tin", e_mot, a_n)

    motor_n = u_cmd.shape[-1]
    motor_phase_C = a_n.shape[-1]
    fi = fig.subplots(motor_n, 1)
    for n in range(motor_n):
        if n > 1:
            f = fi[n]
        else:
            f = fi
        f.plot(t, u_cmd[:, 0, n], 'k', alpha=0.3, label=M.MOTOR_COMMAND)
        f.plot(t, u_exp_cmb[:, 0, n], 'b', label=M.MOTOR_EXPECTED)
        f.plot(t, e_mot_cmb[:, 0, n], 'r', alpha=0.8, label=M.MOTOR_ERROR)
        jmp = 0.7/motor_phase_C
        for d in range(motor_phase_C):
            if d == motor_phase_C - 1:
                f.plot(t, -a_n[:, 0, d], 'g--', alpha=1, label=M.MOTOR_PHASE_ACTIVATOR)
            else:
                f.plot(t, -a_n[:, 0, d], 'g--', alpha=0.3 + jmp * d)

    if motor_n > 1:
        fi[motor_n - 1].legend()
    else:
        fi.legend()


def state_syncing(record, fig: Figure):

    fig.suptitle("State synchronisation")
    t = record["t"]
    state_phase = record[M.STATE_PHASE_ESTIMATION][:, 0]
    motor_phase = record[M.MOTOR_PHASE_CONTROLLER][:, 0]
    epicycle_phase = record[M.EPICYCLE_PHASE][:, 0]
    p_s = record[M.STATE_PERTURBATION][:, 0]
    p_m = record[M.MOTOR_PERTURBATION][:, 0]
    p_ep = record[M.EPICYCLE_PERTURBATION][:, 0]

    f = fig.subplots(3, 1)
    fig_evol_state = f[0]
    fig_evol_motor = f[1]
    fig_evol_epicycle = f[2]

    fig_evol_state.plot(t, np.sin(state_phase[:, 0]), label="sin " + M.STATE_PHASE_ESTIMATION)
    fig_evol_state.plot(t, p_s, label=M.STATE_PERTURBATION)

    fig_evol_motor.plot(t, np.sin(motor_phase[:, 0]), label="sin " + M.MOTOR_PHASE_CONTROLLER)
    fig_evol_motor.plot(t, p_m, label=M.MOTOR_PERTURBATION)

    fig_evol_epicycle.plot(t, np.sin(epicycle_phase[:, 0]), label="sin " + M.EPICYCLE_PHASE)
    fig_evol_epicycle.plot(t, p_ep, label=M.EPICYCLE_PERTURBATION)

    fig_evol_state.legend()
    fig_evol_motor.legend()
    fig_evol_epicycle.legend()


def phase_combined_unbiased_sensory_estimation(record, fig: Figure):
    fig.suptitle("Combined unbiased sensor estimation")
    t = record["t"]
    a_s = record[M.STATE_PHASE_ACTIVATOR]
    a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR]
    y_est = record[M.SENSORY_ESTIMATION]
    y_inp = record[M.SENSORY_INPUT]
    u_cmd = record[M.MOTOR_COMMAND]
    # combine
    y_est_cmb = np.einsum("timjck,tic,tik->tim", y_est, a_s, a_ep)

    sensor_m = y_inp.shape[-1]
    state_phase_D = a_s.shape[-1]
    fi = fig.subplots(sensor_m, 1)
    for m in range(sensor_m):
        if m > 1:
            f = fi[m]
        else:
            f = fi
        f.plot(t, u_cmd[:, 0, 0], 'r', alpha=0.5, label= M.MOTOR_COMMAND)
        f.plot(t, y_inp[:, 0, m], 'k', alpha=0.8, label=M.SENSORY_INPUT)
        f.plot(t, y_est_cmb[:, 0, m], 'b', alpha=0.8, label=M.SENSORY_ESTIMATION)
        jmp = 0.7/state_phase_D
        for d in range(state_phase_D):
            if d == state_phase_D - 1:
                f.plot(t, -a_s[:, 0, d], 'g--', alpha=1, label=M.STATE_PHASE_ACTIVATOR)
            else:
                f.plot(t, -a_s[:, 0, d], 'g--', alpha=0.3 + jmp * d)

    if sensor_m > 1:
        fi[sensor_m - 1].legend()
    else:
        fi.legend()


def period_comparison_overlap(record, fig: Figure, iter_length, compare_periods, pivot_period, title_on=False):
    if title_on:
        fig.suptitle("Estimation overlap ")

    t = record["t"]
    a_s = record[M.STATE_PHASE_ACTIVATOR]
    a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR]
    y_est = record[M.SENSORY_ESTIMATION]
    y_inp = record[M.SENSORY_INPUT]
    u_cmd = record[M.MOTOR_COMMAND]
    # combine
    y_est_cmb = np.einsum("timjck,tic,tik->tim", y_est, a_s, a_ep)

    switch = a_s[1:, 0, 0] - a_s[:len(a_s)-1, 0, 0]
    start_idxs = np.argwhere(switch > 0.9)

    pivot_start = start_idxs[pivot_period, 0]
    pivot_end = start_idxs[pivot_period, 0] + iter_length
    pivot_t = t[pivot_start:pivot_end]

    fi = fig.subplots(1, 1)
    m = 0
    figr = fi
    for i, period_id in enumerate(compare_periods):
        if period_id < len(start_idxs):
            start_idx = start_idxs[period_id, 0]
            end_idx = min(start_idx + iter_length, len(t)-1)
            figr.plot(pivot_t[:end_idx], y_inp[start_idx:end_idx, 0, m], 'k', alpha=0.2)
            figr.plot(pivot_t[:end_idx], y_est_cmb[start_idx:end_idx, 0, m], 'k', alpha=0.2)

    figr.plot(pivot_t, y_inp[pivot_start:pivot_end, 0, m], 'k', alpha=0.8, label="output")
    figr.plot(pivot_t, y_est_cmb[pivot_start:pivot_end, 0, m], 'b', alpha=0.8, label="estimation")
    figr.plot(pivot_t, u_cmd[pivot_start:pivot_end, 0, 0], 'r', alpha=0.8, label="perturbation")
    figr.set_ylabel("sensory output")
    figr.legend()


def compare_syncing(records, fig: Figure, params, title_on=False, x_lim=None):
    if title_on:
        fig.suptitle("Synchronization and Desynchronization comparison")

    t = records[0]["t"]
    y_inp = records[0][M.SENSORY_INPUT]
    u_cmd = records[0][M.MOTOR_COMMAND]

    fi = fig.subplots(1, 2)
    figr = fi[0]
    figr_err = fi[1]
    m = 0

    figr.plot(t, y_inp[:, 0, m], 'k', alpha=0.8, label="output")
    figr.plot(t, u_cmd[:, 0, 0], 'r', alpha=0.8, label="perturbation")

    for i, record in enumerate(records):
        param = params[i]
        # combine
        a_s = record[M.STATE_PHASE_ACTIVATOR]
        a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR]
        y_est = record[M.SENSORY_ESTIMATION]
        y_est_cmb = np.einsum("timjck,tic,tik->tim", y_est, a_s, a_ep)
        figr.plot(t, y_est_cmb[:, 0, m], color=param["color"], alpha=0.7,
                  label=param["label"])
        figr.set_ylabel("estimation")
        err = _convolved_square_error(record, hann_win=2000)
        figr_err.plot(t, err, color=param["color"], label=param["label"])
        figr_err.set_ylabel("model error")

    if x_lim is not None:
        figr.set_xlim(x_lim)
        figr_err.set_xlim(x_lim)
    figr.legend()
    figr_err.legend()


def estimation_error_comparison(records, fig: Figure, params, x_lim=None, title_on=False, signal_legend_on=True, hide_ground_truth=False):
    if title_on:
        fig.suptitle("Estimation error comparison")

    t = records[0]["t"]

    fi = fig.subplots(2, 1)
    figr = fi[0]
    figr_err = fi[1]
    m = 0


    for i, record in enumerate(records):
        param = params[i]
        # combine
        a_s = record[M.STATE_PHASE_ACTIVATOR]
        a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR]
        y_est = record[M.SENSORY_ESTIMATION]
        y_inp = record[M.SENSORY_INPUT]
        u_cmd = record[M.MOTOR_COMMAND]

        y_est_cmb = np.einsum("timjck,tic,tik->tim", y_est, a_s, a_ep)
        offset = i*4

        figr.plot(t, y_est_cmb[:, 0, m] + offset, color=param["color"], alpha=0.9,
                  label=param["label"])
        if not hide_ground_truth:
            figr.plot(t, y_inp[:, 0, m] + offset, color='k', alpha=0.4)
        if i==0:
            figr.plot(t, u_cmd[:, 0, 0] + offset, color='r', alpha=0.9, label="perturbation")
        else:
            figr.plot(t, u_cmd[:, 0, 0] + offset, color='r', alpha=0.9)
        if x_lim is not None:
            figr.set_xlim(x_lim)
        figr.set_yticks([])
        figr.set_ylabel("estimation")


        err = _convolved_square_error(record, hann_win=2000)
        figr_err.plot(t, err, color=param["color"], label=param["label"])
        figr_err.set_ylabel("model error")

    if signal_legend_on:
        figr.legend()
    figr_err.legend()



def period_comparison_overlap_with_motor(record, fig: Figure, iter_length, compare_periods, pivot_period, pivot_phase, title_on=False, legend_on=True):
    if title_on:
        fig.suptitle("Estimation overlap ")

    t = record["t"]
    a_s = record[M.STATE_PHASE_ACTIVATOR]
    a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR]
    y_est = record[M.SENSORY_ESTIMATION]
    y_inp = record[M.SENSORY_INPUT]
    y_ref = record[M.SENSORY_REFERENCE]
    u_cmd = record[M.MOTOR_COMMAND]
    # combine
    y_est_cmb = np.einsum("timjck,tic,tik->tim", y_est, a_s, a_ep)
    y_ref_cmb = np.einsum("timjc,tic->tim", y_ref, a_s)

    switch = -a_s[1:, 0, pivot_phase] + a_s[:len(a_s)-1, 0, pivot_phase]
    start_idxs = np.argwhere(switch > 0.9)

    pivot_start = start_idxs[pivot_period, 0] - iter_length//2
    pivot_end = start_idxs[pivot_period, 0] + iter_length//2
    pivot_t = t[pivot_start:pivot_end]

    fi = fig.subplots(1, 1)
    m = 0
    figr = fi
    for i, period_id in enumerate(compare_periods):
        if period_id < len(start_idxs):
            start_idx = max(start_idxs[period_id, 0] - iter_length//2, 0)
            end_idx = min(start_idxs[period_id, 0] + iter_length//2, len(t)-1)
            # figr.plot(pivot_t[:end_idx], y_inp[start_idx:end_idx, 0, m], 'k', alpha=0.5)
            # figr.plot(pivot_t[:end_idx], y_est_cmb[start_idx:end_idx, 0, m], 'b', alpha=0.5)

    figr.plot(pivot_t, y_inp[pivot_start:pivot_end, 0, m], 'k', alpha=0.7, label="output")
    figr.plot(pivot_t, y_est_cmb[pivot_start:pivot_end, 0, m], 'b', alpha=0.9, label="estimation")
    figr.plot(pivot_t, u_cmd[pivot_start:pivot_end, 0, 0], 'r', alpha=0.9, label="control")
    figr.plot(pivot_t, y_ref_cmb[pivot_start:pivot_end, 0, 0], 'g', alpha=0.9, label="reference")
    figr.set_ylabel("estimation,control")
    if legend_on:
        figr.legend()


def true_control_error(record, fig: Figure, pivot_phase, title_on=False, legend_on=True):
    if title_on:
        fig.suptitle("True control error ")

    t = record["t"]
    a_s = record[M.STATE_PHASE_ACTIVATOR]
    y_inp = record[M.SENSORY_INPUT]
    y_ref = record[M.SENSORY_REFERENCE]
    a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR]
    y_est = record[M.SENSORY_EFFERENT_ESTIMATION]

    y_ref_cmb = np.einsum("timjc,tic->tim", y_ref, a_s)

    err_tru = np.square(y_inp[:,0,0] - y_ref_cmb[:, 0,0])
    err_est = np.square(y_est[:,0,0,0,pivot_phase] - y_ref[:,0,0,0,pivot_phase])

    switch = -a_s[1:, 0, pivot_phase] + a_s[:len(a_s)-1, 0, pivot_phase]
    start_idxs = np.argwhere(switch > 0.9).reshape((-1, ))
    fi = fig.subplots(1, 1)
    figr = fi


    figr.plot(t[start_idxs], err_tru[start_idxs], 'ko',label="true control error ph. {}".format(pivot_phase), alpha=0.7)
    figr.plot(t, err_est, label="estimated control error ph. {}".format(pivot_phase), alpha=0.7)
    if legend_on:
        figr.legend()


def true_control_errors(records, fig: Figure, pivot_phase, params, title_on=False, legend_on=True):
    if title_on:
        fig.suptitle("True control error ")

    fi = fig.subplots(1, 1)
    figr = fi

    t = records[0]["t"]
    y_est = records[0][M.SENSORY_EFFERENT_ESTIMATION]
    y_ref = records[0][M.SENSORY_REFERENCE]
    err_est = np.square(y_est[:, 0, 0, 0, pivot_phase] - y_ref[:, 0, 0, 0, pivot_phase])
    figr.plot(t, err_est, 'k', label="virtual", alpha=0.7)
    for i, record in enumerate(records):
        t = record["t"]
        a_s = record[M.STATE_PHASE_ACTIVATOR]
        y_inp = record[M.SENSORY_INPUT]
        y_ref = record[M.SENSORY_REFERENCE]
        y_est = record[M.SENSORY_EFFERENT_ESTIMATION]

        y_ref_cmb = np.einsum("timjc,tic->tim", y_ref, a_s)

        err_tru = np.square(y_inp[:,0,0] - y_ref_cmb[:, 0,0])

        switch = -a_s[1:, 0, pivot_phase] + a_s[:len(a_s)-1, 0, pivot_phase]
        start_idxs = np.argwhere(switch > 0.9).reshape((-1, ))

        figr.plot(t[start_idxs], err_tru[start_idxs], '--o', color=params[i]["color"], label=params[i]["label"], alpha=0.7)
        figr.set_ylabel("control error")
    if legend_on:
        figr.legend()


def model_parameters(record, fig: Figure, start_from_period=-1, sensor_m=0, motor_n=0):
    period = _get_period(record[M.MOTOR_PHASE_ACTIVATOR][:, 0, 1])
    t = -1 + ((start_from_period + 1) * period)
    fig.suptitle("Weights from t={} of M={}, N={}".format(t, sensor_m, motor_n))
    W_mod = record[M.MODEL_WEIGHTS][t, :, :, :, :]
    b_mod = record[M.MODEL_BIAS][t, :, :, :, :]

    w = W_mod[motor_n, sensor_m, :, :]
    b = b_mod[0, :, 0, :]

    f = fig.subplots(2, 1)
    f[0].matshow(w)
    f[0].set_xlabel("state phase D")
    f[0].set_ylabel("motor phase C")
    f[1].matshow(b)


def model_parameters_W(record, fig: Figure, start_from_period=-1, sensor_m=0, motor_n=0):
    period = _get_period(record[M.MOTOR_PHASE_ACTIVATOR][:, 0, 1])
    t = -1 + ((start_from_period + 1) * period)
    W_mod = record[M.MODEL_WEIGHTS][t, :, :, :, :]

    w = W_mod[motor_n, sensor_m, :, :]

    f = fig.subplots(1, 1)
    f.matshow(w, cmap=plt.cm.get_cmap("seismic"))
    f.set_xlabel("state phase")
    f.set_ylabel("control phase")


def model_parameters_b(record, fig: Figure, start_from_period=-1):
    period = _get_period(record[M.MOTOR_PHASE_ACTIVATOR][:, 0, 1])
    t = -1 + ((start_from_period + 1) * period)
    b_mod = record[M.MODEL_BIAS][t, :, :, :, :]

    b = b_mod[0, :, 0, :]

    f = fig.subplots(1, 1)
    f.matshow(b, cmap=plt.cm.get_cmap("seismic"))
    f.set_yticks([])


def epicycle_syncing(record, fig: Figure):
    t = record["t"]
    ep_phase = record[M.EPICYCLE_PHASE][:, 0]
    state_phase = record[M.STATE_PHASE_ESTIMATION][:, 0]
    perturbation = record[M.EPICYCLE_PERTURBATION]
    _E = record[M.EPICYCLE_PHASE_ACTIVATOR].shape[2]
    # motor_phase = record[M.MOTOR_PHASE_CONTROLLER][:, 0]
    # epicycle_phase = record[M.EPICYCLE_PHASE][:, 0]

    omega = _E*(ep_phase[-1] - ep_phase[0])/(t[-1] - t[0])
    print("omega: {}".format(omega))
    fig.suptitle("Epicycle evolution, Omega:{}")

    act = np.sin(ep_phase[:, 0] * _E)
    act_flt = act > 0.9
    act_sel = np.where(act_flt == 1)


    f = fig.subplots(1, 1)
    fig_evol = f

    dif = np.max(perturbation) - np.min(perturbation)

    fig_evol.plot(t, np.mod(ep_phase[:, 0], 2 * np.pi/64), label=M.EPICYCLE_PHASE)
    fig_evol.plot(t, perturbation / dif, label=M.EPICYCLE_PERTURBATION)
    # fig_evol.plot(t, np.mod(state_phase[:, 0], 2 * np.pi) - 2*np.pi, label=M.STATE_PHASE_ESTIMATION)
    # fig_evol.plot(t, motor_phase[:, 0], label=M.MOTOR_PHASE_CONTROLLER)
    # fig_evol.plot(t, epicycle_phase[:, 0], label=M.EPICYCLE_PHASE)
    fig_evol.legend()

    # fig_acti.plot(t, a_s[:, 0], 'k', label=M.STATE_PHASE_ACTIVATOR)
    # fig_acti.plot(t, a_m[:, 0], 'r', label=M.MOTOR_PHASE_ACTIVATOR)
    # fig_acti.plot(t, a_ep)
    # fig_acti.legend()