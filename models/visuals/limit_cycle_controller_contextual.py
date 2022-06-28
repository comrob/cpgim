from matplotlib.pyplot import Figure
import models.limit_cycle_controller_contextual as M
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as colors

COLORS = [
    "orange", "lime", "royalblue", "crimson", "aquamarine", "gold", "teal", "fuchsia", "peru", "olive",
# "orange", "lime", "royalblue", "crimson", "aquamarine",
# "orange", "lime", "royalblue", "crimson", "aquamarine",
# "orange", "lime", "royalblue", "crimson", "aquamarine",
# "orange", "lime", "royalblue", "crimson", "aquamarine",
# "orange", "lime", "royalblue", "crimson", "aquamarine",
# "orange", "lime", "royalblue", "crimson", "aquamarine",
# "orange", "lime", "royalblue", "crimson", "aquamarine",
# "orange", "lime", "royalblue", "crimson", "aquamarine",
]


def get_context_management_durations(record, ctx=0):
    lrn_sel = record[M.CONTEXT_LEARNING_SELECTION][:, 0, ctx]
    ctr_sel = record[M.CONTEXT_CONTROL_SELECTION][:, 0, ctx]
    cmd_sel = record[M.CONTEXT_COMMANDING_SELECTION][:, 0, ctx]

    ret = []
    selections = [lrn_sel, ctr_sel, cmd_sel]

    for sel in selections:
        diff = sel[1:] - sel[:-1]
        starts = np.argwhere(diff == 1).tolist()
        ends = np.argwhere(diff == -1).tolist()
        if len(starts) == len(ends) + 1:
            ends.append([len(lrn_sel) - 1])
        ret.append(list(zip(starts,ends)))

    return tuple(ret)


def _get_nmcdx_shape(record):
    w = record[M.MODEL_WEIGHTS]
    return w[0, :, :, :, :, :].shape


def context_control_evol(record, fig: Figure):
    fig.suptitle("Context Control Evolution (log values)")
    t = record["t"]
    q_perf = record[M.REAL_PERFORMANCE_QUALITY]
    q = record[M.CONTEXT_MODEL_QUALITY][:, 0, :]
    q_mu = record[M.CONTEXT_MODEL_QUALITY_MEAN][:, 0, :]
    th_in = record[M.CONTEXT_QUALITY_INNER_THRESHOLD_DYN]
    th_out = record[M.CONTEXT_QUALITY_OUTER_THRESHOLD_DYN]

    a = record[M.CONTEXT_ACTIVABLE][:, 0, :]

    opt_fr = record[M.CONTEXT_IS_BEST_FREEZED][:, 0, :]
    opt_ho = record[M.CONTEXT_IS_BEST_HOT][:, 0, :]

    lrn_s = record[M.CONTEXT_LEARNING_SELECTION][:, 0, :]
    ctr_s = record[M.CONTEXT_CONTROL_SELECTION][:, 0, :]
    cmd_s = record[M.CONTEXT_COMMANDING_SELECTION][:, 0, :]

    lrn_state = record[M.CONTEXT_LEARNING_STATE][:, 0, :]

    f = fig.subplots(3, 1)

    fig_q_perf = f[0]
    fig_q_perf.set_title(M.REAL_PERFORMANCE_QUALITY)
    fig_q = f[1]
    fig_q.set_title(M.CONTEXT_MODEL_QUALITY)
    fig_a = f[2]
    fig_a.set_title("Control signals")

    fig_q_perf.plot(t, np.log10(q_perf), color='b')
    # fig_q_perf.hlines([M.PERFORMANCE_QUALITY_THRESHOLD], np.min(t), np.max(t), colors='k', label="q_real thr")
    # fig_q_perf.legend()

    fig_q.plot(t, np.log10(th_in), color='k', label=M.CONTEXT_QUALITY_INNER_THRESHOLD_DYN)
    fig_q.plot(t, np.log10(th_out), color='k', label=M.CONTEXT_QUALITY_OUTER_THRESHOLD_DYN)

    fig_a.hlines([-1.5 * i - 0.1 for i in range(7)], np.min(t), np.max(t), colors='k', alpha=0.5)
    fig_a.hlines([-1.5 * i + 1.1 for i in range(7)], np.min(t), np.max(t), linestyles='--', colors='k', alpha=0.5)

    for ctx in range(a.shape[-1]):
        fig_q.plot(t, np.log10(q[:, ctx]), linestyle='--', color=COLORS[ctx], alpha=0.5)
        fig_q.plot(t, np.log10(q_mu[:, ctx]), color=COLORS[ctx], label="log " + str(ctx))

        fig_a.plot(t, a[:, ctx] - 1.5 * 0, color=COLORS[ctx], alpha=0.8)
        fig_a.plot(t, opt_fr[:, ctx] - 1.5 * 1, color=COLORS[ctx], alpha=0.8)
        fig_a.plot(t, opt_ho[:, ctx] - 1.5 * 2, color=COLORS[ctx], alpha=0.8)
        fig_a.plot(t, lrn_s[:, ctx] - 1.5 * 3, color=COLORS[ctx], alpha=0.8)
        fig_a.plot(t, ctr_s[:, ctx] - 1.5 * 4, color=COLORS[ctx], alpha=0.8)
        fig_a.plot(t, cmd_s[:, ctx] - 1.5 * 5, color=COLORS[ctx], alpha=0.8)
        fig_a.plot(t, lrn_state[:, ctx] - 1.5 * 6, color=COLORS[ctx], alpha=0.8)

    fig_a.set_yticks([-1.5 * i for i in range(7)])
    fig_a.set_yticklabels(
        [M.CONTEXT_ACTIVABLE, M.CONTEXT_IS_BEST_FREEZED, M.CONTEXT_IS_BEST_HOT, M.CONTEXT_LEARNING_SELECTION,
         M.CONTEXT_CONTROL_SELECTION, M.CONTEXT_COMMANDING_SELECTION, M.CONTEXT_LEARNING_STATE]
    )

    fig_q.legend()


def performance_quality_error(record, fig: Figure, log_it=False):
    fig.suptitle("True Performance Quality Evolution - log={}".format(log_it))
    t = record["t"]
    q_perf = record[M.REAL_PERFORMANCE_QUALITY]
    e_mot_fe = record[M.MOTOR_ERROR_FREE_ENERGY][:, 0, :, 0, :]
    e_mot_flt = record[M.MOTOR_ERROR_FILTERED][:, 0, :, 0, :]

    f = fig.subplots(2, 1)
    _, _M, _D = e_mot_flt.shape

    fig_q = f[0]
    fig_q.set_title("Squared Performance error")
    fig_var = f[1]
    fig_var.set_title("Sensory wise squared performance errors")

    fig_q.plot(t, q_perf, label=M.REAL_PERFORMANCE_QUALITY)
    fig_q.legend()

    for i in range(_M):
        err_fe = np.average(np.square(e_mot_fe[:, i, :]), axis=1)
        err_flt = np.average(np.square(e_mot_flt[:, i, :]), axis=1)
        if log_it:
            err_fe = np.log10(err_fe)
            err_flt = np.log10(err_flt)
        fig_var.plot(t, err_fe, '--',label=M.MOTOR_ERROR_FREE_ENERGY + "[{}]^2".format(i), alpha=0.5, color=COLORS[i])
        fig_var.plot(t, err_flt, label=M.MOTOR_ERROR_FILTERED + "[{}]^2".format(i), alpha=0.5, color=COLORS[i])
    fig_var.legend()


def context_quality_evol(record, fig: Figure, log=False):
    fig.suptitle("Context Quality Evolution log:{}".format(log))
    t = record["t"]
    q = record[M.CONTEXT_MODEL_QUALITY][:, 0, :]
    q_mu = record[M.CONTEXT_MODEL_QUALITY_MEAN][:, 0, :]
    q_var = record[M.CONTEXT_MODEL_QUALITY_VARIANCE][:, 0, :]
    th_in = record[M.CONTEXT_QUALITY_INNER_THRESHOLD_DYN]


    f = fig.subplots(3, 1)

    fig_q = f[0]
    fig_q.set_title("Quality and stats")
    fig_var = f[1]
    fig_var.set_title("Quality variance")
    fig_deb = f[2]
    if log:
        l_th_in = np.log10(th_in)
        fig_q.plot(t, l_th_in, color='k', label="thr_inn")
    else:
        fig_q.plot(t, th_in, color='k', label="thr_inn")

    for ctx in range(q.shape[-1]):
        l_mu = q_mu[:, ctx]
        l_std_p = l_mu + np.sqrt(q_var[:, ctx])
        l_std_n = l_mu - np.sqrt(q_var[:, ctx])
        l_q = q[:, ctx]
        l_q_var = q_var[:, ctx]
        if log:
            l_mu = np.log10(l_mu)
            l_std_p = np.log10(l_std_p)
            l_q_var = np.log10(q_var[:, ctx])
            l_q = np.log10(q[:, ctx])

        fig_q.plot(t, l_q, color=COLORS[ctx], alpha=0.5)
        fig_q.plot(t, l_mu, color=COLORS[ctx], label=M.CONTEXT_MODEL_QUALITY_MEAN+"_" + str(ctx), alpha=0.8)
        fig_q.plot(t, l_std_p, '--',color=COLORS[ctx], alpha=0.8)
        if not log:
            fig_q.plot(t, l_std_n, '--',color=COLORS[ctx], alpha=0.8)

        fig_var.plot(t, l_q_var, color=COLORS[ctx])

    fig_q.legend()


def motor_context_convergence(record, fig: Figure, squared_norm=False):
    fig.suptitle("Motor Context")
    t = record["t"]
    u_ctx = record[M.MOTOR_CONTEXT][:, 0, :, 0, :, :]
    u_out = record[M.MOTOR_OUTPUT][:, 0, :, 0, :, :]
    u_mem = record[M.MOTOR_MEMORY][:, 0, :, 0, :, :]
    ctx_sel_cmd = record[M.CONTEXT_COMMANDING_SELECTION][:, 0, :]

    active_u_out = np.einsum("tncx,tx->tnc",u_out, ctx_sel_cmd)

    u_mem_mean = np.mean(u_mem, axis=3)
    f = fig.subplots(2, 2)

    if squared_norm:
        def norm(_x):
            return np.sum(np.square(_x), axis=(1, 2))
    else:
        def norm(_x):
            return np.sqrt(np.sum(np.square(_x), axis=(1, 2)))

    fig_conv = f[0][0]
    fig_conv.set_title("Sum {}".format(M.MOTOR_CONTEXT))
    u_ctx_sum = norm(u_ctx)
    u_out_sum = norm(u_out)
    for ctx in range(u_ctx.shape[-1]):
        fig_conv.plot(t, u_ctx_sum[:, ctx], color=COLORS[ctx], label="sum" + M.MOTOR_CONTEXT + str(ctx))
        fig_conv.plot(t, u_out_sum[:, ctx], '--', color=COLORS[ctx], label="sum" + M.MOTOR_OUTPUT + str(ctx))

    fig_conv.plot(t, norm(u_mem_mean), color='k', label=M.MOTOR_MEMORY)
    fig_conv.plot(t, norm(active_u_out), color='r', label="sel " + M.MOTOR_OUTPUT, alpha=0.5)
    fig_conv.legend()

    fig_projs = [
        f[0][1],
        f[1][0],
        f[1][1],
    ]
    center_n = u_ctx.shape[2]
    dims = [(0, center_n//3), (center_n//3,(2*center_n)//3), ((2*center_n)//3, center_n-1)]

    for i in range(len(dims)):
        figr = fig_projs[i]
        figr.set_title("Proj {}".format(dims[i]))
        for ctx in range(u_ctx.shape[-1]):
            figr.plot(u_ctx[:, 0, dims[i][0], ctx], u_ctx[:, 0, dims[i][1], ctx], color=COLORS[ctx], alpha=0.5)
            figr.plot(u_ctx[-1, 0, dims[i][0], ctx], u_ctx[-1, 0, dims[i][1], ctx], 'o', color=COLORS[ctx], alpha=1)


def model_context_parameter_convergence(record, fig: Figure, context=0):
    fig.suptitle("Model Parameter Convergence of ctx={}".format(context))
    t = record["t"]

    W_mod = record[M.MODEL_WEIGHTS][:, :, :, :, :, context]
    u_exp = record[M.MOTOR_EXPECTED][:, 0, :, 0, :, context]
    b_mod = record[M.MODEL_BIAS][:, 0, :, 0, :, context]
    # e_mot = record[M.MOTOR_ERROR][:, 0, :, 0, :, context]
    e_mot = record[M.MOTOR_ERROR][:, 0, :, 0, :]
    e_mod = record[M.MODEL_ERROR][:, 0, :, 0, :, :, context]
    e_mod_var = record[M.PROB_VAR_SENSORY_MOTOR][:, 0, :, 0, :, context]

    sigm_yu = record[M.PROB_VAR_SENSORY_MOTOR][:, 0, :, 0, :, context]


    f = fig.subplots(3, 2)
    _N, _M, _C, _D, _X = _get_nmcdx_shape(record)
    fig_W_mod = f[0][0]
    fig_b_mod = f[1][0]
    fig_e_mod = f[2][0]

    fig_u_exp = f[0][1]
    fig_sigm_yu =f[1][1]
    fig_e_mot = f[2][1]

    for n in range(_N):
        for m in range(_M):
            fig_W_mod.plot(t, np.sum(np.sum(np.abs(W_mod[:, n, m, :, :]), axis=1), axis=1),
                           label=M.MODEL_WEIGHTS + "[n:{},m:{}]".format(n, m))
    # fig_W_mod.legend()

    for m in range(_M):
        fig_b_mod.plot(t, np.sum(b_mod[:, m, :], axis=1), label=M.MODEL_BIAS + "[m:{}]".format(m))
    fig_b_mod.legend()

    for n in range(_N):
        fig_u_exp.plot(t, np.sum(u_exp[:, n, :], axis=1), label=M.MOTOR_EXPECTED + "[n:{}]".format(n))
    fig_u_exp.legend()

    std_e_mod = np.einsum("tmdk,tmd->tmdk", e_mod, 1/np.sqrt(sigm_yu))
    for m in range(_M):
        fig_e_mod.plot(t, np.average(np.sum(np.square(std_e_mod[:, m, :, :]), axis=1), axis=1), label=M.MODEL_ERROR + "[avg m:{}]/std".format(m))
    fig_e_mod.legend()

    for m in range(_M):
        fig_e_mot.plot(t, np.sum(e_mot[:, m, :], axis=1), label=M.MOTOR_ERROR + "[m:{}]".format(m))
    fig_e_mot.legend()

    for m in range(_M):
        fig_sigm_yu.plot(t, np.sum(sigm_yu[:, m, :], axis=1), label=M.PROB_VAR_SENSORY_MOTOR + "[m:{}]".format(m))
    fig_sigm_yu.legend()


def weight_matrix_analysis(record, fig: Figure, sensor_m=0, motor_n=0, ctx=0):
    t = -1
    fig.suptitle("Weights from t={} of M={}, N={}, ctx={}".format(t, sensor_m, motor_n, ctx))
    W_mod = record[M.MODEL_WEIGHTS][t, :, :, :, :, ctx]

    w = W_mod[motor_n, sensor_m, :, :]
    f = fig.subplots(2, 2)
    f[0][0].matshow(w)
    f[0][0].set_xlabel("state phase D")
    f[0][0].set_ylabel("motor phase C")
    f[1][0].matshow(np.sum(w, axis=0, keepdims=True))
    f[0][1].matshow(np.sum(w, axis=1, keepdims=True))


def multi_io_matricies(record, fig: Figure, ctx=0, sensor_modewise_norming=False):
    t = -1
    fig.suptitle("Weights from t={}, ctx={}".format(t, ctx))

    W_mod = record[M.MODEL_WEIGHTS][t, :, :, :, :, ctx]

    cmap = 'bwr'

    n_mot = W_mod.shape[0]
    m_sen = W_mod.shape[1]
    f = fig.subplots(n_mot, m_sen)
    for m in range(m_sen):
        if not sensor_modewise_norming:
            norm = colors.Normalize(vmin=W_mod.min(), vmax=W_mod.max())
        else:
            norm = colors.Normalize(vmin=W_mod[:, m, :, :].min(), vmax=W_mod[:, m, :, :].max())
        for n in range(n_mot):
            if n_mot == 1 and m_sen == 1:
                figr = f
            elif n_mot == 1:
                figr = f[m]
            elif m_sen == 1:
                figr = f[n]
            else:
                figr = f[n][m]

            ax = figr.matshow(W_mod[n, m, :, :], norm=norm, cmap=cmap)
            figr.set_xticks([])
            figr.set_yticks([])
            if n == (n_mot-1):
                figr.set_xlabel("m:{}".format(m))
            if m == 0:
                figr.set_ylabel("n:{}".format(n))
    fig.subplots_adjust(right=0.8)
    if not sensor_modewise_norming:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(ax, cax=cbar_ax)

    # fig.colorbar(im, ax=axes.ravel().tolist())


def bias_evolution(record, fig: Figure, sensor_m=0, ctx=0):
    fig.suptitle("Bias evolution of sensor {} context {}".format(sensor_m, ctx))
    b_mod = record[M.MODEL_BIAS][:, :, :, :, :, ctx]
    coln = 1000
    _T,_,_M,_,_D = b_mod.shape
    rows = _T//coln + 1
    evol = np.zeros((rows * _D, coln))
    for i in range(rows):
        seq = b_mod[coln*i:coln*(i + 1), 0, sensor_m, 0, :]
        evol[_D * i:_D * (i+1), :seq.shape[0]] = seq.T
    f = fig.subplots(1, 1)
    f.matshow(evol)
    f.set_yticks([i * _D for i in range(rows)])
    f.set_yticklabels([i*coln for i in range(rows)])


def weight_matrix_evolution(record, fig: Figure, motor_n=0, sensor_m=0, size=20, ctx=0):
    fig.suptitle("Weight evol - n:{}".format(motor_n))
    w_mod = record[M.MODEL_WEIGHTS][:, :, :, :, :, ctx]
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


def e_mod_evolution(record, fig: Figure, sensor_m=0, size=20, ctx=0):
    fig.suptitle("Model Error evol - m:{}".format(sensor_m))
    y_mem = record[M.MODEL_ERROR][:, :, :, :, :, :, ctx]
    _T,_,_M,_, _D, _K = y_mem.shape
    jump = max(_T // (size * size), 1)

    f = fig.subplots(size, size)
    cou = 0
    for i in range(size):
        for j in range(size):
            cur_t = cou * jump
            if cur_t < _T:
                f[i][j].matshow(y_mem[cur_t, 0, sensor_m, 0, :, :])
                f[i][j].set_yticks([])
                f[i][j].set_xticks([])
            if j == 0:
                f[i][0].set_ylabel(cur_t)
            if i == size-1:
                f[i][j].set_xlabel(j * jump)
            cou += 1


def weight_matrix_gradient_analysis(record, fig: Figure, sensor_m=0, motor_n=0, ctx=0):
    t = -1
    fig.suptitle("Weights gradient from t={} of M={}, N={}".format(t, sensor_m, motor_n))
    d_W_mod = record["d_" + M.MODEL_WEIGHTS][t, :, :, :, :, ctx]
    dd_W_mod = record["dd_" + M.MODEL_WEIGHTS][t, :, :, :, :, ctx]

    d_w = d_W_mod[motor_n, sensor_m, :, :]
    dd_w = dd_W_mod[motor_n, sensor_m, :, :]
    f = fig.subplots(2, 2)
    f[0][0].matshow(np.sign(d_w))
    f[0][1].matshow(np.sign(dd_w))
    f[1][0].matshow(np.sign(dd_w) - np.sign(d_w))
    f[1][1].matshow(d_w)


def bias_learning(record, fig: Figure, sensor_m=0, phase_d=0, ctx=0):
    fig.suptitle("Bias Learning n:{}, d:{}, ctx:{}".format(sensor_m, phase_d, ctx))
    b_mod = record[M.MODEL_BIAS][:, 0, sensor_m, 0, phase_d, ctx]
    t = record["t"]
    a_s = record[M.STATE_PHASE_ACTIVATOR][:,0,:]
    a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR][:,0,:]
    y_mem = record[M.SENSORY_MEMORY][:, 0, sensor_m, 0, phase_d, :]
    y_inp = record[M.SENSORY_INPUT][:, 0, sensor_m]
    y_eff = record[M.SENSORY_EFFERENT_ESTIMATION][:, 0, sensor_m, 0, phase_d, ctx]
    e_mod = record[M.MODEL_ERROR][:, 0, sensor_m, 0, phase_d, :, ctx]

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


def weight_evolution(record, fig: Figure, weight_nmcd_list=((0, 0, 0, 0),), ctx=0):
    fig.suptitle("Weight evols ctx:{}".format(ctx))
    t = record["t"]
    W_mod = record[M.MODEL_WEIGHTS][:,:,:,:,:,ctx]
    e_mod = record[M.MODEL_ERROR][:, 0, :, 0, :, :, ctx]
    u_dif = record[M.MOTOR_DIFF][:, 0, :, 0, :, :,ctx]

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
        jmp = 0.7/e_mod.shape[-1]
        for k in range(e_mod.shape[-1]):
            if k == e_mod.shape[-1] - 1:
                figr.plot(t, e_mod[:, m, d, k], 'r--', label=M.MODEL_ERROR + "[m,d]")
                figr.plot(t, u_dif[:, n, c, k], 'm--', label=M.MOTOR_DIFF + "[n,c]")

            else:
                figr.plot(t, e_mod[:, m, d, k], 'r--', alpha=k * jmp + 0.3)
                figr.plot(t, u_dif[:, n, c, k], 'm--', alpha=k * jmp + 0.3)

        figr.plot(t, a_s[:, d], label=M.STATE_PHASE_ACTIVATOR + "[d]")

    if len(weight_nmcd_list) == 1:
        f.legend()
    else:
        f[-1].legend()


def phase_combined_sensory_evolution(record, fig: Figure, ctx=0):
    fig.suptitle("Combined sensory evolution ctx:{}".format(ctx))
    t = record["t"]
    a_s = record[M.STATE_PHASE_ACTIVATOR]
    a_ep = record[M.EPICYCLE_PHASE_ACTIVATOR]
    y_eff = record[M.SENSORY_EFFERENT_ESTIMATION][:, :, :, :, :, ctx]
    y_est = record[M.SENSORY_ESTIMATION][:, :, :, :, :, :, ctx]
    y_inp = record[M.SENSORY_INPUT]
    y_ref = record[M.SENSORY_REFERENCE]
    b = record[M.MODEL_BIAS][:, :, :, :, :, ctx]
    var = record[M.PROB_VAR_SENSORY_MOTOR][:, :, :, :, :, ctx]

    # combine
    y_eff_cmb = np.einsum("timjc,tic->tim", y_eff, a_s)
    y_ref_cmb = np.einsum("timjc,tic->tim", y_ref, a_s)
    y_est_cmb = np.einsum("timjck,tic,tik->tim", y_est, a_s, a_ep)
    b_cmb = np.einsum("timjc,tic->tim", b, a_s)
    var_cmb = np.einsum("timjc,tic->tim", var, a_s)
    std_cmb = np.sqrt(var_cmb)

    sensor_m = y_inp.shape[-1]
    state_phase_D = a_s.shape[-1]
    fi = fig.subplots(sensor_m, 1)
    for m in range(sensor_m):
        if sensor_m > 1:
            f = fi[m]
        else:
            f = fi
        f.plot(t, y_inp[:, 0, m], 'k', label=M.SENSORY_INPUT)
        f.plot(t, y_eff_cmb[:, 0, m], 'b', label=M.SENSORY_EFFERENT_ESTIMATION)
        f.plot(t, y_ref_cmb[:, 0, m], 'r--', label=M.SENSORY_REFERENCE)
        f.plot(t, y_est_cmb[:, 0, m], 'b', alpha=0.3, label=M.SENSORY_ESTIMATION)
        f.plot(t, b_cmb[:, 0, m], 'g', alpha=0.2, label=M.MODEL_BIAS)
        f.plot(t, (y_est_cmb + std_cmb)[:, 0, m], 'b--', alpha=0.2)
        f.plot(t, (y_est_cmb - std_cmb)[:, 0, m], 'b--', alpha=0.2, label="sqrt " + M.PROB_VAR_SENSORY_MOTOR)
        jmp = 0.7/state_phase_D
        # for d in range(state_phase_D):
        #     if d == state_phase_D - 1:
        #         f.plot(t, -a_s[:, 0, d], 'g--', alpha=1, label=M.STATE_PHASE_ACTIVATOR)
        #     else:
        #         f.plot(t, -a_s[:, 0, d], 'g--', alpha=0.3 + jmp * d)

    if sensor_m > 1:
        fi[sensor_m - 1].legend()
    else:
        fi.legend()


def bias_learning_debug(record, fig: Figure, sensor_m=0, phase_d=0, ctx=0):
    fig.suptitle("Bias Learning n:{}, d:{}, ctx:{}".format(sensor_m, phase_d, ctx))
    b_mod = record[M.MODEL_BIAS][:, 0, sensor_m, 0, phase_d, ctx]
    d_b_mod = record["d_" + M.MODEL_BIAS][:, 0, sensor_m, 0, phase_d, ctx]
    dd_b_mod = record["dd_" + M.MODEL_BIAS][:, 0, sensor_m, 0, phase_d, ctx]
    rms_b_mod = record[M.RMS_MODEL_BIAS][:, 0, sensor_m, 0, phase_d, ctx]
    t = record["t"]
    y_inp = record[M.SENSORY_INPUT][:, 0, sensor_m]
    ctx_sel_lrn = record[M.CONTEXT_LEARNING_SELECTION][:, 0, ctx]
    ctx_sel_ctr = record[M.CONTEXT_CONTROL_SELECTION][:, 0, ctx]

    f = fig.subplots(1, 1)
    f.plot(t, b_mod, 'b', label=M.MODEL_BIAS, alpha=0.5)
    f.plot(t, b_mod, '.b')
    f.plot(t, d_b_mod, 'm', label="d_" + M.MODEL_BIAS, alpha=0.5)
    f.plot(t, d_b_mod, '.m')
    f.plot(t, dd_b_mod, 'c', label="dd_" + M.MODEL_BIAS, alpha=0.5)
    f.plot(t, dd_b_mod, '.c')
    f.plot(t, rms_b_mod, 'y', label=M.RMS_MODEL_BIAS, alpha=0.5)
    f.plot(t, rms_b_mod, '.y')


    f.plot(t, ctx_sel_lrn, 'r', label=M.CONTEXT_LEARNING_SELECTION, alpha=0.5)
    f.plot(t, ctx_sel_ctr, 'g', label=M.CONTEXT_CONTROL_SELECTION, alpha=0.5)
    f.plot(t, y_inp, 'k', alpha=0.7, label=M.SENSORY_INPUT)
    f.legend()


def command_evolution(record, fig: Figure, coln=200):
    u_cmd = record[M.MOTOR_COMMAND][:, 0, :]
    _T, _D = u_cmd.shape
    rows = _T//coln + 1
    evol = np.zeros((rows * _D, coln))
    for i in range(rows):
        seq = u_cmd[coln*i:coln*(i + 1), :]
        evol[_D * i:_D * (i+1), :seq.shape[0]] = seq.T
    f = fig.subplots(1, 1)
    f.matshow(evol)
    f.set_yticks([i * _D for i in range(rows)])
    f.set_yticklabels([i*coln for i in range(rows)])


def multi_control_outputs(record, fig: Figure):
    fig.suptitle("u_out")

    u_out = record[M.MOTOR_OUTPUT][-1, 0, :, 0, :, :]
    norm = colors.Normalize(vmin=u_out.min(), vmax=u_out.max())
    cmap = 'bwr'

    n_ctx = u_out.shape[-1]
    f = fig.subplots(n_ctx, 1)
    for n in range(n_ctx):
        if n_ctx == 1:
            figr = f
        else:
            figr = f[n]

        figr.matshow(u_out[:, :, n], norm=norm, cmap=cmap)
        figr.set_xticks([])
        figr.set_yticks([])
        figr.set_xlabel("ctx:{}".format(n))


def motor_evolution(record, fig: Figure, nc_list=((0, 0),), ctx=0):
    fig.suptitle("Motor evols")
    t = record["t"]
    u_cmd = record[M.MOTOR_COMMAND][:, 0, :]
    u_exp = record[M.MOTOR_EXPECTED][:, 0, :, 0, :, ctx]
    u_cmd_p = record[M.MOTOR_COMMAND_PERTURBATION][:, 0, :, 0, :]
    e_mot = record[M.MOTOR_ERROR][:, 0, :, 0, :, ctx]
    u_mem = record[M.MOTOR_MEMORY][:, 0, :, 0, :, :]

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
        jmp = 0.7/u_mem.shape[-1]
        for k in range(u_mem.shape[-1]):
            if k == u_mem.shape[-1] - 1:
                figr.plot(t, u_mem[:, n, c, k], 'g--', label=M.MOTOR_MEMORY, alpha=1)
            else:
                figr.plot(t, u_mem[:, n, c, k], 'g--', alpha=k * jmp + 0.3)

        figr.plot(t, -a_m[:, c], label=M.MOTOR_PHASE_ACTIVATOR)

    if len(nc_list) == 1:
        f.legend()
    else:
        f[-1].legend()

def perturbation_histogram(record, fig: Figure):
    fig.suptitle("Motor Perturbation Histograms")
    t = record["t"]
    u_prt = record[M.MOTOR_COMMAND_PERTURBATION][:, 0, :, 0, :]

    T, N, C = u_prt.shape
    f = fig.subplots(C, N)
    u_prt_max = np.max(u_prt)
    for n in range(N):
        for c in range(C):
            if N == 1 and C == 1:
                figr = f
            elif N == 1:
                figr = f[c]
            elif C == 1:
                figr = f[n]
            else:
                figr = f[c][n]
            pert = u_prt[u_prt[:, n, c] > 0, n, c]
            figr.hist(pert)
            figr.set_xlim((0, u_prt_max))


def perturbation_amplitude(record, fig: Figure):
    fig.suptitle("Motor Perturbation Amplitude")
    t = record["t"]
    u_prt = record[M.MOTOR_COMMAND_PERTURBATION][:, 0, :, 0, :]

    u_prt_amp = np.sqrt(np.sum(np.square(u_prt), axis=2))
    T, N, C = u_prt.shape
    f = fig.subplots(N, 2)
    for n in range(N):
            if N == 1:
                figr = f
            else:
                figr = f[n]
            figr[0].plot(t, u_prt_amp[:, n])
            figr[1].hist(u_prt_amp[:,n])


def u_dif_evolution(record, fig: Figure, motor_n=0, size=4, ctx=0):
    fig.suptitle("Motor dif evol - n:{}, ctx:{}".format(motor_n,ctx))
    u_mem = record[M.MOTOR_DIFF][:, :, :, :, :, :, ctx]

    _T,_,_N,_, _C, _K = u_mem.shape
    jump = max(_T // (size * size), 1)

    f = fig.subplots(size, size)
    cou = 0
    for i in range(size):
        for j in range(size):
            cur_t = cou * jump
            if cur_t < _T:
                f[i][j].matshow(u_mem[cur_t, 0, motor_n, 0, :, :])
                f[i][j].set_yticks([])
                f[i][j].set_xticks([])
            if j == 0:
                f[i][0].set_ylabel(cur_t)
            if i == size-1:
                f[i][j].set_xlabel(j * jump)
            cou += 1


def sensory_posterior_variance(record, fig: Figure, sensor_m=0, ctx=0):
    fig.suptitle("Variance evolution of sensor {} context {}".format(sensor_m, ctx))
    var_sens_post = record[M.PROB_VAR_SENSORY_MOTOR][:, :, :, :, :, ctx]
    coln = 1000
    _T,_,_M,_,_D = var_sens_post.shape
    rows = _T//coln + 1
    evol = np.zeros((rows * _D, coln))
    for i in range(rows):
        seq = var_sens_post[coln*i:coln*(i + 1), 0, sensor_m, 0, :]
        evol[_D * i:_D * (i+1), :seq.shape[0]] = seq.T
    f = fig.subplots(1, 1)
    f.matshow(evol)
    f.set_yticks([i * _D for i in range(rows)])
    f.set_yticklabels([i*coln for i in range(rows)])


def joint_motorsensory_distribution(record, fig: Figure):
    fig.suptitle("Joint p(u,y) distribution")
    t = record["t"]
    pdf = record[M.PROB_JOINT_SENSORYMOTOR][:, 0, :, :]
    T, _, _X = pdf.shape
    f = fig.subplots(_X, 1)

    for i in range(_X):
        if _X == 1:
            figr = f
        else:
            figr = f[i]
        figr.plot(t, pdf[:, 1, i], label="p(u)")
        figr.plot(t, pdf[:, 2, i], label="p(y|u)")
        figr.plot(t, pdf[:, 0, i], label="p(y,u)")
        if i == _X-1:
            figr.legend()


def debug_system_performance_error(record, fig: Figure, only_pos_error=True):
    fig.suptitle("System performance (y_ref - y_mem)^2")
    t = record["t"]
    y_mem = record[M.SENSORY_MEMORY][:, 0, :, :, :, :]
    y_ref = record[M.SENSORY_REFERENCE][:, 0, :, :, :]
    y_ref_mask = record[M.SENSORY_REFERENCE_MASK][:, 0, :, :, :]

    y_mem_mean = np.mean(y_mem, axis=4)
    if not only_pos_error:
        y_dif = y_mem_mean - y_ref
    else:
        y_dif = np.maximum(y_ref - y_mem_mean, 0)

    err = np.mean(np.square(y_dif) * y_ref_mask, axis=(1,3))

    f = fig.subplots(1, 1)

    f.plot(t, np.log10(err))


def param_stat_detail(record, fig: Figure, tnmcdx=((0, 0, 0, 0 ,0, 0),)):
    fig.suptitle("Stat")
    tim = record["t"]
    W = record[M.MODEL_WEIGHTS]
    b = record[M.MODEL_BIAS]
    sigm_yu = record[M.PROB_VAR_SENSORY_MOTOR]
    y_mem = record[M.SENSORY_MEMORY]
    u_mem = record[M.MOTOR_MEMORY]
    u_ctx = record[M.MOTOR_CONTEXT]

    _T, _N, _M, _C, _D, _X = W.shape
    plt_n = len(tnmcdx)

    f = fig.subplots(plt_n, 1)

    for i in range(plt_n):
        if plt_n == 1:
            figr = f
        else:
            figr = f[i]
        t, n, m, c, d, x = tnmcdx[i]
        figr.set_title("t:{}({}), n:{}, m:{}, c:{}, d:{}, x:{}".format(t,tim[t], n, m, c, d, x))

        _w = W[t, n, m, c, d, x]
        _b = b[t, 0, m, 0, d, x]
        _sigm_yu = sigm_yu[t, 0, m, 0, d, x]
        _u_ctx = u_ctx[t, 0, n, 0, c, x]
        _y_mem = y_mem[t, 0, m, 0, d, :]
        _u_mem = u_mem[t, 0, n, 0, c, :]

        _u_min = np.min(_u_mem)
        _u_max = np.max(_u_mem)

        us = np.linspace(start=_u_min, stop=_u_max)
        _ws = (us - _u_ctx) * _w + _b

        figr.plot(_u_mem, _y_mem, ".", label="uy_mem")
        figr.plot([_u_ctx], [_b], ".", label="[u_ctx,b]")
        figr.plot(us, _ws, label="w")
        figr.plot(us, _ws + np.sqrt(_sigm_yu), 'k--')
        figr.plot(us, _ws - np.sqrt(_sigm_yu), 'k--')


        if i == 0:
            figr.legend()


def posterior_sensory_variance_detail(record, fig: Figure, t=0, sensor_m=0, ctx=0, title_psfx=""):
    fig.suptitle("Posterior sensory variance detail{}; t:{}, m:{}, ctx:{}".format(title_psfx, t, sensor_m, ctx))
    tim = record["t"]
    sigm_yu = record[M.PROB_VAR_SENSORY_MOTOR][t, 0, sensor_m, 0, :, ctx]
    y_est = record[M.SENSORY_ESTIMATION][t, 0, sensor_m, 0, :, :, ctx]
    y_mem = record[M.SENSORY_MEMORY][t, 0, sensor_m, 0, :, :]
    ep_act = record[M.EPICYCLE_PHASE_ACTIVATOR][t, :, :]
    ep_id = np.argmax(ep_act)

    _D, _K = y_est.shape

    f = fig.subplots(_D, 1)
    epc = [i for i in range(_K)]
    for i in range(_D):
        if _D == 1:
            figr = f
        else:
            figr = f[i]

        figr.plot(epc, y_est[i, :], 'r.', label="y_est")
        figr.plot(epc, y_mem[i, :], 'k.', label="y_mem")
        figr.plot(epc, y_est[i, :] + np.sqrt(sigm_yu[i]), 'rx', alpha=0.2)
        figr.plot(epc, y_est[i, :] - np.sqrt(sigm_yu[i]), 'rx', alpha=0.2, label="sigm_yu")
        figr.axvline(x=ep_id, color='g', label=M.EPICYCLE_PHASE_ACTIVATOR, alpha=0.2)


        if i == 0:
            figr.legend()


def free_energy_motor_error_comparison(record, fig: Figure, sensor_m=0):
    fig.suptitle("Free energy motor error comparison, sensor_m:{}".format(sensor_m))
    tim = record["t"]
    e_mot_fe = record[M.MOTOR_ERROR_FREE_ENERGY][:, 0, sensor_m, 0, :]
    e_mot = record[M.MOTOR_ERROR][:, 0, sensor_m, 0, :]

    _T, _D = e_mot_fe.shape

    f = fig.subplots(_D, 1)
    for i in range(_D):
        if _D == 1:
            figr = f
        else:
            figr = f[i]

        figr.plot(tim, e_mot[:, i], 'k', label=M.MOTOR_ERROR)
        figr.plot(tim, e_mot_fe[:, i], 'b', label=M.MOTOR_ERROR_FREE_ENERGY)

        if i == 0:
            figr.legend()


def compare_performance_error_means(records, fig: Figure, record_labels, mean_interval=(0, -1), ctx=0, conv_mean_window=1000):
    start, end = mean_interval

    perf_err = [record[M.REAL_PERFORMANCE_QUALITY] for record in records]
    # estm_err = [record[M.CONTEXT_MODEL_QUALITY][:, 0, ctx] for record in records]
    estm_err = [record[M.CONTEXT_MODEL_QUALITY][:, 0, ctx] for record in records]

    f = fig.subplots(2, 2)
    figr_aggregate_perf = f[1][0]
    figr_aggregate_estm = f[1][1]

    figr_evols_perf = f[0][0]
    figr_evols_estm = f[0][1]

    figr_aggregate_perf.plot(record_labels, [np.mean(sig[start:end]) for sig in perf_err], 'o-')
    figr_aggregate_estm.plot(record_labels, [np.mean(sig[start:end]) for sig in estm_err], 'o-')

    for i, record in enumerate(records):
        _t = record["t"]
        err_estm_mean = np.convolve(estm_err[i], np.ones(conv_mean_window) / conv_mean_window, mode='valid')
        err_perf_mean = np.convolve(perf_err[i], np.ones(conv_mean_window) / conv_mean_window, mode='valid')

        figr_evols_perf.plot(_t[conv_mean_window-1:], np.log10(err_perf_mean), label="{}".format(record_labels[i]), color=COLORS[i])
        figr_evols_estm.plot(_t[conv_mean_window-1:], np.log10(err_estm_mean), label="{}".format(record_labels[i]), color=COLORS[i])
        figr_aggregate_perf.plot([record_labels[i]], [np.mean(perf_err[i][start:end])], 'x', color=COLORS[i])
        figr_aggregate_estm.plot([record_labels[i]], [np.mean(estm_err[i][start:end])], 'x', color=COLORS[i])

    figr_evols_perf.legend()
    figr_evols_perf.set_title("Performance error")
    figr_evols_estm.legend()
    figr_evols_estm.set_title("Estimation error")


def compare_sensory_means(records, fig: Figure, record_labels, mean_interval=(0, -1), conv_mean_window=1000):
    start, end = mean_interval

    sens_inp = [record[M.SENSORY_INPUT][:, 0, :] for record in records]
    sens_ref = [record[M.SENSORY_REFERENCE][:, 0, :, 0, 0] for record in records]

    _T, _N = sens_inp[0].shape
    f = fig.subplots(2, _N)
    _t = records[0]["t"]

    for i in range(_N):
        fig_evol = f[0][i]
        fig_aggr = f[1][i]

        fig_aggr.plot(record_labels, [np.mean(sig[start:end, i]) for sig in sens_inp], 'o-')

        fig_evol.plot(_t, sens_ref[0][:, i], 'g')

        for j, record in enumerate(records):
            y_inp_mean = np.convolve(sens_inp[j][:, i], np.ones(conv_mean_window) / conv_mean_window, mode='valid')

            fig_evol.plot(_t[conv_mean_window-1:], y_inp_mean, label="{}".format(record_labels[j]), color=COLORS[j],
                          alpha=0.7)
            fig_aggr.plot(record_labels[j], [np.mean(sens_inp[j][start:end, i])], 'x', color=COLORS[j])

    f[0][0].legend()


def comparison_epicycle_syncing(records, fig: Figure, frequencies, labels, base_label):
    t = records[0]["t"]
    _E = records[0][M.EPICYCLE_PHASE_ACTIVATOR].shape[2]
    base_id = labels.index(base_label)

    ep_phases = [record[M.EPICYCLE_PHASE][:, 0] for record in records]
    state_phases = [record[M.STATE_PHASE_ESTIMATION][:, 0] for record in records]
    perturbations = [record[M.EPICYCLE_PERTURBATION] for record in records]

    pert_sels = [np.where(perturbation > 0.9) for perturbation in perturbations]

    obs_frequencies = [(phs[-1, 0] - phs[0, 0])/(t[-1] - t[0]) for phs in ep_phases]
    f = fig.subplots(4, 1)
    figr_phs = f[0]
    figr_phs_forc = f[1]
    figr_evol = f[2]
    figr_obss = f[3]

    figr_obss.plot(frequencies, obs_frequencies)

    for i, lab in enumerate(labels):
        figr_phs.plot(t, ep_phases[i] - ep_phases[base_id], label="{}".format(lab), color=COLORS[i])
        figr_phs.set_title("observed - base observed")
        pertph = np.asarray([ep_phases[i][0, 0] + _t * frequencies[i]/64 for _t in (t-t[0])])
        figr_phs_forc.plot(t, ep_phases[i][:,0] - pertph, label="{}".format(lab), color=COLORS[i])
        figr_phs_forc.set_title("observed - forced")
        x = np.mod(ep_phases[i][:, 0], 2 * np.pi/64)
        dif_x = np.max(x) - np.min(x)
        figr_evol.plot(t, x + dif_x * i, alpha=0.3, label="{}".format(lab), color=COLORS[i])
        figr_evol.plot(t[pert_sels[i]], x[pert_sels[i]] + dif_x * i, '.', color=COLORS[i])
        figr_obss.plot([frequencies[i]], [obs_frequencies[i]], 'o', color=COLORS[i])

    figr_phs.legend()
    figr_evol.legend()


