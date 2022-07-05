from matplotlib.pyplot import Figure
import models.limit_cycle_controller_contextual as CTR
import models.robot_goal_differential_control as REFPROV
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.cm

COLORS_CTX = [
    "orange", "fuchsia", "royalblue", "crimson", "aquamarine", "gold", "teal", "fuchsia", "peru", "olive",
]

COLORS_PHS = [
    "indianred", "royalblue", "fuchsia", "royalblue", "crimson", "aquamarine", "gold", "teal", "fuchsia", "peru", "olive",
]

COLORS_EXTERNAL = [
    "indianred", "royalblue", "fuchsia", "royalblue", "crimson", "aquamarine", "gold", "teal", "fuchsia", "peru", "olive",
]


MARKERS = [
    'o', '^', 's', 'x'
]

LABELS = {
    CTR.SENSORY_INPUT: "$y$",
    CTR.SENSORY_REFERENCE: "$y^*$",
}

SENSORY_NAMES = {
    0: "Velocity",
    1: "Roll",
    2: "Pitch",
    3: "Yaw"
}

TIME_LABEL = "time (arbitrary unit)"
ESTIMATION_ERROR_LABEL = "Estimation Error"
PERFORMANCE_ERROR_LABEL = "Performance Error"
RHYTHM_FREQUENCY_LABEL = "Rhythm frequency ($\omega_p$)"
OBSERVED_FREQUENCY_LABEL = "Observed frequency ($\Omega$)"


def get_context_management_durations(record, ctx=0):
    lrn_sel = record[CTR.CONTEXT_LEARNING_SELECTION][:, 0, ctx]
    ctr_sel = record[CTR.CONTEXT_CONTROL_SELECTION][:, 0, ctx]
    cmd_sel = record[CTR.CONTEXT_COMMANDING_SELECTION][:, 0, ctx]

    ret = []
    selections = [lrn_sel, ctr_sel, cmd_sel]

    for sel in selections:
        diff = sel[1:] - sel[:-1]
        start = np.argwhere(diff == 1)
        end = np.argwhere(diff == -1)

        if len(start) == 0:
            _start = len(lrn_sel) - 1
        else:
            _start = start[0,0]

        if len(end) == 0:
            _end = len(lrn_sel) - 1
        else:
            _end = end[0,0]
        ret.append((_start, _end))

    return tuple(ret)


def weight_matrix_pretty(record, fig: Figure, sensor_m=0, motor_n=0, ctx=0, title=None, mean=False, norm=None, cmap='bwr'):
    t = -1
    if not title is None:
        fig.suptitle(title)
    W_mod = record[CTR.MODEL_WEIGHTS][t, :, :, :, :, ctx]

    w = W_mod[motor_n, sensor_m, :, :]
    if norm is None:
        norm = colors.Normalize(vmin=w.min(), vmax=w.max())
    cmap = cmap
    f = fig.subplots(1, 1)
    if mean:
        ax = f.matshow(np.mean(w, keepdims=True), norm=norm, cmap=cmap)
    else:
        ax = f.matshow(w, norm=norm, cmap=cmap)

    f.set_xticks([])
    f.set_yticks([])

    f.set_xlabel("sensory phase")
    f.set_ylabel("motor phase")
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(ax, cax=cbar_ax)
    return norm


def multi_io_matricies(record, fig: Figure, ctx=0, sensor_modewise_norming=False, cmap='bwr', override_title=None):
    t = -1
    if override_title is None:
        override_title = "Weights from t={}, ctx={}".format(t, ctx)
    fig.suptitle(override_title)

    W_mod = record[CTR.MODEL_WEIGHTS][t, :, :, :, :, ctx]

    reorganized = [0,1,8,9,4,5,2,3,10,11,6,7]

    labels = [
        "L1c",#0
        "R1c",# 1
        "L1f",#  2
        "R1f",#   3
        "L3c",#    4
        "R3c",#     5
        "L3f",#      6
        "R3f",#       7
        "L2c",#        8
        "R2c",#         9
        "L2f",#          10
        "R2f",#            11
    ]
    W_mod = W_mod[reorganized, :, :, :]

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
                figr.set_xlabel("{}".format(SENSORY_NAMES[m]))
            if m == 0:
                figr.set_ylabel("{}".format(labels[reorganized[n]]))
    fig.subplots_adjust(right=0.8)
    if not sensor_modewise_norming:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(ax, cax=cbar_ax)


def control_pretty(record, fig: Figure, ctx=0, title=None, t=-1, norm=None, show_colorbar=True, ylabel=None):
    if not title is None:
        fig.suptitle(title)
    u_ctx = record[CTR.MOTOR_OUTPUT][t, 0, :, 0, :, ctx]
    # reorganized = [0,2,8,10,4,6,1,3,9,11,5,7]
    reorganized = [0,1,8,9,4,5,2,3,10,11,6,7]

    labels = [
        "L1c",#0
        "R1c",# 1
        "L1f",#  2
        "R1f",#   3
        "L3c",#    4
        "R3c",#     5
        "L3f",#      6
        "R3f",#       7
        "L2c",#        8
        "R2c",#         9
        "L2f",#          10
        "R2f",#            11
    ]
    w = u_ctx[reorganized, :]


    if norm is None:
        norm = colors.Normalize(vmin=w.min(), vmax=w.max())
    cmap = 'bwr'
    f = fig.subplots(1, 1)
    ax = f.matshow(w, norm=norm, cmap=cmap)
    if ylabel is not None:
        f.set_ylabel(ylabel)
    f.set_xticks([])
    f.set_yticks([i for i in range(12)])

    f.set_yticklabels(np.asarray(labels)[reorganized])

    f.set_xlabel("motor phase")
    if show_colorbar:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(ax, cax=cbar_ax)
    return norm


def control_stat_pretty(record, fig: Figure, stat=lambda x: np.mean(x, axis=0), ctx=0, title=None, interval=(0,-1), norm=None, show_colorbar=True, ylabel=None):
    if not title is None:
        fig.suptitle(title)
    u_ctx = record[CTR.MOTOR_OUTPUT][:, 0, :, 0, :, ctx]
    # reorganized = [0,2,8,10,4,6,1,3,9,11,5,7]
    reorganized = [0,1,8,9,4,5,2,3,10,11,6,7]

    labels = [
        "L1c",#0
        "R1c",# 1
        "L1f",#  2
        "R1f",#   3
        "L3c",#    4
        "R3c",#     5
        "L3f",#      6
        "R3f",#       7
        "L2c",#        8
        "R2c",#         9
        "L2f",#          10
        "R2f",#            11
    ]
    w = u_ctx[:, reorganized, :]
    w = stat(w[interval[0]:interval[1], :, :])


    if norm is None:
        norm = colors.Normalize(vmin=w.min(), vmax=w.max())
    cmap = 'bwr'
    f = fig.subplots(1, 1)
    ax = f.matshow(w, norm=norm, cmap=cmap)
    if ylabel is not None:
        f.set_ylabel(ylabel)
    f.set_xticks([])
    f.set_yticks([i for i in range(12)])

    f.set_yticklabels(np.asarray(labels)[reorganized])

    f.set_xlabel("motor phase")
    if show_colorbar:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(ax, cax=cbar_ax)
    return norm


def waypoint_navigation(record, fig: Figure, contexts=(0,1,2), title="Navigation", damage_iter=None,
                        start_goal=((0,0), (50,50))):
    if title is not None:
        fig.suptitle(title)
    t = record["t"]
    position_curr = record[REFPROV.CURRENT_POSITION]

    mng = []
    for i in range(record[CTR.MODEL_WEIGHTS].shape[-1]):
        mng.append(get_context_management_durations(record, i))

    f = fig.subplots(1, 1)

    fig_map = f

    fig_map.plot(position_curr[:, 0], position_curr[:, 1], 'k', alpha=0.5)

    for ctx in contexts:
        start = mng[ctx][1][0]
        end = mng[ctx][1][1]
        # fig_map.plot(position_curr[start:end, 0], position_curr[start:end, 1], '-', color=COLORS[ctx])
        fig_map.plot(position_curr[start:end, 0], position_curr[start:end, 1],
                     marker=MARKERS[ctx], color=COLORS_CTX[ctx], linestyle='', markersize=5, alpha=0.8, markevery=0.05)
        # fig_map.plot(position_curr[end, 0], position_curr[end, 1], '-', color=COLORS[ctx])

    fig_map.set_xlabel("X coordinate")
    fig_map.set_ylabel("Y coordinate")

    # INSIGHT
    mid_point = [0.15, -0.1]
    square = 0.7
    axins = fig_map.inset_axes([0.6, 0.08, 0.47, 0.47])
    axins.plot(position_curr[:, 0], position_curr[:, 1], 'k', alpha=0.5)
    axins.set_xlim(mid_point[0] - square, mid_point[0] + square)
    axins.set_ylim(mid_point[1] - square, mid_point[1] + square)
    fig_map.indicate_inset_zoom(axins, edgecolor="black")
    for ctx in contexts:
        start = mng[ctx][1][0]
        end = mng[ctx][1][1]
        # fig_map.plot(position_curr[start:end, 0], position_curr[start:end, 1], '-', color=COLORS[ctx])
        axins.plot(position_curr[start:end:10, 0], position_curr[start:end:10, 1],
                   marker=MARKERS[ctx], color=COLORS_CTX[ctx], linestyle='', markersize=5, alpha=0.8, markevery=0.05)
        # fig_map.plot(position_curr[end, 0], position_curr[end, 1], '-', color=COLORS[ctx])


    # if start_goal is not None:
    #     fig_map.scatter([start_goal[0][0]], [start_goal[0][1]])
        # plot.annotate("start", ([start_goal[0][0]], [start_goal[0][1]]), textcoords="offset points", xytext=(0, 10), ha='center')


    # fig_map.plot([position_curr[-1, 0]], [position_curr[-1, 1]], 'ob')
    if damage_iter is not None and record["t"][-1] > damage_iter:
        dmg_x = position_curr[damage_iter, 0]
        dmg_y = position_curr[damage_iter, 1]
        fig_map.scatter([dmg_x], [dmg_y], marker='o', s=100, c='white', edgecolors='r')
        # fig_map.plot([dmg_x], [dmg_y],  color='red', marker='x', linestyle='', markersize=12)

        fig_map.annotate("damage", (dmg_x, dmg_y), textcoords="offset points", xytext=(0,10), ha='center')

    if start_goal is not None:
        _x, _y = start_goal[0]
        fig_map.scatter([_x], [_y], marker='o', s=100, c='white', edgecolors='r')
        # fig_map.plot([_x], [_y],  color='red', marker='x', linestyle='', markersize=12)
        fig_map.annotate("start", (_x, _y), textcoords="offset points", xytext=(15,-10), ha='center')
        _x, _y = start_goal[1]
        fig_map.scatter([_x], [_y], marker='o', s=100, c='white', edgecolors='r')
        # fig_map.plot([_x], [_y],  color='red', marker='x', linestyle='', markersize=12)
        fig_map.annotate("goal", (_x, _y), textcoords="offset points", xytext=(0,-10), ha='center')


def sensory_ref_clearance(record, fig: Figure, sensors=(0, 3), contexts=(0, 1, 2), title=None, damage_iter=None):
    if title is not None:
        fig.suptitle(title)
    t = record["t"]
    # clearance
    a_s = record[CTR.STATE_PHASE_ACTIVATOR]
    y_inp = record[CTR.SENSORY_INPUT]
    y_ref = record[CTR.SENSORY_REFERENCE]
    # model error
    q_perf = record[CTR.REAL_PERFORMANCE_QUALITY]
    q = record[CTR.CONTEXT_MODEL_QUALITY][:, 0, :]
    q_mu = record[CTR.CONTEXT_MODEL_QUALITY_MEAN][:, 0, :]
    th_in = record[CTR.CONTEXT_QUALITY_INNER_THRESHOLD_DYN]
    th_out = record[CTR.CONTEXT_QUALITY_OUTER_THRESHOLD_DYN]


    mng = []
    for i in range(record[CTR.MODEL_WEIGHTS].shape[-1]):
        mng.append(get_context_management_durations(record, i))

    # combine
    y_ref_cmb = np.einsum("timjc,tic->tim", y_ref, a_s)
    N = 10

    fi = fig.subplots(len(sensors) + 2, 1)
    f_est_err = fi[0]
    f_prf_err = fi[-1]

    # ESTIMATION ERROR and LIFE-CYCLE CONTROL
    # f_est_err.plot(t, np.log10(th_in), color='k', label=CTR.CONTEXT_QUALITY_INNER_THRESHOLD_DYN)
    f_est_err.plot(t, np.log10(th_out), color='k', label=CTR.CONTEXT_QUALITY_OUTER_THRESHOLD_DYN)

    q_mu_l = np.log10(q_mu)
    lab_pos_d, lab_pos_u = (np.max(q_mu_l) + 0.1, np.max(q_mu_l) + 0.6)
    lab_pos_mid = (lab_pos_d + lab_pos_u) / 2

    # f.set_yticks([yf_mid, -1, 0, 1, 2])
    # f.set_yticklabels(['stage', -1, 0, 1, 2])
    f_est_err.set_ylabel(ESTIMATION_ERROR_LABEL)

    for ctx in contexts:
        f_est_err.plot(t, q_mu_l[:, ctx], color=COLORS_CTX[ctx], marker=MARKERS[ctx], markevery=0.05)

        f_est_err.axvline(mng[ctx][0][0], color=COLORS_CTX[ctx], linestyle='-')
        f_est_err.axvline(mng[ctx][1][0], color=COLORS_CTX[ctx], linestyle='--')

        lrn_start, lrn_end = mng[ctx][0]
        ctr_start, ctr_end = mng[ctx][1]
        yf = np.arange(lab_pos_d, lab_pos_u, 0.01)
        f_est_err.fill_betweenx(yf, lrn_start, ctr_end, color=COLORS_CTX[ctx], alpha=0.7)

        _x = (lrn_start + lrn_end) / 2
        f_est_err.annotate("learn", (_x, lab_pos_mid), textcoords="offset points", xytext=(0, -4), ha='center')

        _x = (ctr_start + ctr_end) / 2
        f_est_err.annotate("control", (_x, lab_pos_mid), textcoords="offset points", xytext=(0, -4), ha='center')

    # PERFORMANCE ERROR
    f_prf_err.plot(t, np.log10(q_perf), 'k', alpha=0.8)
    f_prf_err.set_ylabel(PERFORMANCE_ERROR_LABEL)

    # MOTOR CONTROL
    for i, m in enumerate(sensors):
        f = fi[i+1]
        f.set_ylabel(SENSORY_NAMES[m])
        if m==0:
            # y = np.clip(y_inp[:, 0, m], -1, 1)
            y = y_inp[:, 0, m]
            y_ref = np.maximum(y_ref_cmb[:, 0, m], 0)
        else:
            y = y_inp[:, 0, m]
            y_ref = y_ref_cmb[:, 0, m]
        y_inp_mean = np.convolve(y, np.ones(N) / N, mode='valid')
        f.plot(t[N-1:], y_inp_mean, 'k', label=LABELS[CTR.SENSORY_INPUT], alpha=0.8)
        for ctx in contexts:
            ctr_start, ctr_end = mng[ctx][1]
            f.plot(t[ctr_start:ctr_end], y_ref[ctr_start:ctr_end], 'g')


    for i, f in enumerate(fi):
        f.set_xlim(0, len(t))
        for ctx in contexts:
            f.axvline(mng[ctx][0][0], color=COLORS_CTX[ctx], linestyle='-')
            f.axvline(mng[ctx][1][0], color=COLORS_CTX[ctx], linestyle='--')
        if damage_iter is not None:
            f.axvline(damage_iter, color='r', linestyle='-.')

        if i == 0:
            f.annotate("damage", (damage_iter, 3), textcoords="offset points", xytext=(0, -4), ha='left')
        if i != len(fi)-1:
            f.set_xticklabels([])
        else:
            # f.annotate("damage", (damage_iter, 4), textcoords="offset points", xytext=(0, -4), ha='left')
            f.set_xlabel(TIME_LABEL)


    # fi[-1].legend()


def context_control_evol(record, fig: Figure, contexts=(0,1,2)):
    fig.suptitle("Ensemble Interaction Control")
    t = record["t"]
    q_perf = record[CTR.REAL_PERFORMANCE_QUALITY]
    q = record[CTR.CONTEXT_MODEL_QUALITY][:, 0, :]
    q_mu = record[CTR.CONTEXT_MODEL_QUALITY_MEAN][:, 0, :]
    th_in = record[CTR.CONTEXT_QUALITY_INNER_THRESHOLD_DYN]
    th_out = record[CTR.CONTEXT_QUALITY_OUTER_THRESHOLD_DYN]

    a = record[CTR.CONTEXT_ACTIVABLE][:, 0, :]

    opt_fr = record[CTR.CONTEXT_IS_BEST_FREEZED][:, 0, :]
    opt_ho = record[CTR.CONTEXT_IS_BEST_HOT][:, 0, :]

    lrn_s = record[CTR.CONTEXT_LEARNING_SELECTION][:, 0, :]
    ctr_s = record[CTR.CONTEXT_CONTROL_SELECTION][:, 0, :]
    cmd_s = record[CTR.CONTEXT_COMMANDING_SELECTION][:, 0, :]

    lrn_state = record[CTR.CONTEXT_LEARNING_STATE][:, 0, :]

    f = fig.subplots(3, 1)

    fig_q_perf = f[2]
    fig_q_perf.set_title(CTR.REAL_PERFORMANCE_QUALITY)
    fig_q = f[0]
    fig_q.set_title(CTR.CONTEXT_MODEL_QUALITY)
    fig_a = f[1]
    fig_a.set_title("Control signals")

    fig_q_perf.plot(t, np.log10(q_perf), color='b')

    fig_q.plot(t, np.log10(th_in), color='k', label=CTR.CONTEXT_QUALITY_INNER_THRESHOLD_DYN)
    fig_q.plot(t, np.log10(th_out), color='k', label=CTR.CONTEXT_QUALITY_OUTER_THRESHOLD_DYN)

    fig_a.hlines([-1.5 * i - 0.1 for i in range(2)], np.min(t), np.max(t), colors='k', alpha=0.5)
    fig_a.hlines([-1.5 * i + 1.1 for i in range(2)], np.min(t), np.max(t), linestyles='--', colors='k', alpha=0.5)

    for ctx in contexts:
        # fig_q.plot(t, np.log10(q[:, ctx]), linestyle='--', color=COLORS[ctx], alpha=0.5)
        fig_q.plot(t, np.log10(q_mu[:, ctx]), color=COLORS_CTX[ctx], label="log " + str(ctx))

        # fig_a.plot(t, a[:, ctx] - 1.5 * 0, color=COLORS[ctx], alpha=0.8)
        # fig_a.plot(t, opt_fr[:, ctx] - 1.5 * 1, color=COLORS[ctx], alpha=0.8)
        # fig_a.plot(t, opt_ho[:, ctx] - 1.5 * 2, color=COLORS[ctx], alpha=0.8)
        fig_a.plot(t, lrn_s[:, ctx] - 1.5 * 0, color=COLORS_CTX[ctx], alpha=0.8)
        fig_a.plot(t, ctr_s[:, ctx] - 1.5 * 1, color=COLORS_CTX[ctx], alpha=0.8)
        # fig_a.plot(t, cmd_s[:, ctx] - 1.5 * 5, color=COLORS[ctx], alpha=0.8)
        # fig_a.plot(t, lrn_state[:, ctx] - 1.5 * 6, color=COLORS[ctx], alpha=0.8)

    # fig_a.set_yticks([-1.5 * i for i in range(7)])
    # fig_a.set_yticklabels(
    #     [CTR.CONTEXT_ACTIVABLE, CTR.CONTEXT_IS_BEST_FREEZED, CTR.CONTEXT_IS_BEST_HOT, CTR.CONTEXT_LEARNING_SELECTION,
    #      CTR.CONTEXT_CONTROL_SELECTION, CTR.CONTEXT_COMMANDING_SELECTION, CTR.CONTEXT_LEARNING_STATE]
    # )
    fig_a.set_yticks([-1.5 * i for i in range(2)])
    fig_a.set_yticklabels(
        [CTR.CONTEXT_LEARNING_SELECTION, CTR.CONTEXT_CONTROL_SELECTION]
    )

    fig_q.legend()


def sensory_estimation(record, fig: Figure, title=None, ctx=0, sensors=(0, 1, 2, 3),
                       t_ranges=((0, -1), (0, -1), (0, -1)), subtitles=("a", "b", "c")):
    if title is not None:
        fig.suptitle(title)
    t = record["t"]
    a_s = record[CTR.STATE_PHASE_ACTIVATOR]
    a_ep = record[CTR.EPICYCLE_PHASE_ACTIVATOR]
    y_est = record[CTR.SENSORY_ESTIMATION][:, :, :, :, :, :, ctx]
    y_inp = record[CTR.SENSORY_INPUT]
    var = record[CTR.PROB_VAR_SENSORY_MOTOR][:, :, :, :, :, ctx]

    quality = record[CTR.CONTEXT_MODEL_QUALITY][:, 0, ctx]
    # quality = record[CTR.CONTEXT_MODEL_QUALITY_MEAN][:, 0, ctx]

    mng = []
    for i in range(record[CTR.MODEL_WEIGHTS].shape[-1]):
        mng.append(get_context_management_durations(record, i))

    # combine
    y_est_cmb = np.einsum("timjck,tic,tik->tim", y_est, a_s, a_ep)
    var_cmb = np.einsum("timjc,tic->tim", var, a_s)
    std_cmb = np.sqrt(var_cmb)

    sensor_m = y_inp.shape[-1]
    fi = fig.subplots(len(sensors) + 1, len(t_ranges))
    for i, m in enumerate(sensors):
        for j, rng in enumerate(t_ranges):
            f = fi[i][j]
            start, end = rng
            _t = t[start:end]
            f.plot(_t, y_inp[start:end, 0, m], 'k', alpha=0.7, label=CTR.SENSORY_INPUT)
            f.plot(_t, y_est_cmb[start:end, 0, m], 'b', alpha=0.8, label=CTR.SENSORY_ESTIMATION)
            # f.plot(_t, (y_est_cmb + std_cmb)[start:end, 0, m], 'b--', alpha=0.8)
            # var_low = (y_est_cmb - std_cmb)[start:end, 0, m]
            # f.plot(_t, var_low, 'b--', alpha=0.8, label="sqrt " + CTR.PROB_VAR_SENSORY_MOTOR)
            # err = np.sqrt(np.square(y_inp[start:end, 0, m] - y_est_cmb[start:end, 0, m]))
            # err_mean = np.log10(np.convolve(err, np.ones(N) / N, mode='valid'))
            # shift = -np.max(err_mean) + np.min(var_low)
            #
            # f.plot(_t[N-1:], err_mean + shift - 0.1, 'r')

    glob_start = t_ranges[0][0]
    glob_end = t_ranges[-1][1]
    ql = np.log10(quality[glob_start:glob_end])
    ql_min = np.min(ql)
    ql_max = np.max(ql)
    for j, rng in enumerate(t_ranges):
        f = fi[-1][j]
        start, end = rng
        f.plot(t[start:end], quality[start:end], 'k')
        # f.set_ylim(ql_min - 0.3, ql_max + 0.3)

    if subtitles is not None:
        for j, rng in enumerate(t_ranges):
            f = fi[0][j]
            f.set_title(subtitles[j])

    fi[-1][0].set_ylabel(ESTIMATION_ERROR_LABEL)
    # fi[0][0].legend()


def sensory_estimation_single(record, fig: Figure, title=None, ctx=0, sensors=(0, 1, 2, 3), interval=(0, -1),
                              damage_iter=None, learning_start=None, show_y_labels=True, stat_view=False):
    if title is not None:
        fig.suptitle(title)
    t = record["t"]
    a_s = record[CTR.STATE_PHASE_ACTIVATOR]
    a_ep = record[CTR.EPICYCLE_PHASE_ACTIVATOR]
    y_est = record[CTR.SENSORY_ESTIMATION][:, :, :, :, :, :, ctx]
    y_inp = record[CTR.SENSORY_INPUT]
    var = record[CTR.PROB_VAR_SENSORY_MOTOR][:, :, :, :, :, ctx]

    quality = record[CTR.CONTEXT_MODEL_QUALITY][:, 0, ctx]
    # quality = record[CTR.CONTEXT_MODEL_QUALITY_MEAN][:, 0, ctx]

    lrn_mng, ctr_mng, cmd_mng = get_context_management_durations(record, ctx)

    # combine
    y_est_cmb = np.einsum("timjck,tic,tik->tim", y_est, a_s, a_ep)
    var_cmb = np.einsum("timjc,tic->tim", var, a_s)
    std_cmb = np.sqrt(var_cmb)
    start, end = interval

    fi = fig.subplots(len(sensors) + 1, 1)
    for i, m in enumerate(sensors):
        f = fi[i]
        _t = t[start:end]
        f.plot(_t, y_inp[start:end, 0, m], 'k', alpha=0.5, label=CTR.SENSORY_INPUT)
        f.plot(_t, y_est_cmb[start:end, 0, m], color=COLORS_CTX[ctx], alpha=0.7, label=CTR.SENSORY_ESTIMATION)
        f.set_xticks([])
        if show_y_labels:
            f.set_ylabel(SENSORY_NAMES[m])


    ###
    f = fi[-1]
    conv_win = 10
    if not stat_view:
        f.plot(t[start:end], quality[start:end], color=COLORS_CTX[ctx], linestyle='-')
        if show_y_labels:
            f.set_ylabel(ESTIMATION_ERROR_LABEL)
        f.set_xlabel(TIME_LABEL)
    else:
        f.plot(t[start:end], quality[start:end], color='k', linestyle='-', alpha=0.3)
        conv_quality = np.convolve(quality[start:end], np.ones(conv_win)/conv_win, mode='valid')
        f.plot(t[start+conv_win-1:end], conv_quality, color=COLORS_CTX[ctx], linestyle='-')

    if show_y_labels:
        f.set_ylabel(ESTIMATION_ERROR_LABEL)
    f.set_xlabel(TIME_LABEL)
    ###
    lrn_start = lrn_mng[0]
    ctr_start = ctr_mng[0]
    max_q = (np.min(quality[start:end]) + np.max(quality[start:end])) / 2

    for i, f in enumerate(fi):
        if start < lrn_start < end:
            f.axvline(lrn_start, color=COLORS_CTX[ctx], linestyle='-')
            if i == len(fi) - 1:
                f.annotate(" learning", (lrn_start, max_q), textcoords="offset points", xytext=(1, 0), ha='left')

        if start < ctr_start < end:
            f.axvline(ctr_start, color=COLORS_CTX[ctx], linestyle='--')
            if i == len(fi) - 1:
                f.annotate("control", (ctr_start, max_q), textcoords="offset points", xytext=(1, 0), ha='left')

        if damage_iter is not None and start < damage_iter < end:
            f.axvline(damage_iter, color='r', linestyle='-.')
            if i == len(fi) - 1:
                f.annotate("damage", (damage_iter, max_q), textcoords="offset points", xytext=(1, 0), ha='left')

        if learning_start is not None and start < learning_start < end:
            f.axvline(learning_start, color='k', linestyle='-')
            if i == len(fi) - 1:
                f.annotate(" learning", (learning_start, max_q), textcoords="offset points", xytext=(0, -4), ha='left')


def sensory_estimation_double(record, fig: Figure, ctx_previous, ctx_current,
                              title=None, sensors=(0, 1, 2, 3), interval=(0, -1),
                              damage_iter=None, learning_start=None, show_y_labels=True, stat_view=False):
    if title is not None:
        fig.suptitle(title)
    t = record["t"]
    a_s = record[CTR.STATE_PHASE_ACTIVATOR]
    a_ep = record[CTR.EPICYCLE_PHASE_ACTIVATOR]
    y_inp = record[CTR.SENSORY_INPUT]


    y_est_cur = record[CTR.SENSORY_ESTIMATION][:, :, :, :, :, :, ctx_current]
    var_cur = record[CTR.PROB_VAR_SENSORY_MOTOR][:, :, :, :, :, ctx_current]
    quality_cur = np.log(record[CTR.CONTEXT_MODEL_QUALITY][:, 0, ctx_current])
    lrn_mng_cur, ctr_mng_cur, cmd_mng_cur = get_context_management_durations(record, ctx_current)

    y_est_prv = record[CTR.SENSORY_ESTIMATION][:, :, :, :, :, :, ctx_previous]
    var_prv = record[CTR.PROB_VAR_SENSORY_MOTOR][:, :, :, :, :, ctx_previous]
    quality_prv = np.log(record[CTR.CONTEXT_MODEL_QUALITY][:, 0, ctx_previous])
    lrn_mng_prv, ctr_mng_prv, cmd_mng_prv = get_context_management_durations(record, ctx_previous)
    lrn_start = lrn_mng_cur[0]
    ctr_start = ctr_mng_cur[0]

    # combine
    y_est_cmb_cur = np.einsum("timjck,tic,tik->tim", y_est_cur, a_s, a_ep)
    y_est_cmb_prv = np.einsum("timjck,tic,tik->tim", y_est_prv, a_s, a_ep)


    start, end = interval
    _t_prev = t[start:lrn_start]
    _t_curr = t[lrn_start:end]
    _t = t[start:end]

    fi = fig.subplots(len(sensors) + 1, 1)
    for i, m in enumerate(sensors):
        f = fi[i]

        f.plot(_t, y_inp[start:end, 0, m], 'k', alpha=0.5,
               label="Ground truth")

        # prev
        f.plot(_t_prev, y_est_cmb_prv[start:lrn_start, 0, m], color=COLORS_CTX[ctx_previous], alpha=0.7,
               label="Context {}".format(ctx_previous))
        # cur
        f.plot(_t_curr, y_est_cmb_cur[lrn_start:end, 0, m], color=COLORS_CTX[ctx_current], alpha=0.7,
               label="Context {}".format(ctx_current))

        f.set_xticks([])
        if show_y_labels:
            f.set_ylabel(SENSORY_NAMES[m])
    fi[0].legend()


    ###
    f = fi[-1]
    f.plot(_t, quality_prv[start:end], color=COLORS_CTX[ctx_previous], linestyle='-')
    f.plot(_t_curr[:], quality_cur[lrn_start:end], color=COLORS_CTX[ctx_current], linestyle='-')
    if show_y_labels:
        f.set_ylabel(ESTIMATION_ERROR_LABEL)
    f.set_xlabel(TIME_LABEL)


    if show_y_labels:
        f.set_ylabel(ESTIMATION_ERROR_LABEL)
    f.set_xlabel(TIME_LABEL)
    ###

    max_q = np.min(quality_cur[start:end]) * 0.3 + np.max(quality_cur[start:end]) * 0.7

    for i, f in enumerate(fi):
        if start < lrn_start < end:
            f.axvline(lrn_start, color='k', linestyle='-', alpha=0.5)
            if i == len(fi) - 1:
                f.annotate(" context {} Learning".format(ctx_current), ((lrn_start + end)/2, max_q), textcoords="offset points", xytext=(0, 0), ha='center')
                f.annotate(" context {} Controlling".format(ctx_previous), ((lrn_start + start)/2, max_q), textcoords="offset points", xytext=(0, 0), ha='center')

        if learning_start is not None and start < learning_start < end:
            f.axvline(learning_start, color='k', linestyle='-')
            if i == len(fi) - 1:
                f.annotate(" learning", (learning_start, max_q), textcoords="offset points", xytext=(0, -4), ha='left')



def embedding_visualisation(record, fig: Figure, title=None, n_motor=0, interval=(0, 100), ep=17, phs=(3, 5)):
    start, end = interval
    if title is not None:
        fig.suptitle(title)
    t = record["t"][start:end]
    label_iter = -60
    label_time = t[label_iter]
    a_s = record[CTR.STATE_PHASE_ACTIVATOR]
    a_ep = record[CTR.EPICYCLE_PHASE_ACTIVATOR]

    print(np.unique(np.argmax(a_ep, axis=2)[:, 0]))
    u_cmd = record[CTR.MOTOR_COMMAND][start:end, 0, n_motor]
    u_cmd_max = np.max(u_cmd)
    u_cmd_min = np.min(u_cmd)
    diff = u_cmd_max - u_cmd_min

    u_mem = record[CTR.MOTOR_MEMORY][start:end, 0, n_motor, 0, :, :]
    _T, _C, _E = u_mem.shape
    fi = fig.subplots(1, 1)
    fi.plot(t - 0.03, u_cmd, 'k', alpha=0.7)

    cpg_size = np.abs(diff) / 4
    shift_ticks = []
    shift_labels = []
    for i, p in enumerate(phs):
        col = i

        ph = a_s[start:end, 0, p] * a_ep[start:end, 0, ep]
        fi.plot(t[ph > 0.5], u_mem[ph > 0.5, p, ep], '.', color=COLORS_PHS[col], markevery=0.005)
        fi.plot(t, u_mem[:, p, ep], color=COLORS_PHS[col], alpha=0.8)

        shift = u_cmd_min * 2 - ((1 + i) * cpg_size * 1.2)
        shift_ticks.append(shift)
        shift_labels.append("$\Phi_{}$".format(p))
        fi.plot(t, [shift]*len(t), 'k--', alpha=0.3)
        fi.plot(t, a_s[start:end, 0, p] * cpg_size + shift, color=COLORS_PHS[col])
        ##
        a_sd = a_s[start+1:end, 0, p] - a_s[start:end-1, 0, p]
        a_sd_xup = np.where(a_sd > 0.5)[0]
        a_sd_xdown = np.where(a_sd < -0.5)[0]
        for j in range(len(a_sd_xup)):
            t_up = t[a_sd_xup[j]]
            t_down = t[a_sd_xdown[j]]
            t_mid = np.abs((t_down - t_up)/2)
            fi.axvline(t_up + t_mid, color=COLORS_PHS[col], linestyle='--', alpha=0.7)
            fi.annotate(r"$\upsilon_{}$".format(p), (label_time, u_mem[label_iter, p, ep]),
                        textcoords="offset points", xytext=(0, 3), ha='center')

    fi.annotate(r"$u$", (label_time, u_cmd[label_iter]),
                textcoords="offset points", xytext=(0, 3), ha='center')

    fi.fill_between(t[a_ep[start:end, 0, ep] > 0.5], u_cmd_min - 0.1, u_cmd_max + 0.1, alpha=0.5, color='tab:green')
    fi.fill_between(t[a_ep[start:end, 0, ep-1] > 0.5], u_cmd_min - 0.1, u_cmd_max + 0.1, alpha=0.5, color='gold')
    fi.set_xlim(t[0], t[-1])

    # u_ticks = np.linspace(u_cmd_min,u_cmd_max,4)
    u_ticks = [np.floor(u_cmd_min * 10)/10, 0, np.ceil(u_cmd_max*10)/10]
    fi.set_xlim(t[0], t[-1] - 0.05)

    fi.set_yticks(u_ticks + shift_ticks)
    fi.set_yticklabels(list(map(str, u_ticks)) + shift_labels)
    fi.set_xlabel(TIME_LABEL)
    fi.set_ylabel("Motor command, Embedding")
    ###
    a_ep_d = a_ep[start + 1:end, 0, ep] - a_ep[start:end - 1, 0, ep]
    a_ep_start = np.where(a_ep_d > 0.5)[0][0]
    fi.annotate(r"perturbed gait", (t[a_ep_start], u_cmd_max),
                textcoords="offset points", xytext=(1, 3), ha='left')

    ##
    a_ep_d = a_ep[start + 1:end, 0, ep-1] - a_ep[start:end - 1, 0, ep-1]
    sel = np.where(a_ep_d > 0.5)
    if len(sel[0]) == 0:
        a_ep_start = 0
    else:
        a_ep_start = np.where(a_ep_d > 0.5)[0][0]
    fi.annotate(r"base gait", (t[a_ep_start], u_cmd_max),
                textcoords="offset points", xytext=(1, 3), ha='left')


def embedding_phasespace(record, fig: Figure, title=None, n_motor=0, interval=(0, 100), ep=17, phs=(3,5)):
    start, end = interval
    if title is not None:
        fig.suptitle(title)

    a_ep = record[CTR.EPICYCLE_PHASE_ACTIVATOR]

    print(np.unique(np.argmax(a_ep, axis=2)[:, 0]))

    u_mem = record[CTR.MOTOR_MEMORY][start:end, 0, n_motor, 0, :, :]
    _T, _C, _E = u_mem.shape
    fi = fig.subplots(1, 1)

    ##
    a_ep_d = a_ep[start + 1:end, 0, ep] - a_ep[start:end - 1, 0, ep]
    a_ep_end = np.where(a_ep_d < -0.5)[0][0]
    fi.scatter(u_mem[a_ep_end, phs[0], ep], u_mem[a_ep_end, phs[1], ep],s=100, color='tab:green')
    fi.annotate(r"perturbed gait", (u_mem[a_ep_end, phs[0], ep], u_mem[a_ep_end, phs[1], ep]),
                textcoords="offset points", xytext=(6, -1), ha='left')
    ##
    a_ep_d = a_ep[start + 1:end, 0, ep-1] - a_ep[start:end - 1, 0, ep-1]
    a_ep_end = np.where(a_ep_d < -0.5)[0][0]
    fi.scatter(u_mem[a_ep_end, phs[0], ep], u_mem[a_ep_end, phs[1], ep],s=100, color='gold')
    fi.annotate(r"base gait", (u_mem[a_ep_end, phs[0], ep], u_mem[a_ep_end, phs[1], ep]),
                textcoords="offset points", xytext=(0, 6), ha='right')

    fi.plot(u_mem[:, phs[0], ep], u_mem[:, phs[1], ep], 'k')


    ##
    fi.set_xlabel(r"embedding $\upsilon_{}$".format(phs[0]))
    fi.set_ylabel(r"embedding $\upsilon_{}$".format(phs[1]))


def compare_performance_error_aggrs_observed_order(records, fig: Figure, forcing_frequencies, mean_interval=(0, -1), ctx=1, show_ylabel=True):
    start, end = mean_interval
    t = records[0]["t"]


    ep_phases = [record[CTR.EPICYCLE_PHASE][:, 0] for record in records]

    perf_err = [record[CTR.REAL_PERFORMANCE_QUALITY] for record in records]
    estm_err = [record[CTR.CONTEXT_MODEL_QUALITY][:, 0, ctx] for record in records]
    perf_mean = np.asarray([np.log10(np.mean(sig[start:end])) for sig in perf_err])
    estm_mean = np.asarray([np.log10(np.mean(sig[start:end])) for sig in estm_err])

    obs_frequencies = np.asarray([(phs[-1, 0] - phs[0, 0])/(t[-1] - t[0]) for phs in ep_phases]) * 64
    ordering = np.argsort(obs_frequencies)

    f = fig.subplots(2, 1)
    figr_aggregate_estm = f[0]
    figr_aggregate_perf = f[1]

    figr_aggregate_perf.plot(obs_frequencies[ordering], perf_mean[ordering], 'o-', color='k')
    for i in range(len(records)):
        figr_aggregate_perf.annotate("$\omega_p={}$".format(np.ceil(forcing_frequencies[i]*100)/100),
                    (obs_frequencies[i], perf_mean[i]),
                    textcoords="offset points", xytext=(6, -1), ha='left')

    figr_aggregate_estm.plot(obs_frequencies[ordering], estm_mean[ordering], 'o-', color='k')
    for i in range(len(records)):
        figr_aggregate_estm.annotate("$\omega_p={}$".format(np.ceil(forcing_frequencies[i]*100)/100),
                    (obs_frequencies[i], estm_mean[i]),
                    textcoords="offset points", xytext=(6, -1), ha='left')

    f[0].set_xticks([])
    # f[1].set_xticks(obs_frequencies[ordering])
    # f[1].set_xticklabels(["{}".format(np.ceil(frq*100)/100) for frq in obs_frequencies[ordering]])
    f[1].set_xlabel(OBSERVED_FREQUENCY_LABEL)

    f[0].spines['right'].set_visible(False)
    f[0].spines['top'].set_visible(False)
    f[0].spines['bottom'].set_visible(False)

    f[1].spines['right'].set_visible(False)
    f[1].spines['top'].set_visible(False)

    if show_ylabel:
        figr_aggregate_estm.set_ylabel("Estimation error")
        figr_aggregate_perf.set_ylabel("Performation error")


def compare_performance_error_aggrs(records, fig: Figure, forcing_frequencies, mean_interval=(0, -1), ctx=1):
    start, end = mean_interval

    perf_err = [record[CTR.REAL_PERFORMANCE_QUALITY] for record in records]
    estm_err = [record[CTR.CONTEXT_MODEL_QUALITY][:, 0, ctx] for record in records]

    f = fig.subplots(2, 1)
    figr_aggregate_estm = f[0]
    figr_aggregate_perf = f[1]

    figr_aggregate_perf.plot(forcing_frequencies, [np.log10(np.mean(sig[start:end])) for sig in perf_err], 'o-', color='k')
    figr_aggregate_estm.plot(forcing_frequencies, [np.log10(np.mean(sig[start:end])) for sig in estm_err], 'o-', color='k')

    f[0].set_xticks([])
    f[1].set_xticks(forcing_frequencies)
    f[1].set_xticklabels(["{}".format(np.ceil(frq*100)/100) for frq in forcing_frequencies])
    f[1].set_xlabel(RHYTHM_FREQUENCY_LABEL)


def compare_performance_error_evol(records, fig: Figure, frequencies, ctx=0, conv_mean_window=1000):

    perf_err = [record[CTR.REAL_PERFORMANCE_QUALITY] for record in records]
    estm_err = [record[CTR.CONTEXT_MODEL_QUALITY][:, 0, ctx] for record in records]

    conv_perf_err = np.log10(np.asarray([np.convolve(err, np.ones(conv_mean_window) / conv_mean_window, mode='valid') for err in perf_err]))
    conv_estm_err = np.log10(np.asarray([np.convolve(err, np.ones(conv_mean_window) / conv_mean_window, mode='valid') for err in estm_err]))

    perf_err_diff = conv_perf_err.max() - conv_perf_err.min()
    estm_err_diff = conv_estm_err.max() - conv_estm_err.min()

    conv_perf_err = (conv_perf_err - conv_perf_err.min())/perf_err_diff
    conv_estm_err = (conv_estm_err - conv_estm_err.min())/estm_err_diff

    f = fig.subplots(2, 1)

    figr_evols_estm = f[0]
    figr_evols_perf = f[1]
    shift = 1.8
    _t = records[0]["t"][conv_mean_window - 1:]

    for i in range(len(records)):

        # for j in range(len(records)):
        #     figr_evols_perf.plot(_t, conv_perf_err[j] + i * shift, color='w', alpha=1)
        #     figr_evols_estm.plot(_t, conv_estm_err[j] + i * shift, color='w', alpha=1)

        figr_evols_perf.plot(_t, np.mean(conv_perf_err, axis=0) + i * shift, '--', color='w', alpha=1)
        figr_evols_estm.plot(_t, np.mean(conv_estm_err, axis=0) + i * shift, '--', color='w', alpha=1)

        figr_evols_perf.plot(_t, conv_perf_err[i] + i * shift, '-', color='k')
        figr_evols_estm.plot(_t, conv_estm_err[i] + i * shift, '-', color='k')

        figr_evols_perf.fill_between(_t, -0.1 + i * shift, 1.1 + i * shift, alpha=0.4, color='k', edgecolor="None" )
        figr_evols_estm.fill_between(_t, -0.1 + i * shift, 1.1 + i * shift, alpha=0.4, color='k', edgecolor="None" )

    figr_evols_estm.set_ylabel(RHYTHM_FREQUENCY_LABEL)
    figr_evols_estm.set_xlim(_t[0], _t[-1])
    figr_evols_estm.set_xticks([])
    figr_evols_estm.set_yticks([0.5 + i * shift for i in range(len(records))])
    figr_evols_estm.set_yticklabels(["{}".format(np.ceil(frq * 100) / 100) for frq in frequencies])
    figr_evols_estm.spines['right'].set_visible(False)
    figr_evols_estm.spines['top'].set_visible(False)
    figr_evols_estm.spines['left'].set_visible(False)
    figr_evols_estm.spines['bottom'].set_visible(False)

    figr_evols_perf.set_ylabel(RHYTHM_FREQUENCY_LABEL)
    figr_evols_perf.set_yticks([0.5 + i * shift for i in range(len(records))])
    figr_evols_perf.set_yticklabels(["{}".format(np.ceil(frq * 100) / 100) for frq in frequencies])
    figr_evols_perf.set_xlabel(TIME_LABEL)
    figr_evols_perf.set_xlim(_t[0], _t[-1])
    figr_evols_perf.spines['right'].set_visible(False)
    figr_evols_perf.spines['top'].set_visible(False)
    figr_evols_perf.spines['left'].set_visible(False)
    # figr_evols_perf.spines['bottom'].set_visible(False)


def compare_sensory_evolutions(records, fig: Figure, frequencies, sel_sens=(0,3), conv_mean_window=1000):

    f = fig.subplots(len(sel_sens), 1)

    shift = 1.8
    _t = records[0]["t"][conv_mean_window - 1:]

    for fig_n, m in enumerate(sel_sens):
        figr = f[fig_n]
        sens_inp = np.asarray([record[CTR.SENSORY_INPUT][:, 0, m] for record in records])
        conv_sens_inp = np.asarray(
            [np.convolve(err, np.ones(conv_mean_window) / conv_mean_window, mode='valid') for err in sens_inp])
        sens_diff = conv_sens_inp.max() - conv_sens_inp.min()
        conv_sens_inp = (conv_sens_inp - conv_sens_inp.min()) / sens_diff

        for i in range(len(records)):

            figr.plot(_t, np.mean(conv_sens_inp, axis=0) + i * shift, '--', color='w', alpha=1)
            figr.plot(_t, conv_sens_inp[i] + i * shift, '-', color='k')
            figr.fill_between(_t, -0.1 + i * shift, 1.1 + i * shift, alpha=0.4, color='k', edgecolor="None" )


        if fig_n != len(sel_sens) - 1:
            figr.set_xticks([])
            figr.spines['bottom'].set_visible(False)

        figr.set_ylabel(RHYTHM_FREQUENCY_LABEL)
        figr.set_xlim(_t[0], _t[-1])
        figr.set_yticks([0.5 + i * shift for i in range(len(records))])
        figr.set_yticklabels(["{}".format(np.ceil(frq * 100) / 100) for frq in frequencies])
        figr.spines['right'].set_visible(False)
        figr.spines['top'].set_visible(False)
        figr.spines['left'].set_visible(False)

    f[-1].set_xlabel(TIME_LABEL)


def compare_sensory_aggrs_observed_order(records, fig: Figure, forcing_frequencies, mean_interval=(0, -1),
                                         sel_sens=(0, 3), show_ylabel=True):
    start, end = mean_interval
    t = records[0]["t"]

    ep_phases = [record[CTR.EPICYCLE_PHASE][:, 0] for record in records]

    obs_frequencies = np.asarray([(phs[-1, 0] - phs[0, 0])/(t[-1] - t[0]) for phs in ep_phases]) * 64
    ordering = np.argsort(obs_frequencies)

    f = fig.subplots(len(sel_sens), 1)

    for fig_n, m in enumerate(sel_sens):
        figr = f[fig_n]
        sens_inp = [record[CTR.SENSORY_INPUT][:, 0, m] for record in records]
        sens_mean = np.asarray([np.mean(sig[start:end]) for sig in sens_inp])

        figr.plot(obs_frequencies[ordering], sens_mean[ordering], 'o-', color='k')
        for i in range(len(records)):
            figr.annotate("$\omega_p={}$".format(np.ceil(forcing_frequencies[i]*100)/100),
                        (obs_frequencies[i], sens_mean[i]),
                        textcoords="offset points", xytext=(6, -1), ha='left')

        if fig_n != len(sel_sens)-1:
            figr.set_xticks([])
            figr.spines['bottom'].set_visible(False)

        figr.spines['right'].set_visible(False)
        figr.spines['top'].set_visible(False)

        if show_ylabel:
            figr.set_ylabel(SENSORY_NAMES[m])

    f[-1].set_xlabel(OBSERVED_FREQUENCY_LABEL)


def observed_freqs(records, fig: Figure, frequencies, oscillator_phase_name="phs", obs_interval=(10, -1),
                   marked_forced_frequencies_sync=(), marked_forced_frequencies_nonsync=()):
    t = records[0]["t"]
    start, end = obs_interval
    _frequencies = np.asarray(frequencies)
    phases = [record[oscillator_phase_name][:, 0, 0] for record in records]
    obs_vels = [(phase[end] - phase[start])/(t[end] - t[start]) for phase in phases]
    beat_vels = _frequencies/np.asarray(obs_vels)

    f = fig.subplots(1, 1)
    figr_obs = f

    figr_obs.plot(_frequencies, beat_vels, 'k', alpha=0.7)

    label_flag = True
    for frq in marked_forced_frequencies_sync:
        closest = np.argmin(np.abs(_frequencies - frq))
        _frq = _frequencies[closest]
        _beat = beat_vels[closest]
        print("set forced:{}, found forced:{}, found observed:{}".format(frq, _frq, _beat))
        if label_flag:
            figr_obs.plot([frq], [_beat], 'o', color='k', label="synchronous")
            label_flag = False
        else:
            figr_obs.plot([frq], [_beat], 'o', color='k')

    label_flag = True
    for frq in marked_forced_frequencies_nonsync:
        closest = np.argmin(np.abs(_frequencies - frq))
        _frq = _frequencies[closest]
        _beat = beat_vels[closest]
        _obs = obs_vels[closest]
        print("set forced:{}, found forced:{}, found observed:{}".format(frq, _frq, _beat))

        figr_obs.annotate("$\Omega={}$".format(np.ceil(_obs * 100) / 100),
                      (_frq, _beat),
                      textcoords="offset points", xytext=(4, -5), ha='left')

        if label_flag:
            figr_obs.plot([frq], [_beat], '^', color='k', label="non-synchronous")
            label_flag = False
        else:
            figr_obs.plot([frq], [_beat], '^', color='k')

    figr_obs.set_xlabel(RHYTHM_FREQUENCY_LABEL)
    figr_obs.set_ylabel("Speedup rate ($\omega_p \Omega^{-1}$)")
    figr_obs.spines['right'].set_visible(False)
    figr_obs.spines['top'].set_visible(False)
    figr_obs.legend(frameon=False)


def comparison_epicycle_syncing(records, fig: Figure, frequencies, start_end=(0,-1)):
    start, end = start_end
    t = records[0]["t"][start:end]
    _E = records[0][CTR.EPICYCLE_PHASE_ACTIVATOR].shape[2]

    ep_phases = [record[CTR.EPICYCLE_PHASE][start:end, 0] for record in records]
    perturbations = [record[CTR.EPICYCLE_PERTURBATION][start:end] for record in records]

    pert_sels = [np.where(perturbation > 0.9) for perturbation in perturbations]

    f = fig.subplots(1, 1)
    figr_evol = f

    shift = 1.8
    for i in range(len(records)):
        x = np.mod(ep_phases[i][:, 0], 2 * np.pi/64)
        dif_x = np.max(x) - np.min(x)
        x = (x - x.min())/dif_x
        figr_evol.plot(t, x + i * shift, '-', color='k')
        figr_evol.fill_between(t, -0.1 + i * shift, 1.1 + i * shift,
                               alpha=0.4, color='k', where=perturbations[i] < 0.9, edgecolor="None" )
        # figr_evol.fill_between(t[pert_sels[i]], -0.1 + i * shift, 1.1 + i * shift, color='k', alpha=0.5)

    figr_evol.spines['right'].set_visible(False)
    figr_evol.spines['left'].set_visible(False)
    figr_evol.spines['top'].set_visible(False)
    figr_evol.set_xlim(t[0], t[-1])
    figr_evol.set_xlabel(TIME_LABEL)
    figr_evol.set_ylabel(RHYTHM_FREQUENCY_LABEL)
    figr_evol.set_yticks([0.5 + i * shift for i in range(len(records))])
    figr_evol.set_yticklabels(["{}".format(np.ceil(frq * 100) / 100) for frq in frequencies])


