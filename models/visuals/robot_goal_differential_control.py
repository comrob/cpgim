from matplotlib.pyplot import Figure
import models.robot_goal_differential_control as M
import numpy as np


def diff_control_evol(record, fig: Figure, title_psfx=""):
    fig.suptitle("Valhead {}".format(title_psfx))
    t = record["t"]
    velhead = record[M.COMMAND]
    velhead_curr = record[M.CURRENT_COMMAND]
    position_curr = record[M.CURRENT_POSITION]
    coord_trg = record[M.TARGET_POSITION]

    f = fig.subplots(4, 1)

    fig_map = f[0]
    fig_map.set_title("Map")
    fig_evol_vel = f[1]
    fig_evol_vel.set_title("Velocity ")
    fig_evol_head = f[2]
    fig_evol_head.set_title("Heading")
    fig_evol_xy = f[3]
    fig_evol_xy.set_title("XY")

    fig_evol_vel.plot(t, velhead[:, 0], 'k', label=M.COMMAND + " vel")
    fig_evol_vel.plot(t, velhead_curr[:, 0], 'k', label=M.CURRENT_COMMAND + " vel", alpha=0.5)
    fig_evol_vel.legend()

    fig_evol_head.plot(t, velhead[:, 1], 'k', label=M.COMMAND + " head")
    fig_evol_head.plot(t, velhead_curr[:, 1], 'k', label=M.CURRENT_COMMAND + " head", alpha=0.5)
    fig_evol_head.legend()

    fig_map.plot(position_curr[:, 0], position_curr[:, 1], 'b', label=M.CURRENT_POSITION)
    fig_map.plot([position_curr[-1, 0]], [position_curr[-1, 1]], 'ob', label=M.CURRENT_POSITION)
    fig_map.plot(coord_trg[:, 0], coord_trg[:, 1], '.k', label=M.TARGET_POSITION)
    # fig_map.set_xlim(-10, 10)
    # fig_map.set_ylim(-10, 10)
    fig_map.legend()


    fig_evol_xy.plot(t, position_curr[:, 0], 'r', label=M.CURRENT_POSITION + "[x]")
    fig_evol_xy.plot(t, coord_trg[:, 0], '--r', label=M.TARGET_POSITION + "[x]", alpha=0.5)
    fig_evol_xy.plot(t, position_curr[:, 1], 'b', label=M.CURRENT_POSITION + "[y]")
    fig_evol_xy.plot(t, coord_trg[:, 1], '--b', label=M.TARGET_POSITION + "[y]", alpha=0.5)
    fig_evol_xy.legend()


def pid_evol(record, fig: Figure):
    fig.suptitle("Valhead PID")
    t = record["t"]
    velhead = record[M.COMMAND]
    velhead_curr = record[M.CURRENT_COMMAND]
    velhead_int = record[M.INT_COMMAND]
    velhead_der = record[M.DER_COMMAND]
    velhead_mean = record[M.MEAN_COMMAND]

    f = fig.subplots(2, 1)

    fig_evol_vel = f[0]
    fig_evol_vel.set_title("Velocity")
    fig_evol_head = f[1]
    fig_evol_head.set_title("Head")

    fig_evol_vel.plot(t, velhead[:, 0], label=M.COMMAND)
    fig_evol_vel.plot(t, velhead_curr[:, 0], label=M.CURRENT_COMMAND, alpha=0.5)
    fig_evol_vel.plot(t, velhead_int[:, 0], label=M.INT_COMMAND, alpha=0.5)
    fig_evol_vel.plot(t, velhead_der[:, 0], label=M.DER_COMMAND, alpha=0.5)
    fig_evol_vel.plot(t, velhead_mean[:, 0], label=M.MEAN_COMMAND, alpha=0.5)

    fig_evol_head.plot(t, velhead[:, 1], label=M.COMMAND)
    fig_evol_head.plot(t, velhead_curr[:, 1], label=M.CURRENT_COMMAND, alpha=0.5)
    fig_evol_head.plot(t, velhead_int[:, 1], label=M.INT_COMMAND, alpha=0.5)
    fig_evol_head.plot(t, velhead_der[:, 1], label=M.DER_COMMAND, alpha=0.5)
    fig_evol_head.plot(t, velhead_mean[:, 1], label=M.MEAN_COMMAND, alpha=0.5)
    fig_evol_head.legend()
