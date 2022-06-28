from matplotlib.pyplot import Figure
from matplotlib import cm
import models.robot_position_control as M
import models.limit_cycle_controller_contextual as CTR
from models.visuals.limit_cycle_controller_contextual import COLORS
import numpy as np
LEG_DICT = {
    "fl": (0, 2),
    "fr": (1, 3),
    "hl": (4, 6),
    "hr": (5, 7),
    "ml": (8, 10),
    "mr": (9, 11)
}
def position_command_evolution(record, fig: Figure):

    fig.suptitle("Joint Position Command Evolution")
    t = record["t"]
    pos = record[M.POS_COMMAND]


    f = fig.subplots(12, 1)

    for i in range(12):
        f[i].plot(t, pos[:, 0, i])
        f[i].set_ylim([0,1000])
        f[i].set_ylabel("{}".format(i))


def debug(record, fig: Figure):

    fig.suptitle("Debug")
    t = record["t"]
    pos = record["debug"]


    f = fig.subplots(3, 1)

    for i in range(3):
        f[i].plot(t, np.abs(pos[:, 0, i, 0]))
        f[i].set_ylabel("{}".format(i))


def heading_acceleration(record, fig: Figure):
    fig.suptitle("s_hac")
    t = record["t"]
    pos = record["s_hac"][:, 0, 0, 0]

    f = fig.subplots(1, 1)
    f.plot(t, pos)


def debug_q(record, fig: Figure):

    fig.suptitle("Debug q")
    t = record["t"]
    pos = record["debug_q"][:,0,:,:]


    f = fig.subplots(7, 1)

    for i in range(4):
        f[i].plot(t, pos[:, i, 0])
        f[i].set_ylabel("{}".format(i))
    # atan2 2xw - 2yz, 1 - 2xx - 2zz
    X=pos[:,0,0]
    Y=pos[:,1,0]
    Z=pos[:,2,0]
    W=pos[:,3,0]

    yaw = np.arctan2(2*X*W - 2*Y*Z, 1 - 2*X*X - 2*Z*Z)
    f[4].plot(t,yaw)
    f[4].set_ylabel("yaw")

    pitch = np.arcsin(2*X*Y + 2*Z*W)
    f[5].plot(t,pitch)
    f[5].set_ylabel("pitch")

    roll = np.arctan2(2*Y*W - 2*X*Z, 1 - 2*Y*Y - 2*Z*Z)
    f[6].plot(t,roll)
    f[6].set_ylabel("roll")


def debug_xy_position(record, fig: Figure, vec_step=1, stat_window=20):
    fig.suptitle("XY position")
    pos = record["debug_loc"][:,0,0:2,0]
    q = record["debug_q"][:,0,:,:]
    t = record["t"]
    s_hac = record["s_hac"][:, 0, 0, 0]

    X=q[:,0,0]
    Y=q[:,1,0]
    Z=q[:,2,0]
    W=q[:,3,0]
    orn = np.arctan2(2*X*W - 2*Y*Z, 1 - 2*X*X - 2*Z*Z) + np.pi

    vel_vec = np.zeros(pos.shape)
    vel_vec[1:, :] = pos[1:, :] - pos[:-1, :]
    head_vec = np.zeros(pos.shape)
    head_vec[:, 0] = np.cos(orn)
    head_vec[:, 1] = np.sin(orn)

    f = fig.subplots(2, 2)
    figr_plane = f[0][0]

    origins = [pos[::vec_step, 0], pos[::vec_step, 1]]

    dirs = np.asarray([head_vec[::vec_step, 0], head_vec[::vec_step, 1]]) * 0.1
    vel_dirs = np.asarray([vel_vec[::vec_step, 0], vel_vec[::vec_step, 1]])
    figr_plane.set_title("Heading and velocity vectors")

    figr_plane.quiver(*origins, dirs[0,:], dirs[1,:], scale=1, color='b')
    figr_plane.quiver(*origins, vel_dirs[0, :], vel_dirs[1, :], scale=1, color='r')
    figr_plane.plot(pos[:,0], pos[:,1])
    figr_plane.plot(pos[:1,0], pos[:1,1], 'g.')
    figr_plane.set_xlabel("X")
    figr_plane.set_ylabel("Y")

    ## weighted vel
    figr_directed_vel = f[1][0]
    figr_directed_vel.set_title("Heading velocity magnitude")
    vel_dir = np.arctan2(vel_vec[:,1], vel_vec[:, 0])

    head_vel = np.sqrt(np.sum(np.square(vel_vec[:, 0:2]), axis=1)) * np.cos(vel_dir - orn)

    head_vel_mean = np.asarray([np.mean(head_vel[i-stat_window:i]) for i in range(stat_window, head_vel.shape[0])])
    head_vel_std = np.asarray([np.std(head_vel[i-stat_window:i]) for i in range(stat_window, head_vel.shape[0])])


    figr_directed_vel.plot(t, head_vel)
    figr_directed_vel.plot(t[stat_window:], head_vel_mean, 'r')
    figr_directed_vel.plot(t[stat_window:], head_vel_mean + head_vel_std, 'r--')
    figr_directed_vel.plot(t[stat_window:], head_vel_mean - head_vel_std, 'r--')
    figr_directed_vel.plot(t, s_hac * 10, 'k')

    ## weighted vel vis
    figr_plane_vel = f[0][1]
    figr_plane_vel.set_title("Heading velocity")
    figr_plane_vel.plot(pos[:, 0], pos[:, 1])
    mn_vel = np.mean(np.abs(head_vel))
    vr_vel = np.std(np.abs(head_vel))
    pos_head_vel = head_vel[head_vel > 0]
    neg_head_vel = np.abs(head_vel[head_vel < 0])
    pos_loc = pos[head_vel > 0, :]
    neg_loc = pos[head_vel < 0, :]
    figr_plane_vel.scatter(pos_loc[:, 0], pos_loc[:, 1], color='g', s=5*(pos_head_vel - mn_vel)/vr_vel)
    figr_plane_vel.scatter(neg_loc[:, 0], neg_loc[:, 1], color='r', s=5*(neg_head_vel - mn_vel)/vr_vel)
    figr_plane_vel.set_xlabel("X")
    figr_plane_vel.set_ylabel("Y")

    ## Heading velocity stats
    figr_stats = f[1][1]
    figr_stats.set_title("Heading velocity stats")
    figr_stats.boxplot(head_vel)


def provider_xy_position(record, fig: Figure, vec_step=100, stat_window=20):
    fig.suptitle("Provider XY position")
    pos_last = record["s_hac"][:, 0, 6:8, 0]
    pos = record["s_hac"][:, 0, 9:11, 0]
    vel = record["s_hac"][:, 0, 3:5, 0]
    q = record["s_hac"][:, 0, 12:,:]
    t = record["t"]
    s_hac = record["s_hac"][:, 0, 0, 0]

    X=q[:,0,0]
    Y=q[:,1,0]
    Z=q[:,2,0]
    W=q[:,3,0]
    orn = np.arctan2(2*X*W - 2*Y*Z, 1 - 2*X*X - 2*Z*Z) + np.pi

    vel_vec = vel
    vel_mag = np.sqrt(np.sum(np.square(vel_vec[:, 0:2]), axis=1))

    # vel_vec = p_vec
    head_vec = np.zeros(pos.shape)
    head_vec[:, 0] = np.cos(orn)
    head_vec[:, 1] = np.sin(orn)

    f = fig.subplots(2, 2)
    figr_plane = f[0][0]

    origins = [pos[::vec_step, 0], pos[::vec_step, 1]]

    dirs = np.asarray([head_vec[::vec_step, 0], head_vec[::vec_step, 1]]) * 0.1
    vel_dirs = np.asarray([vel_vec[::vec_step, 0], vel_vec[::vec_step, 1]]) / np.mean(vel_mag) * 0.05
    figr_plane.set_title("Heading and velocity vectors")

    figr_plane.quiver(*origins, dirs[0,:], dirs[1,:], scale=1, color='b')
    figr_plane.quiver(*origins, vel_dirs[0, :], vel_dirs[1, :], scale=1, color='r')
    figr_plane.plot(pos[:,0], pos[:,1])
    figr_plane.plot(pos[:1,0], pos[:1,1], 'g.')
    figr_plane.set_xlabel("X")
    figr_plane.set_ylabel("Y")

    ## weighted vel
    figr_directed_vel = f[1][0]
    figr_directed_vel.set_title("Heading velocity magnitude")
    vel_dir = np.arctan2(vel_vec[:,1], vel_vec[:, 0])

    head_vel = np.sqrt(np.sum(np.square(vel_vec[:, 0:2]), axis=1)) * np.cos(vel_dir - orn)

    head_vel_mean = np.asarray([np.mean(head_vel[i-stat_window:i]) for i in range(stat_window, head_vel.shape[0])])
    head_vel_std = np.asarray([np.std(head_vel[i-stat_window:i]) for i in range(stat_window, head_vel.shape[0])])


    figr_directed_vel.plot(t, head_vel)
    figr_directed_vel.plot(t[stat_window:], head_vel_mean, 'r')
    figr_directed_vel.plot(t[stat_window:], head_vel_mean + head_vel_std, 'r--')
    figr_directed_vel.plot(t[stat_window:], head_vel_mean - head_vel_std, 'r--')
    figr_directed_vel.plot(t, s_hac, 'k')

    ## weighted vel vis
    figr_plane_vel = f[0][1]
    figr_plane_vel.set_title("Heading velocity")
    figr_plane_vel.plot(pos[:, 0], pos[:, 1])
    mn_vel = np.mean(np.abs(head_vel))
    vr_vel = np.std(np.abs(head_vel))
    pos_head_vel = head_vel[head_vel > 0]
    neg_head_vel = np.abs(head_vel[head_vel < 0])
    pos_loc = pos[head_vel > 0, :]
    neg_loc = pos[head_vel < 0, :]
    figr_plane_vel.scatter(pos_loc[:, 0], pos_loc[:, 1], color='g', s=5*(pos_head_vel - mn_vel)/vr_vel)
    figr_plane_vel.scatter(neg_loc[:, 0], neg_loc[:, 1], color='r', s=5*(neg_head_vel - mn_vel)/vr_vel)
    figr_plane_vel.set_xlabel("X")
    figr_plane_vel.set_ylabel("Y")

    ## Heading velocity stats
    figr_stats = f[1][1]
    figr_stats.set_title("Heading velocity stats")
    figr_stats.boxplot(head_vel)
    # figr_stats.plot(t, pos_last[:, 0])
    # figr_stats.plot(t, pos[:, 0])
    # figr_stats.plot(t[0:-1], pos[1:, 0])


def u_leg_position(record, fig: Figure, ctx=0):
    fig.suptitle("Leg position phase-commands ctx:{} ".format(ctx))

    u_exp = record[CTR.MOTOR_EXPECTED][:, 0, :, 0, :, ctx]
    u_ctx = record[CTR.MOTOR_CONTEXT][:, 0, :, 0, :, ctx]
    u_out = u_ctx + u_exp

    legs = [["fl", "fr"], ["ml", "mr"], ["hl", "hr"]]
    T, N, C= u_exp.shape
    f = fig.subplots(3, 2)

    cmap = cm.get_cmap('Spectral')
    colors = [cmap(i * (1/C)) for i in range(C)]
    for r in range(3):
        for c in range(2):
            leg = legs[r][c]
            figr = f[r][c]
            figr.set_title(leg)
            coxa, femur = LEG_DICT[leg]
            pts = np.zeros((C + 1, 2))
            for ph in range(C):
                pts[ph,:] = u_out[-1, [coxa, femur], ph]
                if r == 2 and c == 1:
                    figr.plot(u_out[-1, coxa, ph], u_out[-1, femur, ph], '.', color=colors[ph], label="ph{}".format(ph), markersize=10)
                else:
                    figr.plot(u_out[-1, coxa, ph], u_out[-1, femur, ph], '.', color=colors[ph], markersize=10)
            pts[C, :] = u_out[-1, [coxa, femur], 0]
            figr.plot(pts[:, 0], pts[:, 1], '-k')
            if r == 2 and c == 1:
                figr.legend()


def regularization_global_budget(record, fig: Figure):
    fig.suptitle("Regularization: u_out Global Budget (set opt)")
    u_exp = record[CTR.MOTOR_OUTPUT][:, 0, :, 0, :, :]
    t = record["t"]
    budget = np.sum(np.square(u_exp), axis=(1,2))
    T, N, C, X = u_exp.shape
    f = fig.subplots(1, 1)
    for ctx in range(X):
        f.plot(t, budget[:, ctx], color=COLORS[ctx], label="ctx_{}".format(ctx), alpha=0.6)
    f.legend()


def regularization_leg_budget(record, fig: Figure, ctx=0):
    fig.suptitle("Regularization: u_out Leg Budget (set opt)")
    u_out = record[CTR.MOTOR_OUTPUT][:, 0, :, 0, :, ctx]
    t = record["t"]
    legs = ["fl", "fr", "ml", "mr", "hl", "hr"]
    f = fig.subplots(6, 1)
    for i, leg in enumerate(legs):
        budget = np.sum(np.square(u_out[:, LEG_DICT[leg], :]), axis=(1, 2))
        f[i].plot(t, budget)
        f[i].set_ylabel(leg)


def regularization_u_variance(record, fig: Figure):
    fig.suptitle("Regularization: u_out phase-motor variance sum (max)")
    t = record["t"]
    u_out = record[CTR.MOTOR_OUTPUT][:, 0, :, 0, :, :]
    T, N, C, X = u_out.shape
    u_out_varsum = np.sum(np.var(u_out, axis=2), axis=1)  # var by phase

    f = fig.subplots(1, 1)
    for ctx in range(X):
        f.plot(t, u_out_varsum[:, ctx], color=COLORS[ctx], label="ctx_{}".format(ctx), alpha=0.6)


def regularization_contalateral_symmetry(record, fig: Figure):
    fig.suptitle("Regularization: u_out contralateral symmetry (min)")
    t = record["t"]
    u_out = record[CTR.MOTOR_OUTPUT][:, 0, :, 0, :, :]
    T, N, C, X = u_out.shape
    half = C//2
    dif_sym_l = np.sum(np.square(u_out[:, [0,2,4,6,8,10], :half, :] - u_out[:, [1,3,5,7,9,11], half:, :]), axis=(1,2))
    dif_sym_r = np.sum(np.square(u_out[:, [0,2,4,6,8,10], half:, :] - u_out[:, [1,3,5,7,9,11], :half, :]), axis=(1,2))
    dif_sym = dif_sym_l + dif_sym_r

    f = fig.subplots(1, 1)
    for ctx in range(X):
        f.plot(t, dif_sym[:, ctx], color=COLORS[ctx], label="ctx_{}".format(ctx), alpha=0.6)


def regularization_neighbour_closeness(record, fig: Figure):
    fig.suptitle("Regularization: u_out phase-neighbourhood closeness (min)")
    t = record["t"]
    u_out = record[CTR.MOTOR_OUTPUT][:, 0, :, 0, :, :]
    T, N, C, X = u_out.shape

    u_out_cls = np.sum(np.square(u_out[:, :, 1:, :] - u_out[:, :, :-1, :]), axis=(1,2))
    u_out_cls += np.sum(np.square(u_out[:, :, -1, :] - u_out[:, :, 0, :]), axis=1)

    f = fig.subplots(1, 1)
    for ctx in range(X):
        f.plot(t, u_out_cls[:, ctx], color=COLORS[ctx], label="ctx_{}".format(ctx), alpha=0.6)


def heading_change_rate(record, fig: Figure):
    fig.suptitle("s_hch")
    t = record["t"]
    s_hch = record["s_hch"][:, 0, 0:3, 0]
    s_hch_raw = record["s_hch"][:, 0, 3:6, 0]
    heading_now = record["s_hch"][:, 0, 6:9, 0]
    heading_last = record["s_hch"][:, 0, 9:, 0]

    f = fig.subplots(4, 3)
    figr_hch_only = f[0]
    figr_hch = f[1]
    figr_heading = f[2]
    figr_deb = f[3]
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        figr_hch_only[i].set_title(name)
        figr_hch_only[i].plot(t, s_hch[:, i], label="s_hch")

        figr_hch[i].plot(t, s_hch[:, i], label="s_hch")
        figr_hch[i].plot(t, s_hch_raw[:, i], label="s_hch raw")

        figr_heading[i].plot(t, heading_now[:, i], label="heading now")
        figr_heading[i].plot(t, heading_last[:, i], label="heading last")



        dist = np.sin(heading_now - heading_last)

        figr_deb[i].plot(t, dist[:, i])

    figr_hch_only[2].legend()
    figr_hch[2].legend()
    figr_heading[2].legend()


def s_btq(record, fig: Figure):

    fig.suptitle("s_btq")
    t = record["t"]
    pos = record["s_btq"]


    f = fig.subplots(3, 1)

    for i in range(3):
        f[i].plot(t, np.abs(pos[:, 0, i, 0]))
        f[i].set_ylabel("{}".format(i))