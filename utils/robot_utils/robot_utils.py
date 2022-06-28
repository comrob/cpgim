import numpy as np
import time
# import robot.Robot as rob
rob = None

"""VREP hexapod model specific utils."""

MAX_ROBOT_TORQUE = 1023
MAX_SI_TORQUE = 3


def torque_si_to_ticks(si):
    return np.trunc((si / MAX_SI_TORQUE) * MAX_ROBOT_TORQUE).astype(np.int)


def ticks_to_angle(ticks):
    ret = (5 * np.pi / 3072) * ticks - 5 * np.pi / 6
    return ret


def angle_to_ticks(angles):
    return (angles + 5 * np.pi / 6)/(5 * np.pi / 3072)


def do_pose_command(ctr, robot, sleep_time=0.001):
    """

    :param ctr: ()
    :param robot: robot class with ".set_servo_position" method
    :param sleep_time: sleep after each command
    :return:
    """
    ctr = ticks_to_angle(ctr)
    for j in range(0, 18):
        robot.set_servo_position(j + 1, ctr[j])
    time.sleep(sleep_time)


def order_leg_contacts(leg_contact):
    if len(leg_contact) == 6: #TODO find out why?
        return [leg_contact[k] for k in [0, 3, 2, 5, 1, 4]]
    else:
        return leg_contact


def setup_torque_limits(robot, torque_limits):
    for j in range(1, 19):
        robot.set_servo_torque(j, torque_limits[j - 1])


def observe_joint_positions(robot, observed_joints):
    """

    :param robot:
    :param observed_joints: indexing starts from 0, non observed joints return 0 as default
    :return: numpy array of shape (18, )
    """
    arr = np.zeros((18, ))
    for j in observed_joints:
        arr[j] = angle_to_ticks(robot.get_servo_position(j+1))
    return arr


def get_command(coxa_femur_pos, thibia_pos):
    """

    :param coxa_femur_pos: positions of
    tlc, trc, tlt, trt, blc, brc, blt, brt, mlc, mrc, mlt, mrt
      0    1    2    3    4    5    6    7    8    9   10   11
    :param thibia_pos: positions of
    tli, tri, bli, bri, mli, mri
    :return: target positions:
    tlc, trc, tlt, trt, tli, tri, blc, brc, blt, brt, bli, bri, mlc, mrc, mlt, mrt, mli, mri
      0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17
    """
    trg_pos = np.zeros((18,))
    trg_pos[[4, 5, 10, 11, 16, 17]] = thibia_pos
    trg_pos[[0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]] = coxa_femur_pos[0, :]
    trg_pos[[1, 3, 5, 7, 9, 11, 13, 15, 17]] = 1000 - trg_pos[[1, 3, 5, 7, 9, 11, 13, 15, 17]]
    return trg_pos


def get_real_robot_position_command(coxa_femur_pos, thibia_pos):
    """

    :param coxa_femur_pos: positions of
    tlc, trc, tlt, trt, blc, brc, blt, brt, mlc, mrc, mlt, mrt
      0    1    2    3    4    5    6    7    8    9   10   11
    :param thibia_pos: positions of
    tli, tri, bli, bri, mli, mri
    :return: target positions:
    tlc, trc, tlt, trt, tli, tri, blc, brc, blt, brt, bli, bri, mlc, mrc, mlt, mrt, mli, mri
      0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17
    """
    trg_pos = np.zeros((18,))
    trg_pos[[4, 5, 10, 11, 16, 17]] = thibia_pos
    trg_pos[[0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]] = coxa_femur_pos[0, :]
    trg_pos[[0, 2, 4, 6, 8, 10, 12, 14, 16]] = 1000 - trg_pos[[0, 2, 4, 6, 8, 10, 12, 14, 16]]
    trg_pos[[2, 3, 8, 9, 14, 15]] = 1000 - trg_pos[[2, 3, 8, 9, 14, 15]]
    return trg_pos


def get_torque_command(coxa_femur_trq, thibia_trq):
    """

    :param coxa_femur_pos: positions of
    tlc, trc, tlt, trt, blc, brc, blt, brt, mlc, mrc, mlt, mrt
      0    1    2    3    4    5    6    7    8    9   10   11
    :param thibia_pos: positions of
    tli, tri, bli, bri, mli, mri
    :return: target positions:
    tlc, trc, tlt, trt, tli, tri, blc, brc, blt, brt, bli, bri, mlc, mrc, mlt, mrt, mli, mri
      0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17
    """
    trg_pos = np.zeros((18,))
    trg_pos[[4, 5, 10, 11, 16, 17]] = thibia_trq
    trg_pos[[0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]] = coxa_femur_trq
    return trg_pos


def set_target_torque_limits(robot, target_servos, torque_limits):
    for i in range(len(target_servos)):
        robot.set_joint_torque(target_servos[i] + 1, torque_limits[i])


def get_servos_position(servo_position_getter: callable):
    selected_servos = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]
    ret = angle_to_ticks(np.asarray([servo_position_getter(i + 1) for i in selected_servos]))
    ret[[1, 3, 5, 7, 9, 11]] = 1000 - ret[[1, 3, 5, 7, 9, 11]]
    return ret.reshape((1, 1, -1))


def get_all_servos_position(servo_position_getter: callable):
    return angle_to_ticks(np.asarray([servo_position_getter(i + 1) for i in range(0, 18)]))
