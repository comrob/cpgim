from dynsys_framework.execution_helpers.model_executor_parameters import ModelExecutorParameters
import numpy as np
_D = "d_"
CURRENT_COMMAND = "velhead_curr"
COMMAND = "velhead"
MEAN_COMMAND = "velhead_mean"
INT_COMMAND = "velhead_int"
DER_COMMAND = "velhead_der"
INT_ON = "velhead_int_on"

TARGET_POSITION = "coord_trg"
CURRENT_POSITION = "position_curr"


def model(mep: ModelExecutorParameters, current_position_variable=CURRENT_POSITION,
          target_coordinates_variable=TARGET_POSITION, integration_reset_variable=INT_ON):
    mep.add(CURRENT_COMMAND, (current_position_variable, target_coordinates_variable), differential_control)

    mep.add(_D + MEAN_COMMAND, (MEAN_COMMAND, CURRENT_COMMAND), get_d_command_mean(1.),
            default_initial_values={MEAN_COMMAND: np.zeros((2, ))})

    mep.add(_D + INT_COMMAND, (INT_COMMAND, CURRENT_COMMAND, integration_reset_variable), get_d_command_integration(0.01, 0.001),
            default_initial_values={INT_COMMAND: np.zeros((2, ))})

    mep.add(DER_COMMAND, (MEAN_COMMAND, CURRENT_COMMAND), get_command_derivative_expr())

    mep.add(COMMAND, (CURRENT_COMMAND, INT_COMMAND, DER_COMMAND), get_command_pid(1.,1.,1.))


def differential_control(curr_position, trg_coord):
    """

    :param curr_position: (x,y,heading)
    :param trg_coord: (x, y)
    :return: target (velocity, heading)
    """
    coord_diff = trg_coord - curr_position[:2]
    direction = np.arctan2(coord_diff[1], coord_diff[0])
    direction_diff = direction - curr_position[2]
    velocity = np.maximum(np.sqrt(np.sum(np.square(coord_diff))) * np.cos(direction_diff), 0)
    angular_velocity = np.mod(direction_diff, 2 * np.pi)
    if angular_velocity > np.pi:
        angular_velocity = - 2 * np.pi + angular_velocity
    return np.asarray([velocity, angular_velocity])


def get_d_command_mean(adaptation_rate):
    def exprs(command, curr_command):
        return (curr_command - command) * adaptation_rate
    return exprs


def get_command_derivative_expr():
    def exprs(command_mean, curr_command):
        return curr_command - command_mean
    return exprs


def get_d_command_integration(adaptation_rate, decay):
    def exprs(integration, curr_command, int_on):
        return curr_command * adaptation_rate * int_on - integration * decay
    return exprs


def get_command_proportional(proportional_gain):
    def exprs(curr_command):
        return curr_command * proportional_gain
    return exprs


def get_command_pid(proportional_gain, integration_gain, derivative_gain):
    def exprs(curr_command, command_integration, command_derivative):
        return curr_command * proportional_gain + command_integration * integration_gain + command_derivative * derivative_gain
    return exprs


def get_command_transformation(num_state_centers, vel_min_max=(-1, 1), ang_min_max=(-0.5, 0.5),
                               pitch_weight=0.001, roll_weight=0.001):
    """
    Transformation of (velocity, heading) command into goal setup.
    The goal is a tensor of shape: (1, sensor_dim, 1, state_centers_num, configurations)
    where the expected sensor_dim are [heading_velocity, roll, pitch, current_heading]
    and configurations are [reference, mask, positive_negative_error_mask].

    :param num_state_centers:
    :param vel_min_max:
    :param ang_min_max:
    :param pitch_weight:
    :param roll_weight:
    :return:
    """
    def exprs(_velhead):
        y_goal = np.zeros((1, 4, 1, num_state_centers, 3))

        # Velocity setup
        y_goal[0, 0, 0, :, 0] = np.clip(_velhead[0], a_min=vel_min_max[0], a_max=vel_min_max[1])
        y_goal[0, 0, 0, :, 1] = 1
        y_goal[0, 0, 0, :, 2] = np.sign(_velhead[0])
        # Angular velocity setup
        y_goal[0, 3, 0, :, 0] = np.clip(_velhead[1], a_min=ang_min_max[0], a_max=ang_min_max[1])
        y_goal[0, 3, 0, :, 1] = 1
        y_goal[0, 3, 0, :, 2] = np.sign(_velhead[1])

        # pitch zero setup
        y_goal[0, 1, 0, :, 0] = 0
        y_goal[0, 1, 0, :, 1] = pitch_weight
        y_goal[0, 1, 0, :, 2] = 1  # full error
        # roll zero setup
        y_goal[0, 2, 0, :, 0] = 0
        y_goal[0, 2, 0, :, 1] = roll_weight
        y_goal[0, 2, 0, :, 2] = 0  # full error
        #
        return y_goal

    return exprs


if __name__ == '__main__':
    print(differential_control(np.asarray([0, 0, 0]), np.asarray([1, 1])))
    print(differential_control(np.asarray([0, 0, np.pi/4]), np.asarray([1, 1])))
    print(differential_control(np.asarray([0, 0, 0]), np.asarray([-1, 1])))