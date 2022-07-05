from utils.robot_utils.updatable_provider import UpdatableProvider, LEFT, RIGHT, FEMURS, COXAS
import numpy as np
from typing import Callable, Union
import queue

class GroundContactProvider(UpdatableProvider):
    def __init__(self):
        self._ground_contact_getter = None
        self._position_getter = None
        self._position_threshold = None
        self._simulated_ground_contact = True
        self._updater = self._update_threshold_contact
        self._state = None

    def set_ground_contact_getter(self, ground_contact_getter: Callable[[], list]):
        self._ground_contact_getter = ground_contact_getter

    def set_position_threshold_event(self, position_ground_threshold: np.ndarray):
        self._position_threshold = position_ground_threshold

    def set_position_getter(self, position_getter: Callable[[], np.ndarray]):
        self._position_getter = position_getter

    def _update_true_ground_contact(self):
        contacts = np.asarray(self._ground_contact_getter())
        self._state = np.concatenate([contacts[LEFT], contacts[RIGHT]])

    def _update_threshold_contact(self):
        pose = self._position_getter()
        fem_pose = pose[FEMURS]
        res = np.zeros(fem_pose.shape)
        res[[0, 2, 4]] = fem_pose[[0, 2, 4]] > self._position_threshold[[0, 2, 4]]
        res[[1, 3, 5]] = fem_pose[[1, 3, 5]] < self._position_threshold[[1, 3, 5]]
        self._state = res

    def set_simulated_ground_contact(self, is_simulated=True):
        if is_simulated:
            self._updater = self._update_threshold_contact
            self._simulated_ground_contact = True
        else:
            self._updater = self._update_true_ground_contact
            self._simulated_ground_contact = False

    def update(self):
        self._updater()

    def __call__(self) -> np.ndarray:
        return self._state


class CoxaPositionEventProvider(UpdatableProvider):
    def __init__(self):
        self._coxa_threshold = None
        self._position_getter = None
        self._comparator = None
        self._state = None

    def set_coxa_event_threshold(self, threshold: np.ndarray, comparator=np.less):
        self._coxa_threshold = threshold
        self._comparator = comparator

    def set_position_getter(self, position_getter: Callable[[], np.ndarray]):
        self._position_getter = position_getter

    def update(self):
        res = self._position_getter()[COXAS]
        res = self._comparator(res, self._coxa_threshold)
        res[[1, 3, 5]] = ~res[[1, 3, 5]] #FIXME is this even right? the position getter should have normed directions
        self._state = np.concatenate([res[LEFT], res[RIGHT]])

    def __call__(self) -> np.ndarray:
        return self._state


class FemurSpeedEventProvider(UpdatableProvider):
    def __init__(self):
        self._state = None
        self._comparator = None
        self._femur_threshold = None
        self._speed_getter = None

    def set_femur_event_threshold(self, threshold: Union[np.ndarray, float], comparator=np.greater):
        self._femur_threshold = threshold
        self._comparator = comparator

    def set_speed_getter(self, speed_getter: Callable[[], np.ndarray]):
        self._speed_getter = speed_getter

    def update(self):
        vels = self._speed_getter()[FEMURS]
        vels[[1, 3, 5]] = -vels[[1, 3, 5]]
        res = self._comparator(vels, self._femur_threshold)
        self._state = np.concatenate([res[LEFT], res[RIGHT]])

    def __call__(self) -> np.ndarray:
        return self._state


class CoxaSpeedStopEventProvider(UpdatableProvider):
    def __init__(self):
        self._state = None
        self._comparator = None
        self._coxa_threshold = None
        self._speed_getter = None

    def set_coxa_event_threshold(self, threshold: Union[np.ndarray, float], comparator=np.less):
        self._coxa_threshold = threshold
        self._comparator = comparator

    def set_speed_getter(self, speed_getter: Callable[[], np.ndarray]):
        self._speed_getter = speed_getter

    def update(self):
        vels = self._speed_getter()[COXAS]
        vels[[1, 3, 5]] = -vels[[1, 3, 5]]
        res = self._comparator(vels, self._coxa_threshold)
        self._state = np.concatenate([res[LEFT], res[RIGHT]])

    def __call__(self) -> np.ndarray:
        return self._state


class FemurErrorSpikeEventProvider(UpdatableProvider):
    def __init__(self, first_empty=True):
        self._state = None
        self._comparator = None
        self._femur_threshold = None
        self._pose_sensed_getter = None
        self._pose_commanded_getter = None
        if first_empty:
            self._updater = self._first_update
        else:
            self._updater = self._next_update

    def set_femur_event_threshold(self, threshold: Union[np.ndarray, float], comparator=np.greater):
        self._femur_threshold = threshold
        self._comparator = comparator

    def set_position_sensed_getter(self, getter: Callable[[], np.ndarray]):
        self._pose_sensed_getter = getter

    def set_position_command_getter(self, getter: Callable[[], np.ndarray]):
        self._pose_commanded_getter = getter

    def _first_update(self):
        self._updater = self._next_update
        self._state = np.zeros((6, ))

    def _next_update(self):
        errors = np.abs(self._pose_commanded_getter()[0, FEMURS] - self._pose_sensed_getter()[0, 0, FEMURS])
        # errors[[1, 3, 5]] = -vels[[1, 3, 5]]
        res = self._comparator(errors, self._femur_threshold)
        self._state = np.concatenate([res[LEFT], res[RIGHT]])

    def update(self):
        self._updater()

    def __call__(self) -> np.ndarray:
        return self._state


class SmoothedVelocityEstimationProvider(UpdatableProvider):
    def __init__(self, first_empty=True):
        self._state = None
        self._prev_pose = np.zeros((12, ))
        self._getter = None
        self._smoother = None
        if first_empty:
            self._updater = self._first_update
        else:
            self._updater = self._next_update

    def set_pose_getter(self, getter: Callable[[], np.ndarray]):
        self._getter = getter

    def set_smoothing_processor(self, smoother: Callable[[np.ndarray], np.ndarray]):
        self._smoother = smoother

    def _first_update(self):
        self._updater = self._next_update
        self._state = np.zeros((12, ))

    def _next_update(self):
        pose = self._getter()
        self._state = self._smoother(pose - self._prev_pose)
        self._prev_pose = pose

    def update(self):
        self._updater()

    def __call__(self) -> np.ndarray:
        return self._state.reshape((1, 1, -1))


class AccelerationProvider(UpdatableProvider):
    EXPECTED_SHAPE = (1, 3, 1)

    def __init__(self, first_empty=True):
        self._state = None
        self._getter = None
        self._smoother = lambda x: x
        if first_empty:
            self._updater = self._first_update
        else:
            self._updater = self._next_update

    def set_smoothing_processor(self, smoother: Callable[[np.ndarray], np.ndarray]):
        self._smoother = smoother

    def set_data_getter(self, getter: Callable[[], list]):
        self._getter = getter

    def _first_update(self):
        self._updater = self._next_update
        self._state = np.zeros(self.EXPECTED_SHAPE)

    def _next_update(self):
        val = self._getter()
        if val is not None:
            self._state = self._smoother(np.asarray(val).reshape(self.EXPECTED_SHAPE))

    def update(self):
        self._updater()

    def __call__(self) -> np.ndarray:
        return self._state


class YAxisAccTrqProvider(UpdatableProvider):
    EXPECTED_SHAPE = (1, 2, 1)

    def __init__(self, first_empty=True):
        self._state = None
        self._getter = None
        self._smoother = lambda x: x
        if first_empty:
            self._updater = self._first_update
        else:
            self._updater = self._next_update

    def set_smoothing_processor(self, smoother: Callable[[np.ndarray], np.ndarray]):
        self._smoother = smoother

    def set_data_getter(self, getter: Callable[[], list]):
        self._getter = getter

    def _first_update(self):
        self._updater = self._next_update
        self._state = np.zeros(self.EXPECTED_SHAPE)

    def _next_update(self):
        val = self._getter()
        if val is not None:
            # val[1] *= 1
            self._state = self._smoother(np.asarray(val).reshape(self.EXPECTED_SHAPE))

    def update(self):
        self._updater()

    def __call__(self) -> np.ndarray:
        return self._state


class GenericProvider(UpdatableProvider):

    def __init__(self, expected_shape, first_empty=True):
        self._state = None
        self._getter = None
        self._smoother = lambda x: x
        self.expected_shape = expected_shape
        if first_empty:
            self._updater = self._first_update
        else:
            self._updater = self._next_update

    def set_smoothing_processor(self, smoother: Callable[[np.ndarray], np.ndarray]):
        self._smoother = smoother

    def set_data_getter(self, getter: Callable[[], list]):
        self._getter = getter

    def _first_update(self):
        self._updater = self._next_update
        self._state = np.zeros(self.expected_shape)

    def _next_update(self):
        val = self._getter()
        if val is not None:
            self._state = self._smoother(np.asarray(val).reshape(self.expected_shape))

    def update(self):
        self._updater()

    def __call__(self) -> np.ndarray:
        return self._state


class OrientedPositionDisplacementRate(UpdatableProvider):
    # heading_magnitude - 0
    # heading_direction - 1
    # velocity_direction - 2
    # velocity_vector - 3-5
    # position_previous - 6-8
    # position_current - 9-11
    # orientation_quaternion - 12-15

    EXPECTED_SHAPE = (1, 16, 1)

    def __init__(self, first_empty=True):
        self._state = None
        self._getter = None
        self._smoother = lambda x: x
        self._orientation_i = 0

        if first_empty:
            self._updater = self._first_update
        else:
            self._updater = self._next_update

    def set_smoothing_processor(self, smoother: Callable[[np.ndarray], np.ndarray]):
        self._smoother = smoother

    def set_data_getter(self, getter: Callable[[], list]):
        self._getter = getter

    def _update_mean_heading(self, orientation):
        # self._orientations[self._orientation_i, :] = orientation
        # self._orientation_i += 1
        # self._orientation_i %= self._orientation_window
        q = orientation

        """"
            X=q[:,0,0]
            Y=q[:,1,0]
            Z=q[:,2,0]
            W=q[:,3,0]
            orn = np.arctan2(2*X*W - 2*Y*Z, 1 - 2*X*X - 2*Z*Z) + np.pi
        """

        # arctan2(2XW-2YZ, 1-2XX-2ZZ)
        head = np.arctan2(2 * q[0] * q[3] - 2 * q[1] * q[2], 1 - 2 * q[0] * q[0] - 2 * q[2] * q[2]) + np.pi

        return head

    def _update_velocity_vector(self, new_position):
        #vel_vec[1:, :] = pos[1:, :] - pos[:-1, :]
        new_velocity = new_position - self.last_position
        self.last_position = new_position
        return new_velocity

    def _first_update(self):
        self._updater = self._next_update
        position, orientation = self._getter()
        self.last_position = np.asarray([0, 0, 0])

        self._state = np.zeros(self.EXPECTED_SHAPE)

    def _next_update(self):
        position, orientation = self._getter()

        if position is not None and orientation is not None:
            tmp = self.last_position
            vel_vect = self._update_velocity_vector(np.asarray(position))
            head = self._update_mean_heading(orientation)
            vel_dir = np.arctan2(vel_vect[1], vel_vect[0])

            a = np.sqrt(np.sum(np.square(vel_vect[:2]))) * np.cos(vel_dir - head)
            _a = self._smoother(np.asarray([[a]]))[0]

            state = [[_a], [head], [vel_dir], vel_vect, tmp, position, orientation]


            self._state = np.concatenate(state).reshape(self.EXPECTED_SHAPE)


    def update(self):
        self._updater()

    def __call__(self) -> np.ndarray:
        return self._state


class HeadingChangeRate(UpdatableProvider):
    # pitch change rate smoothed - 0
    # roll change rate smoothed - 1
    # yaw change rate smoothed - 2
    # pitch change rate raw- 3
    # roll change rate raw- 4
    # yaw change rate raw- 5
    # current pitch - 6
    # current roll - 7
    # current yaw - 8
    # last heading - 9
    # last heading - 10
    # last heading - 11

    EXPECTED_SHAPE = (1, 12, 1)

    def __init__(self, first_empty=True):
        self._state = None
        self._getter = None
        self._smoother = lambda x: x
        self._orientation_i = 0

        if first_empty:
            self._updater = self._first_update
        else:
            self._updater = self._next_update

    def set_smoothing_processor(self, smoother: Callable[[np.ndarray], np.ndarray]):
        self._smoother = smoother

    def set_data_getter(self, getter: Callable[[], list]):
        self._getter = getter

    def _update_heading_change(self, orientation):
        q = orientation

        """"
            X=q[:,0,0]
            Y=q[:,1,0]
            Z=q[:,2,0]
            W=q[:,3,0]
            orn = np.arctan2(2*X*W - 2*Y*Z, 1 - 2*X*X - 2*Z*Z) + np.pi
        """

        # arcsin(2XY + 2ZW)
        pitch = np.arcsin(2*q[0]*q[1] + 2*q[2]*q[3])

        # arctan2(2YW-2XZ, 1-2YY-2ZZ)
        roll = np.arctan2(2*q[1]*q[3] - 2*q[0]*q[2], 1 - 2*q[1]*q[1] - 2*q[2]*q[2])

        # arctan2(2XW-2YZ, 1-2XX-2ZZ)
        yaw = np.arctan2(2 * q[0] * q[3] - 2 * q[1] * q[2], 1 - 2 * q[0] * q[0] - 2 * q[2] * q[2]) + np.pi

        heading = np.asarray([roll, pitch, yaw])
        # new_heading_change = np.sqrt(np.square(np.sin(head) - np.sin(self.last_heading)) + np.square(np.cos(head) - np.cos(self.last_heading)))
        new_heading_change = np.sin(heading - self.last_heading)

        self.last_heading = heading

        return new_heading_change

    def _first_update(self):
        self._updater = self._next_update
        orientation = self._getter()
        self.last_heading = np.zeros((3,))
        self._state = np.zeros(self.EXPECTED_SHAPE)

    def _next_update(self):
        orientation = self._getter()

        if orientation is not None:
            tmp = self.last_heading
            _head_chng = self._update_heading_change(orientation)
            head_chng = self._smoother(_head_chng.reshape((1,3)))
            state = [head_chng, _head_chng, self.last_heading, tmp]
            self._state = np.concatenate(state).reshape(self.EXPECTED_SHAPE)

    def update(self):
        self._updater()

    def __call__(self) -> np.ndarray:
        return self._state