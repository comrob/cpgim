from typing import Union, Callable, Tuple, List
import numpy as np

CommandLoader = Callable[[np.ndarray, np.ndarray], None]
RawCommandFilter = Callable[[np.ndarray, np.ndarray], np.ndarray]
RobotJointApi = Callable[[int, object], object]


class Commander:
    def __init__(self, robot_joint_api: RobotJointApi, filter_strategy: RawCommandFilter):
        self._command_loader = self.get_command_loader(robot_joint_api)
        self._last_command = None
        self._filter_strategy = filter_strategy
        self._caller = self._first_call
        self._command_size = None
        self._indexes = None

    def _first_call(self, command: np.ndarray):
        assert command.ndim == 1, "command has dim {}, but should be one".format(command.ndim)

        # settin up parameters
        self._command_size = command.shape[-1]
        self._indexes = np.arange(self._command_size)
        self._last_command = command

        # trying filter strategy
        _filter_check = self._filter_strategy(command, self._last_command)
        assert _filter_check.ndim == 1, "filter must return array of dim 1"
        assert _filter_check.shape[0] == self._command_size, \
            "filter returns wrong size of filter: expected {}, actual{}".format(
                self._command_size, _filter_check.shape[0])
        assert type(_filter_check[0]) is np.bool_, "filter should return bool array"

        # sending command
        self._command_loader(command, self._indexes)

        # switching caller
        self._caller = self._following_call

    def _following_call(self, command: np.ndarray):
        selection = self._filter_strategy(command, self._last_command)
        self._last_command[selection] = command[selection]
        self._command_loader(command[selection], self._indexes[selection])

    def __call__(self, command: np.ndarray):
        self._caller(command)

    @classmethod
    def get_command_loader(cls, robot_joint_api: RobotJointApi) -> CommandLoader:
        def command_loader(command: np.ndarray, joint_indexes: np.ndarray):
            for i in range(command.shape[-1]):
                robot_joint_api(joint_indexes[i] + 1, command[i])
        return command_loader


def get_filter_similar_commands(similarity_threshold: Union[float, np.ndarray]) -> RawCommandFilter:
    def filt(command: np.ndarray, last_command: np.ndarray) -> np.ndarray:
        return np.abs(command - last_command) > similarity_threshold
    return filt
