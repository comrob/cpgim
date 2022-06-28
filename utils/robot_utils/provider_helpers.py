import numpy as np

def _ndim4_assign(target_array, row, index):
    target_array[index, :, :, :] = row


def _ndim3_assign(target_array, row, index):
    target_array[index, :, :] = row


def _ndim2_assign(target_array, row, index):
    target_array[index, :] = row


class WindowMean:
    def __init__(self, init_window: np.ndarray):
        self._window = init_window
        self._window_size = init_window.shape[0]
        self._i = 0
        self._ndim = init_window.ndim
        if self._ndim == 2:
            self._assigner = _ndim2_assign
        elif self._ndim == 3:
            self._assigner = _ndim3_assign
        elif self._ndim == 4:
            self._assigner = _ndim4_assign
        else:
            raise NotImplementedError("ERROR: not implemented for ndim {}".format(self._ndim))

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        self._assigner(self._window, vector, self._i)
        ret = np.mean(self._window, axis=0)
        self._i += 1
        if self._i == self._window_size:
            self._i = 0
        return ret


class LeakySum:
    def __init__(self, init_sum: np.ndarray, leak_constant: float):
        assert 0 < leak_constant < 1, "leak constant must be in (0,1) interval"
        self._current_sum = init_sum
        self._leak_constant = leak_constant

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        self._current_sum += vector
        self._current_sum *= self._leak_constant
        return self._current_sum
