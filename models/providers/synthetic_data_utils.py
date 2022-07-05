from typing import Union, Tuple, Optional, Dict, List
import numpy as np


class Linear1DFuzzySegments(object):
    SCALE = 0
    VARIANCE = 1

    def __init__(self, x_segment_starts: List[float], segment_parameters: List[Tuple[float, float]], no_variance=True):
        assert len(x_segment_starts) == len(segment_parameters) - 1, "there must be parameter for each segment"
        assert all([x_segment_starts[i - 1] < x_segment_starts[i] for i in range(1, len(x_segment_starts))]), "segment starts must be sorted"
        self.x_segment_starts = x_segment_starts
        self.segment_parameters = segment_parameters
        self.no_variance = no_variance
        self.y_starts = self.get_y_starts(x_segment_starts, segment_parameters)

    @staticmethod
    def get_y_starts(x_segment_starts, segment_parameters):
        y_starts = [0]
        for i, x_start in enumerate(x_segment_starts):
            if i == 0:
                _x = x_start
            else:
                _x = x_start - x_segment_starts[i-1]
            v = _x * segment_parameters[i][Linear1DFuzzySegments.SCALE] + y_starts[-1]
            y_starts.append(v)
        return y_starts

    def _get_segment_id(self, u) -> int:
        ret = 0
        for seg_start in self.x_segment_starts:
            if u < seg_start:
                return ret
            ret += 1
        return ret

    def __call__(self, x):
        segment_id = self._get_segment_id(x)
        if segment_id == 0:
            _x = x
        else:
            _x = x - self.x_segment_starts[segment_id-1]
        y = _x * self.segment_parameters[segment_id][self.SCALE] + self.y_starts[segment_id]
        if not self.no_variance:
            y += np.random.normal(0, self.segment_parameters[segment_id][self.VARIANCE])
        return y